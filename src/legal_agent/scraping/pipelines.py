"""Scrapy pipelines: PDF conversion, hierarchical chunking, Qdrant upsert."""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Any
import json
import torch
from openai import OpenAI
from transformers import AutoTokenizer
from legal_agent.config import get_settings
from legal_agent.workflow.prompts import _ENRICHMENT_FEW_SHOT, _POLICY_ENRICHMENT_FEW_SHOT
from legal_agent.scraping.items import ChunkedRegulationItem, RegulatoryDocumentItem
from qdrant_client.models import PointStruct
from legal_agent.utils.models import embed_texts, compute_vectors
from docling.document_converter import DocumentConverter
from legal_agent.config import get_settings
from legal_agent.db.client import client_from_settings

logger = logging.getLogger(__name__)

_QDRANT_BATCH_SIZE = 32
# ---------------------------------------------------------------------------
# Pipeline 1 – Convert PDF to Markdown via Docling
# ---------------------------------------------------------------------------
class DoclingPdfPipeline:
    """Convert PDF items to Markdown using Docling; pass through non-PDF items."""

    def process_item(
        self, item: RegulatoryDocumentItem, spider: Any
    ) -> RegulatoryDocumentItem:
        if not item.get("is_pdf") or not item.get("raw_pdf_bytes"):
            return item

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(item["raw_pdf_bytes"])
                tmp_path = Path(tmp.name)

            converter = DocumentConverter()
            result = converter.convert(str(tmp_path))
            item["full_text"] = result.document.export_to_markdown()
            item["raw_pdf_bytes"] = b""
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Docling PDF conversion failed for %s", item["source_url"], exc_info=True)

        return item


# ---------------------------------------------------------------------------
# Pipeline 2 – Hierarchical chunking
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[;:])\s+|(?<=,)\s+(?=(?:and|or|but)\b)", re.IGNORECASE)


class ChunkingPipeline:
    """Split full_text into token-aware hierarchical chunks with linked chunk metadata."""

    def __init__(self, tokenizer_model: str, max_tokens: int = 512):
        self.tokenizer_model = tokenizer_model
        self.max_tokens = max_tokens
        self.tokenizer = None

    @classmethod
    def from_crawler(cls, crawler):
        settings = get_settings()
        return cls(
            tokenizer_model=settings.chunking_tokenizer_model,
            max_tokens=settings.chunk_max_tokens,
        )

    def open_spider(self, spider):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, use_fast=True)

    def process_item(
        self, item: RegulatoryDocumentItem, spider: Any
    ) -> list[ChunkedRegulationItem]:
        if not isinstance(item, RegulatoryDocumentItem):
            return item

        text: str = item.get("full_text", "").strip()
        if not text:
            return []

        normalized_text = self._normalize_text(text)
        document_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        document_id = hashlib.sha256(
            f"{item['source_url']}:{document_hash}".encode("utf-8")
        ).hexdigest()[:32]

        sections = self._split_by_headings(text)
        raw_chunks: list[dict[str, Any]] = []

        for header_path, section_text in sections:
            for chunk_text in self._split_large(section_text):
                if not chunk_text.strip():
                    continue
                raw_chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "header_path": header_path,
                        "token_count": self._count_tokens(chunk_text),
                    }
                )

        chunk_count = len(raw_chunks)
        chunks: list[ChunkedRegulationItem] = []

        for idx, raw in enumerate(raw_chunks):
            chunk_id = f"{document_id}:{idx}"
            prev_chunk_id = f"{document_id}:{idx - 1}" if idx > 0 else ""
            next_chunk_id = f"{document_id}:{idx + 1}" if idx < chunk_count - 1 else ""

            chunks.append(
                ChunkedRegulationItem(
                    text=raw["text"],
                    header_path=raw["header_path"],
                    source_url=item["source_url"],
                    jurisdiction=item["jurisdiction"],
                    effective_date=item.get("effective_date", ""),
                    is_processed=False,
                    document_id=document_id,
                    document_hash=document_hash,
                    chunk_id=chunk_id,
                    prev_chunk_id=prev_chunk_id,
                    next_chunk_id=next_chunk_id,
                    chunk_index=idx,
                    chunk_count=chunk_count,
                    token_count=raw["token_count"],
                    topic_tags=item.get("topic_tags", []),
                    compliance_domain=item.get("compliance_domain", ""),
                    applies_to_departments=item.get("applies_to_departments", []),
                    obligation_type=item.get("obligation_type", ""),
                )
            )

        logger.info("Chunked '%s' into %d linked pieces.", item.get("title", ""), len(chunks))
        return chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _split_by_headings(text: str) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        heading_stack: list[str] = []
        current_level = 0
        last_pos = 0

        for m in _HEADING_RE.finditer(text):
            level = len(m.group(1))
            title = m.group(2).strip()

            if last_pos < m.start():
                body = text[last_pos:m.start()].strip()
                if body:
                    path = " > ".join(heading_stack) if heading_stack else "Preamble"
                    parts.append((path, body))

            if level > current_level:
                heading_stack.append(title)
            else:
                heading_stack = heading_stack[: level - 1] + [title]

            current_level = level
            last_pos = m.end()

        trailing = text[last_pos:].strip()
        if trailing:
            path = " > ".join(heading_stack) if heading_stack else "Document"
            parts.append((path, trailing))

        if not parts:
            parts.append(("Document", text.strip()))

        return parts

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_large(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        if self._count_tokens(text) <= self.max_tokens:
            return [text]

        units = self._split_into_units(text)
        chunks: list[str] = []
        current_parts: list[str] = []

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            if self._count_tokens(unit) > self.max_tokens:
                if current_parts:
                    chunks.append("\n".join(current_parts).strip())
                    current_parts = []
                chunks.extend(self._split_oversized_unit(unit))
                continue

            candidate = "\n".join(current_parts + [unit]).strip() if current_parts else unit
            if self._count_tokens(candidate) <= self.max_tokens:
                current_parts.append(unit)
            else:
                chunks.append("\n".join(current_parts).strip())
                current_parts = [unit]

        if current_parts:
            chunks.append("\n".join(current_parts).strip())

        return chunks

    def _split_into_units(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units: list[str] = []

        for paragraph in paragraphs:
            if self._count_tokens(paragraph) <= self.max_tokens:
                units.append(paragraph)
                continue

            sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
            if len(sentences) == 1:
                units.append(paragraph)
            else:
                units.extend(sentences)

        return units

    def _split_oversized_unit(self, text: str) -> list[str]:
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
        if len(clauses) > 1:
            chunks: list[str] = []
            current_parts: list[str] = []

            for clause in clauses:
                candidate = " ".join(current_parts + [clause]).strip() if current_parts else clause
                if self._count_tokens(candidate) <= self.max_tokens:
                    current_parts.append(clause)
                else:
                    if current_parts:
                        chunks.append(" ".join(current_parts).strip())
                    if self._count_tokens(clause) <= self.max_tokens:
                        current_parts = [clause]
                    else:
                        chunks.extend(self._hard_split_tokens(clause))
                        current_parts = []

            if current_parts:
                chunks.append(" ".join(current_parts).strip())

            return chunks

        return self._hard_split_tokens(text)

    def _hard_split_tokens(self, text: str) -> list[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: list[str] = []

        for start in range(0, len(token_ids), self.max_tokens):
            window = token_ids[start:start + self.max_tokens]
            if not window:
                continue
            chunks.append(self.tokenizer.decode(window, skip_special_tokens=True).strip())

        return [chunk for chunk in chunks if chunk]


# ---------------------------------------------------------------------------
# Pipeline 3 – Metadata enrichment
# ---------------------------------------------------------------------------
_COLLECTION_REGULATION = "regulation"
_COLLECTION_POLICY = "policy"
 
 
class MetadataEnrichmentPipeline:
    """Use a legal SLM or OpenAI to extract normalised metadata from each item.
 
    The ``collection`` parameter selects which prompt and output schema to use:
 
        collection="regulation"  (default — used by the Scrapy pipeline)
            Prompt  : _ENRICHMENT_FEW_SHOT
            Outputs : topic_tags, compliance_domain,
                      applies_to_departments (list), obligation_type
 
        collection="policy"  (used by loader.py for internal policy PDFs)
            Prompt  : _POLICY_ENRICHMENT_FEW_SHOT
            Outputs : topic_tags, compliance_domain,
                      department (scalar string), obligation_type
                      (policy_id is also returned by the LLM but intentionally
                       ignored — it is not stored in the Qdrant payload)
 
    Scrapy always instantiates pipelines via from_crawler(), which hard-codes
    collection="regulation", so existing spider behaviour is 100% unchanged.
    """
 
    def __init__(self, collection: str = _COLLECTION_REGULATION) -> None:
        if collection not in (_COLLECTION_REGULATION, _COLLECTION_POLICY):
            raise ValueError(
                f"collection must be {_COLLECTION_REGULATION!r} or "
                f"{_COLLECTION_POLICY!r}, got {collection!r}"
            )
        self._collection = collection
        self._tokenizer = None
        self._model = None
        self._settings = None
 
    @classmethod
    def from_crawler(cls, crawler):
        # Scrapy calls this instead of __init__ when ITEM_PIPELINES is used.
        # Always "regulation" so the spider pipeline is unchanged.
        return cls(collection=_COLLECTION_REGULATION)
 
    def open_spider(self, spider):
        self._settings = get_settings()
        if self._settings.use_legal_slm:
            self._tokenizer, self._model = get_legal_slm(
                self._settings.legal_slm_model,
                self._settings.legal_slm_device,
                self._settings.legal_slm_load_in_4bit,
            )
 
    def process_item(self, item, spider):
        if isinstance(item, RegulatoryDocumentItem):
            return self._enrich_document(item)
        return item
 
    # ------------------------------------------------------------------
    # Prompt + defaults selection
    # ------------------------------------------------------------------
 
    def _get_prompt(self) -> str:
        if self._collection == _COLLECTION_POLICY:
            return _POLICY_ENRICHMENT_FEW_SHOT
        return _ENRICHMENT_FEW_SHOT
 
    def _get_defaults(self) -> dict:
        """Return safe fallback values matching the schema for this collection."""
        if self._collection == _COLLECTION_POLICY:
            return {
                "topic_tags": [],
                "compliance_domain": "",
                "department": "General",    # scalar – matches _POLICY_ENRICHMENT_FEW_SHOT
                "obligation_type": "",
            }
        return {
            "topic_tags": [],
            "compliance_domain": "",
            "applies_to_departments": [],   # list – matches _ENRICHMENT_FEW_SHOT
            "obligation_type": "",
        }
 
    # ------------------------------------------------------------------
    # Document-level enrichment (called by process_item for Scrapy items)
    # ------------------------------------------------------------------
 
    def _enrich_document(self, item: RegulatoryDocumentItem) -> RegulatoryDocumentItem:
        full_text = item.get("full_text", "")
        if not full_text:
            return item
        text = self._prepare_enrichment_text(full_text)
        defaults = self._get_defaults()
        try:
            if self._settings.use_legal_slm:
                meta = self._extract_with_slm(text)
            else:
                meta = self._extract_with_openai(text)
            for key in defaults:
                item[key] = meta.get(key, defaults[key])
        except Exception:
            logger.warning("Metadata enrichment failed, using defaults.", exc_info=True)
            for key, val in defaults.items():
                item[key] = val
        return item
 
    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _prepare_enrichment_text(full_text: str, max_chars: int = 12000) -> str:
        full_text = full_text.strip()
        if len(full_text) <= max_chars:
            return full_text
        head = full_text[:4000]
        middle_start = max(0, len(full_text) // 2 - 2000)
        middle = full_text[middle_start:middle_start + 4000]
        tail = full_text[-4000:]
        return f"{head}\n\n[...]\n\n{middle}\n\n[...]\n\n{tail}"
 
    def _extract_with_slm(self, text: str) -> dict:
        prompt = self._get_prompt().replace("{text}", text)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                stop_strings=["###", "### Example"],
            )
        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return json.loads(generated.strip())
 
    def _extract_with_openai(self, text: str) -> dict:
        client = OpenAI(api_key=self._settings.openai_api_key)
        resp = client.chat.completions.create(
            model=self._settings.openai_llm_model_enrichment,
            messages=[
                {"role": "system", "content": "You are a legal metadata tagger. Output JSON only."},
                {"role": "user", "content": self._get_prompt().replace("{text}", text)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)

# ---------------------------------------------------------------------------
# Pipeline 4 – Upsert into Qdrant
# ---------------------------------------------------------------------------
class QdrantPipeline:
    """Embed chunks and upsert them into the regulatory_updates Qdrant collection.
 
    Chunks are processed in batches of _QDRANT_BATCH_SIZE so that a 200-chunk
    EDPB guideline PDF results in ~7 batched embedding calls rather than 200
    sequential single-item calls.
    """

    def __init__(self) -> None:
        self._client = None
        self._settings = None
        self._collection: str = ""
        # Accumulation buffer: chunks are buffered here until a full batch is
        # ready or the spider closes.
        self._buffer: list[ChunkedRegulationItem] = []
 
    def open_spider(self, spider: Any) -> None:
        self._settings = get_settings()
        self._client = client_from_settings(self._settings)
        self._collection = self._settings.qdrant_regulatory_collection
 
    def process_item(self, item: Any, spider: Any) -> Any:
        """Accept a single ChunkedRegulationItem or a list of them."""
        incoming: list[ChunkedRegulationItem] = []
 
        if isinstance(item, list):
            incoming = [c for c in item if isinstance(c, ChunkedRegulationItem)]
        elif isinstance(item, ChunkedRegulationItem):
            incoming = [item]
 
        if not incoming:
            return item
 
        self._buffer.extend(incoming)
 
        # Flush complete batches immediately; keep remainder in buffer.
        while len(self._buffer) >= _QDRANT_BATCH_SIZE:
            batch = self._buffer[:_QDRANT_BATCH_SIZE]
            self._buffer = self._buffer[_QDRANT_BATCH_SIZE:]
            self._upsert_batch(batch)
 
        return item
 
    def close_spider(self, spider: Any) -> None:
        """Flush any remaining buffered chunks when the spider finishes."""
        if self._buffer:
            self._upsert_batch(self._buffer)
            self._buffer = []
 
    def _upsert_batch(self, chunks: list[ChunkedRegulationItem]) -> None:
        embed_inputs = [
            f"[{c.get('compliance_domain', '')}] "
            f"[{', '.join(c.get('topic_tags', []))}] "
            f"{c['text']}"
            for c in chunks
        ]
 
        named_vectors_list = compute_vectors(
            embed_inputs,
            self._settings,
            dense_name=self._settings.qdrant_regulatory_dense_name,
            sparse_name=self._settings.qdrant_sparse_name,
        )
 
        points = [
            self._build_point(chunk, named_vectors)
            for chunk, named_vectors in zip(chunks, named_vectors_list)
        ]
 
        self._client.upsert(collection_name=self._collection, points=points)
        logger.info("Upserted batch of %d regulatory chunks.", len(points))
 
    @staticmethod
    def _build_point(
        chunk: ChunkedRegulationItem, named_vectors: Any
    ) -> PointStruct:
        point_id = int(
            hashlib.sha256(chunk["chunk_id"].encode("utf-8")).hexdigest()[:32], 16
        ) % (2**63)
 
        return PointStruct(
            id=point_id,
            vector=named_vectors,
            payload={
                "text": chunk["text"],
                "header_path": chunk["header_path"],
                "source_url": chunk["source_url"],
                "jurisdiction": chunk["jurisdiction"],
                "effective_date": chunk["effective_date"],
                "is_processed": False,
                "document_id": chunk["document_id"],
                "document_hash": chunk["document_hash"],
                "chunk_id": chunk["chunk_id"],
                "prev_chunk_id": chunk["prev_chunk_id"],
                "next_chunk_id": chunk["next_chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "chunk_count": chunk["chunk_count"],
                "token_count": chunk["token_count"],
                "topic_tags": chunk.get("topic_tags", []),
                "compliance_domain": chunk.get("compliance_domain", ""),
                "applies_to_departments": chunk.get("applies_to_departments", []),
                "obligation_type": chunk.get("obligation_type", ""),
            },
        )