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
from legal_agent.config import get_settings
from legal_agent.workflow.prompts import _ENRICHMENT_FEW_SHOT
from legal_agent.scraping.items import ChunkedRegulationItem, RegulatoryDocumentItem
from qdrant_client.models import PointStruct
from legal_agent.utils.models import embed_texts, compute_vectors
from docling.document_converter import DocumentConverter
from legal_agent.config import get_settings
from legal_agent.db.client import client_from_settings

logger = logging.getLogger(__name__)


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


class ChunkingPipeline:
    """Split full_text into hierarchical chunks with header_path metadata."""

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    def process_item(
        self, item: RegulatoryDocumentItem, spider: Any
    ) -> list[ChunkedRegulationItem]:
        text: str = item.get("full_text", "")
        if not text:
            return []

        sections = self._split_by_headings(text)
        chunks: list[ChunkedRegulationItem] = []

        for idx, (header_path, section_text) in enumerate(sections):
            for sub_idx, chunk_text in enumerate(self._split_large(section_text)):
                chunks.append(
                    ChunkedRegulationItem(
                        text=chunk_text,
                        header_path=header_path,
                        source_url=item["source_url"],
                        jurisdiction=item["jurisdiction"],
                        effective_date=item.get("effective_date", ""),
                        is_processed=False,
                        chunk_index=len(chunks),
                    )
                )

        logger.info("Chunked '%s' into %d pieces.", item.get("title", ""), len(chunks))
        return chunks  # type: ignore[return-value]

    @staticmethod
    def _split_by_headings(text: str) -> list[tuple[str, str]]:
        """Return (header_path, body) tuples following heading hierarchy."""
        parts: list[tuple[str, str]] = []
        heading_stack: list[str] = []
        current_level = 0
        last_pos = 0

        for m in _HEADING_RE.finditer(text):
            level = len(m.group(1))
            title = m.group(2).strip()

            if last_pos < m.start():
                body = text[last_pos : m.start()].strip()
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

    def _split_large(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.max_tokens:
            return [text]
        chunks = []
        for i in range(0, len(words), self.max_tokens):
            chunks.append(" ".join(words[i : i + self.max_tokens]))
        return chunks


# ---------------------------------------------------------------------------
# Pipeline 3 – Metadata enrichment
# ---------------------------------------------------------------------------
class MetadataEnrichmentPipeline:
    """Use a legal SLM or OpenAI to extract normalized metadata from each chunk."""
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._settings = None
    def open_spider(self, spider):
        self._settings = get_settings()
        if self._settings.use_legal_slm:
            self._tokenizer, self._model = get_legal_slm(
                self._settings.legal_slm_model,
                self._settings.legal_slm_device,
                self._settings.legal_slm_load_in_4bit,
            )
    def process_item(self, item, spider):
        if isinstance(item, list):
            return [self._enrich(chunk) for chunk in item]
        if isinstance(item, ChunkedRegulationItem):
            return self._enrich(item)
        return item
    def _enrich(self, chunk):
        text = chunk["text"][:2000]
        defaults = {
            "topic_tags": [],
            "compliance_domain": "",
            "applies_to_departments": [],
            "obligation_type": "",
        }
        try:
            if self._settings.use_legal_slm:
                meta = self._extract_with_slm(text)
            else:
                meta = self._extract_with_openai(text)
            for key in defaults:
                chunk[key] = meta.get(key, defaults[key])
        except Exception:
            logger.warning("Metadata enrichment failed, using defaults.", exc_info=True)
            for key, val in defaults.items():
                chunk[key] = val
        return chunk
    def _extract_with_slm(self, text: str) -> dict:
        prompt = _ENRICHMENT_FEW_SHOT.format(text=text)
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
            model=self._settings.openai_llm_model,
            messages=[
                {"role": "system", "content": "You are a legal metadata tagger. Output JSON only."},
                {"role": "user", "content": _ENRICHMENT_FEW_SHOT.format(text=text)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)
# ---------------------------------------------------------------------------
# Pipeline 4 – Upsert into Qdrant
# ---------------------------------------------------------------------------
class QdrantPipeline:
    """Embed chunks and upsert them into the regulatory_updates Qdrant collection."""

    def __init__(self) -> None:
        self._client = None
        self._embed_model = None
        self._collection: str = ""

    def open_spider(self, spider: Any) -> None:
        self._settings = get_settings()
        self._client = client_from_settings(self._settings)
        self._collection = self._settings.qdrant_regulatory_collection

    def process_item(self, item: Any, spider: Any) -> Any:
        if isinstance(item, list):
            for chunk in item:
                self._upsert_chunk(chunk)
        elif isinstance(item, ChunkedRegulationItem):
            self._upsert_chunk(item)
        return item

    def _upsert_chunk(self, chunk: ChunkedRegulationItem) -> None:
        text = chunk["text"]
        domain = chunk.get("compliance_domain", "")
        tags = chunk.get("topic_tags", [])
        embed_input = f"[{domain}] [{', '.join(tags)}] {text}"
        
        named_vectors = compute_vectors(
            [embed_input],
            self._settings,
            dense_name=self._settings.qdrant_regulatory_dense_name,  # "compliance"
            sparse_name=self._settings.qdrant_sparse_name,            # "legal_clause"
        )[0]
        point_id = hashlib.sha256(
            f"{chunk['source_url']}:{chunk['chunk_index']}".encode()
        ).hexdigest()[:32]
        point_id_int = int(point_id, 16) % (2**63)

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id_int,
                    vector=named_vectors,
                    payload={
                        "text": text,
                        "header_path": chunk["header_path"],
                        "source_url": chunk["source_url"],
                        "jurisdiction": chunk["jurisdiction"],
                        "effective_date": chunk["effective_date"],
                        "is_processed": False,
                        "chunk_index": chunk["chunk_index"],
                        "topic_tags": chunk.get("topic_tags", []),
                        "compliance_domain": chunk.get("compliance_domain", ""),
                        "applies_to_departments": chunk.get("applies_to_departments", []),
                        "obligation_type": chunk.get("obligation_type", ""),
                    },
                )
            ],
        )
