"""Scrapy pipelines: PDF conversion, hierarchical chunking, Qdrant upsert."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from openai import OpenAI
from qdrant_client.models import PointStruct
from scrapy.exceptions import DropItem
from transformers import AutoTokenizer

from legal_agent.config import get_settings
from legal_agent.db.client import client_from_settings
from legal_agent.db.manifest import DocumentManifest
from legal_agent.scraping.content_validation import (
    classify_borderline,
    quarantine_record,
    validate_deterministic,
)
from legal_agent.scraping.items import ChunkedRegulationItem, RegulatoryDocumentItem
from legal_agent.utils.identity import (
    canonicalize_url,
    content_hash,
    revision_id,
    source_key,
)
from legal_agent.utils.identity import point_id as qdrant_point_id
from legal_agent.utils.models import compute_vectors
from legal_agent.workflow.llm_provider import local_chat_template_kwargs
from legal_agent.workflow.prompts import _ENRICHMENT_FEW_SHOT, _POLICY_ENRICHMENT_FEW_SHOT

logger = logging.getLogger(__name__)

_QDRANT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Pipeline 1 – Convert PDF to Markdown via Docling
# ---------------------------------------------------------------------------
class DoclingPdfPipeline:
    """Convert PDF items to Markdown using Docling; pass through non-PDF items."""

    def __init__(self) -> None:
        self._converter: DocumentConverter | None = None

    def process_item(self, item: RegulatoryDocumentItem, spider: Any) -> RegulatoryDocumentItem:
        if not item.get("is_pdf") or not item.get("raw_pdf_bytes"):
            return item

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(item["raw_pdf_bytes"])
                tmp_path = Path(tmp.name)

            if self._converter is None:
                self._converter = DocumentConverter()
            result = self._converter.convert(str(tmp_path))
            item["full_text"] = result.document.export_to_markdown()
            item["raw_pdf_bytes"] = b""
        except Exception:
            logger.warning(
                "Docling PDF conversion failed for %s", item["source_url"], exc_info=True
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        return item


# ---------------------------------------------------------------------------
# Pipeline 2 – Content validation
# ---------------------------------------------------------------------------
class ContentValidationPipeline:
    """Reject obvious non-documents before enrichment, chunking, or embedding."""

    def __init__(self) -> None:
        self._settings = None
        self._manifest = None

    def open_spider(self, spider: Any) -> None:
        self._settings = get_settings()
        self._manifest = DocumentManifest(self._settings.manifest_file)

    def process_item(self, item: RegulatoryDocumentItem, spider: Any) -> RegulatoryDocumentItem:
        if not isinstance(item, RegulatoryDocumentItem):
            return item

        settings = self._settings or get_settings()
        result = validate_deterministic(
            title=item.get("title", ""),
            text=item.get("full_text", ""),
            document_type=item.get("document_type", ""),
            min_chars=getattr(settings, "content_validation_min_chars", 500),
        )
        if result.borderline and getattr(settings, "content_validation_llm_enabled", True):
            try:
                result = classify_borderline(
                    title=item.get("title", ""),
                    text=item.get("full_text", ""),
                    settings=settings,
                )
            except Exception as exc:
                logger.warning("Borderline content classifier failed.", exc_info=True)
                result = type(result)(False, reason=f"classifier failed: {exc}")

        if not result.accepted:
            self._quarantine(item, result, settings)
            raise DropItem(f"Content validation rejected {item.get('source_url', '')}: {result.reason}")

        normalized_text = re.sub(r"\s+", " ", item.get("full_text", "")).strip()
        canonical_url = canonicalize_url(item.get("source_url", ""))
        item_source_id = item.get("source_id", "generic")
        item_source_key = item.get("source_key") or source_key(item_source_id, canonical_url)
        item_revision_id = revision_id(item_source_key, normalized_text)
        manifest = self._manifest or DocumentManifest(settings.manifest_file)
        manifest_result = manifest.prepare(
            source_id=item_source_id,
            source_key=item_source_key,
            source_url=canonical_url,
            document_hash=content_hash(normalized_text),
            revision_id=item_revision_id,
            title=item.get("title", ""),
        )
        if manifest_result["action"] == "unchanged":
            raise DropItem(f"Content validation skipped unchanged document: {canonical_url}")

        item["source_url"] = canonical_url
        item["source_key"] = item_source_key
        item["revision_id"] = item_revision_id
        item["document_id"] = item_revision_id
        item["supersedes_revision_id"] = manifest_result.get("supersedes_revision_id", "")
        item["is_current"] = True
        item["validation_status"] = "accepted"
        item["validation_reason"] = result.reason
        return item

    @staticmethod
    def _quarantine(
        item: RegulatoryDocumentItem,
        result,
        settings: Any,
    ) -> None:
        path = Path(
            getattr(
                settings,
                "quarantine_file",
                "data/quarantine/rejected_documents.jsonl",
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(quarantine_record(dict(item), result)) + "\n")


# ---------------------------------------------------------------------------
# Pipeline 3 – Hierarchical chunking
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
        document_hash = content_hash(normalized_text)
        source_url = canonicalize_url(item["source_url"])
        item_source_id = item.get("source_id", "generic")
        item_source_key = item.get("source_key") or source_key(item_source_id, source_url)
        document_id = revision_id(item_source_key, normalized_text)

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
                    source_url=source_url,
                    jurisdiction=item["jurisdiction"],
                    effective_date=item.get("effective_date", ""),
                    is_processed=False,
                    workflow_status="pending",
                    document_id=document_id,
                    revision_id=document_id,
                    document_hash=document_hash,
                    is_current=True,
                    supersedes_revision_id=item.get("supersedes_revision_id", ""),
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
                    source_id=item_source_id,
                    source_key=item_source_key,
                    document_type=item.get("document_type", ""),
                    published_at=item.get("published_at", ""),
                    legal_identifier=item.get("legal_identifier", ""),
                    language=item.get("language", ""),
                    discovery_url=item.get("discovery_url", ""),
                    mime_type=item.get("mime_type", ""),
                    retrieved_at=item.get("retrieved_at", ""),
                    etag=item.get("etag", ""),
                    last_modified=item.get("last_modified", ""),
                    validation_status=item.get("validation_status", ""),
                    validation_reason=item.get("validation_reason", ""),
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
            window = token_ids[start : start + self.max_tokens]
            if not window:
                continue
            chunks.append(self.tokenizer.decode(window, skip_special_tokens=True).strip())

        return [chunk for chunk in chunks if chunk]


# ---------------------------------------------------------------------------
# Pipeline 3 – Metadata enrichment
# ---------------------------------------------------------------------------
_COLLECTION_REGULATION = "regulation"
_COLLECTION_POLICY = "policy"


def _parse_json_object(content: str | None) -> dict:
    """Parse a JSON object, tolerating a fenced response from local models."""
    if not content:
        raise ValueError("Metadata model returned no content.")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Metadata model response must be a JSON object.")
    return parsed


class MetadataEnrichmentPipeline:
    """Use local llama.cpp, Gemini, or OpenAI to extract normalized metadata.

    The ``collection`` parameter selects which prompt and output schema to use:

        collection="regulation"  (default — used by the Scrapy pipeline)
            Prompt  : _ENRICHMENT_FEW_SHOT
            Outputs : topic_tags, compliance_domain,
                      applies_to_departments (list), obligation_type

        collection="policy"  (used by loader.py for internal policy PDFs)
            Prompt  : _POLICY_ENRICHMENT_FEW_SHOT
            Outputs : topic_tags, compliance_domain,
                      department (scalar string), obligation_type

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
        self._settings = None

    @classmethod
    def from_crawler(cls, crawler):
        # Scrapy calls this instead of __init__ when ITEM_PIPELINES is used.
        # Always "regulation" so the spider pipeline is unchanged.
        return cls(collection=_COLLECTION_REGULATION)

    def open_spider(self, spider):
        self._settings = get_settings()

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
                "department": "General",  # scalar – matches _POLICY_ENRICHMENT_FEW_SHOT
                "obligation_type": "",
            }
        return {
            "topic_tags": [],
            "compliance_domain": "",
            "applies_to_departments": [],  # list – matches _ENRICHMENT_FEW_SHOT
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
            meta = self.extract_metadata(text)
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
        middle = full_text[middle_start : middle_start + 4000]
        tail = full_text[-4000:]
        return f"{head}\n\n[...]\n\n{middle}\n\n[...]\n\n{tail}"

    def extract_metadata(self, text: str) -> dict:
        """Extract metadata using the configured enrichment provider."""
        if self._settings is None:
            self._settings = get_settings()
        if self._settings.use_local_llm:
            return self._extract_with_local_llm(text)
        if self._settings.use_gemini_enrichment:
            return self._extract_with_gemini(text)
        return self._extract_with_openai(text)

    def _extract_with_local_llm(self, text: str) -> dict:
        client = OpenAI(
            base_url=self._settings.local_llm_base_url,
            api_key=self._settings.local_llm_api_key,
            timeout=self._settings.local_llm_timeout_seconds,
        )
        resp = client.chat.completions.create(
            model=self._settings.local_llm_model,
            messages=[
                {"role": "system", "content": "You are a legal metadata tagger. Output JSON only."},
                {"role": "user", "content": self._get_prompt().replace("{text}", text)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            extra_body={
                "chat_template_kwargs": local_chat_template_kwargs("enrichment"),
            },
        )
        return _parse_json_object(resp.choices[0].message.content)

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
        return _parse_json_object(resp.choices[0].message.content)

    def _extract_with_gemini(self, text: str) -> dict:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._settings.gemini_api_key)
        thinking_level = self._settings.enrichment_thinking
        resp = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=self._get_prompt().replace("{text}", text),
            config=types.GenerateContentConfig(
                system_instruction="You are a legal metadata tagger. Output JSON only.",
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return _parse_json_object(resp.text)


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
        self._manifest = None
        # Accumulation buffer: chunks are buffered here until a full batch is
        # ready or the spider closes.
        self._buffer: list[ChunkedRegulationItem] = []

    def open_spider(self, spider: Any) -> None:
        self._settings = get_settings()
        self._client = client_from_settings(self._settings)
        self._collection = self._settings.qdrant_regulatory_collection
        self._manifest = DocumentManifest(self._settings.manifest_file)

    def process_item(self, item: Any, spider: Any) -> Any:
        """Accept a single ChunkedRegulationItem or a list of them."""
        incoming: list[ChunkedRegulationItem] = []

        if isinstance(item, list):
            incoming = [c for c in item if isinstance(c, ChunkedRegulationItem)]
        elif isinstance(item, ChunkedRegulationItem):
            incoming = [item]

        if not incoming:
            return item

        manifest = self._manifest or DocumentManifest(get_settings().manifest_file)
        for chunk in incoming:
            manifest.set_expected_chunks(
                chunk.get("revision_id", chunk["document_id"]),
                chunk.get("chunk_count", 0),
            )
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
            f"[{c.get('compliance_domain', '')}] [{', '.join(c.get('topic_tags', []))}] {c['text']}"
            for c in chunks
        ]

        revisions = {
            chunk.get("revision_id", chunk["document_id"])
            for chunk in chunks
        }
        try:
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
        except Exception as exc:
            manifest = self._manifest or DocumentManifest(self._settings.manifest_file)
            for revision in revisions:
                manifest.mark_failed(revision, str(exc))
            raise

        logger.info("Upserted batch of %d regulatory chunks.", len(points))

        manifest = self._manifest or DocumentManifest(self._settings.manifest_file)
        for revision in revisions:
            revision_chunks = [
                chunk for chunk in chunks if chunk.get("revision_id", chunk["document_id"]) == revision
            ]
            if not manifest.record_chunks(revision, len(revision_chunks)):
                continue
            superseded = manifest.mark_stored(revision)
            if superseded:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                self._client.set_payload(
                    collection_name=self._collection,
                    payload={
                        "is_current": False,
                        "is_processed": True,
                        "workflow_status": "superseded",
                    },
                    points=Filter(
                        must=[
                            FieldCondition(
                                key="revision_id",
                                match=MatchValue(value=superseded),
                            )
                        ]
                    ),
                )

    @staticmethod
    def _build_point(chunk: ChunkedRegulationItem, named_vectors: Any) -> PointStruct:
        point_id = qdrant_point_id(
            chunk.get("revision_id", chunk["document_id"]),
            chunk["chunk_index"],
        )

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
                "workflow_status": chunk.get("workflow_status", "pending"),
                "document_id": chunk["document_id"],
                "revision_id": chunk.get("revision_id", chunk["document_id"]),
                "document_hash": chunk["document_hash"],
                "is_current": chunk.get("is_current", True),
                "supersedes_revision_id": chunk.get("supersedes_revision_id", ""),
                "chunk_id": chunk["chunk_id"],
                "prev_chunk_id": chunk["prev_chunk_id"],
                "next_chunk_id": chunk["next_chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "chunk_count": chunk["chunk_count"],
                "token_count": chunk["token_count"],
                "source_id": chunk.get("source_id", ""),
                "source_key": chunk.get("source_key", ""),
                "document_type": chunk.get("document_type", ""),
                "published_at": chunk.get("published_at", ""),
                "legal_identifier": chunk.get("legal_identifier", ""),
                "language": chunk.get("language", ""),
                "discovery_url": chunk.get("discovery_url", ""),
                "mime_type": chunk.get("mime_type", ""),
                "retrieved_at": chunk.get("retrieved_at", ""),
                "etag": chunk.get("etag", ""),
                "last_modified": chunk.get("last_modified", ""),
                "validation_status": chunk.get("validation_status", ""),
                "validation_reason": chunk.get("validation_reason", ""),
                "topic_tags": chunk.get("topic_tags", []),
                "compliance_domain": chunk.get("compliance_domain", ""),
                "applies_to_departments": chunk.get("applies_to_departments", []),
                "obligation_type": chunk.get("obligation_type", ""),
            },
        )
