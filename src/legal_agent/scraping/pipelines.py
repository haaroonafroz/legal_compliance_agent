"""Scrapy pipelines: PDF conversion, hierarchical chunking, Qdrant upsert."""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from legal_agent.scraping.items import ChunkedRegulationItem, RegulatoryDocumentItem

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
            from docling.document_converter import DocumentConverter

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
# Pipeline 3 – Upsert into Qdrant
# ---------------------------------------------------------------------------
class QdrantPipeline:
    """Embed chunks and upsert them into the regulatory_updates Qdrant collection."""

    def __init__(self) -> None:
        self._client = None
        self._embed_model = None
        self._collection: str = ""

    def open_spider(self, spider: Any) -> None:
        from legal_agent.config import get_settings
        from legal_agent.db.client import client_from_settings

        settings = get_settings()
        self._client = client_from_settings(settings)
        self._collection = settings.qdrant_regulatory_collection

        from openai import OpenAI

        self._openai = OpenAI(api_key=settings.openai_api_key)
        self._embed_model = settings.openai_embedding_model

    def process_item(self, item: Any, spider: Any) -> Any:
        if isinstance(item, list):
            for chunk in item:
                self._upsert_chunk(chunk)
        elif isinstance(item, ChunkedRegulationItem):
            self._upsert_chunk(item)
        return item

    def _upsert_chunk(self, chunk: ChunkedRegulationItem) -> None:
        from qdrant_client.models import PointStruct

        text = chunk["text"]
        resp = self._openai.embeddings.create(input=[text], model=self._embed_model)
        vector = resp.data[0].embedding

        point_id = hashlib.sha256(
            f"{chunk['source_url']}:{chunk['chunk_index']}".encode()
        ).hexdigest()[:32]
        point_id_int = int(point_id, 16) % (2**63)

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id_int,
                    vector=vector,
                    payload={
                        "text": text,
                        "header_path": chunk["header_path"],
                        "source_url": chunk["source_url"],
                        "jurisdiction": chunk["jurisdiction"],
                        "effective_date": chunk["effective_date"],
                        "is_processed": False,
                        "chunk_index": chunk["chunk_index"],
                    },
                )
            ],
        )
