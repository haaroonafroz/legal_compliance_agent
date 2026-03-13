"""utils/loader.py – Ingest an internal policy PDF into the internal_policies Qdrant collection.

Pipeline order
--------------
    Docling → Chunking → per-chunk Enrichment → Embed → Upsert

Enrichment uses MetadataEnrichmentPipeline(collection="policy") which selects
_POLICY_ENRICHMENT_FEW_SHOT.  This prompt returns "department" as a scalar
string (not "applies_to_departments" as a list).  The LLM also returns
"policy_id" but that field is intentionally ignored — it is not stored in
the Qdrant payload.

Payload schema (internal_policies collection)
---------------------------------------------
    text, header_path                   core content
    chunk_id, prev_chunk_id,
    next_chunk_id, chunk_index,
    chunk_count, token_count            chunk linkage / sequence integrity
    document_hash                       document identity (content-based, no URL)
    department, topic_tags,
    compliance_domain, obligation_type  per-chunk enriched metadata
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from qdrant_client.models import PointStruct

from legal_agent.config import Settings, get_settings
from legal_agent.db.client import client_from_settings
from legal_agent.scraping.items import RegulatoryDocumentItem
from legal_agent.scraping.pipelines import (
    ChunkingPipeline,
    DoclingPdfPipeline,
    MetadataEnrichmentPipeline,
)
from legal_agent.utils.models import compute_vectors

logger = logging.getLogger(__name__)

_INTERNAL_SOURCE_PREFIX = "policy::"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_policy_pdf(
    pdf_path: Path,
    settings: Settings | None = None,
    *,
    department: str = "",
    compliance_domain: str = "",
    batch_size: int = 32,
) -> int:
    """Parse, chunk, enrich per-chunk, embed, and upsert a policy PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    settings:
        Application settings; resolved via get_settings() when omitted.
    department:
        Hard override for ``department`` on every chunk.
        When omitted the per-chunk enrichment model infers it.
    compliance_domain:
        Hard override for ``compliance_domain`` on every chunk.
        When omitted the per-chunk enrichment model infers it.
    batch_size:
        Chunks per embedding + Qdrant upsert call.

    Returns
    -------
    int
        Number of Qdrant points upserted.
    """
    if settings is None:
        settings = get_settings()

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"Policy PDF not found: {pdf_path}")

    logger.info("Starting policy ingestion: %s", pdf_path.name)

    # ------------------------------------------------------------------
    # Stage 1 – Docling: PDF bytes → Markdown full_text
    # ------------------------------------------------------------------
    doc_item = RegulatoryDocumentItem(
        title=pdf_path.stem,
        full_text="",
        jurisdiction="Internal",
        effective_date="",
        # Sentinel so ChunkingPipeline can build chunk_id / document_id.
        # Never stored in the final payload.
        source_url=f"{_INTERNAL_SOURCE_PREFIX}{pdf_path.name}",
        is_pdf=True,
        raw_pdf_bytes=pdf_path.read_bytes(),
    )

    docling_pipeline = DoclingPdfPipeline()
    doc_item = docling_pipeline.process_item(doc_item, spider=None)

    if not doc_item.get("full_text"):
        raise RuntimeError(
            f"Docling produced no text from '{pdf_path.name}'. "
            "Ensure the PDF is text-based (not a scanned image)."
        )
    logger.info("Docling: extracted %d characters.", len(doc_item["full_text"]))

    # ------------------------------------------------------------------
    # Stage 2 – Chunking: full_text → linked ChunkedPolicyItems
    # ------------------------------------------------------------------
    chunking_pipeline = ChunkingPipeline(
        tokenizer_model=settings.chunking_tokenizer_model,
        max_tokens=settings.chunk_max_tokens,
    )
    chunking_pipeline.open_spider(spider=None)

    chunks = chunking_pipeline.process_item(doc_item, spider=None)
    # Convert Scrapy Items → plain dicts so Stage 3 can freely write
    # per-chunk enrichment results onto them.
    chunks = [dict(c) for c in chunks]

    if not chunks:
        raise RuntimeError(
            f"ChunkingPipeline produced no chunks from '{pdf_path.name}'."
        )
    logger.info("Chunking: produced %d chunks.", len(chunks))

    document_hash: str = chunks[0]["document_hash"]
    
    # ------------------------------------------------------------------
    enrichment_pipeline = MetadataEnrichmentPipeline(collection="policy")
    enrichment_pipeline._settings = settings
    if settings.use_legal_slm:
        from legal_agent.scraping.pipelines import get_legal_slm
        enrichment_pipeline._tokenizer, enrichment_pipeline._model = get_legal_slm(
            settings.legal_slm_model,
            settings.legal_slm_device,
            settings.legal_slm_load_in_4bit,
        )

    # Fetch the policy-schema defaults (department scalar, no applies_to_departments).
    defaults = enrichment_pipeline._get_defaults()

    for i, chunk in enumerate(chunks):
        header = chunk.get("header_path", "")
        enrichment_input = (
            f"[Section: {header}]\n\n{chunk['text']}" if header else chunk["text"]
        )
        enrichment_input = MetadataEnrichmentPipeline._prepare_enrichment_text(
            enrichment_input
        )

        try:
            if settings.use_legal_slm:
                meta = enrichment_pipeline._extract_with_slm(enrichment_input)
            else:
                meta = enrichment_pipeline._extract_with_openai(enrichment_input)
        except Exception:
            logger.warning(
                "Per-chunk enrichment failed for chunk %d ('%s') – using defaults.",
                i, header, exc_info=True,
            )
            meta = dict(defaults)

        # Write only the four payload fields we keep.
        # policy_id is present in meta but deliberately not read here.
        chunk["topic_tags"] = meta.get("topic_tags", defaults["topic_tags"])
        chunk["compliance_domain"] = meta.get("compliance_domain", defaults["compliance_domain"])
        chunk["department"] = meta.get("department", defaults["department"])
        chunk["obligation_type"] = meta.get("obligation_type", defaults["obligation_type"])

        # Hard overrides always win over LLM inference.
        if compliance_domain:
            chunk["compliance_domain"] = compliance_domain
        if department:
            chunk["department"] = department

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            logger.info("Enriched %d / %d chunks.", i + 1, len(chunks))

    # ------------------------------------------------------------------
    # Stage 4 – Embed + upsert in batches
    # ------------------------------------------------------------------
    qdrant_client = client_from_settings(settings)
    collection = settings.qdrant_policies_collection
    total_upserted = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        embed_inputs = [
            f"[{c.get('compliance_domain', '')}] "
            f"[{', '.join(c.get('topic_tags', []))}] "
            f"{c['text'][:8000]}"
            for c in batch
        ]

        named_vectors_list = compute_vectors(
            embed_inputs,
            settings,
            dense_name=settings.qdrant_policies_dense_name,  # "internal_policy"
            sparse_name=settings.qdrant_sparse_name,          # "legal_clause"
        )

        points = [
            _build_point(chunk, named_vectors, document_hash)
            for chunk, named_vectors in zip(batch, named_vectors_list)
        ]

        qdrant_client.upsert(collection_name=collection, points=points)
        total_upserted += len(points)
        logger.info(
            "Upserted batch %d–%d  (%d / %d total)",
            batch_start, batch_start + len(batch) - 1,
            total_upserted, len(chunks),
        )

    logger.info("Ingestion complete: %d points → '%s'", total_upserted, collection)
    return total_upserted


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_point(
    chunk: dict,
    named_vectors: Any,
    document_hash: str,
) -> PointStruct:
    """Construct a PointStruct with the minimal internal_policies payload."""
    chunk_index: int = chunk["chunk_index"]
    point_id = int(
        hashlib.sha256(f"{document_hash}::{chunk_index}".encode()).hexdigest()[:15], 16
    )

    return PointStruct(
        id=point_id,
        vector=named_vectors,
        payload={
            # -- core content --
            "text": chunk["text"],
            "header_path": chunk.get("header_path", ""),
            # -- chunk linkage --
            "chunk_id": chunk["chunk_id"],
            "prev_chunk_id": chunk.get("prev_chunk_id", ""),
            "next_chunk_id": chunk.get("next_chunk_id", ""),
            "chunk_index": chunk_index,
            "chunk_count": chunk["chunk_count"],
            "token_count": chunk.get("token_count", 0),
            # -- document identity --
            "document_hash": document_hash,
            # -- per-chunk enriched metadata --
            "department": chunk.get("department", "General"),
            "topic_tags": chunk.get("topic_tags", []),
            "compliance_domain": chunk.get("compliance_domain", ""),
            "obligation_type": chunk.get("obligation_type", ""),
        },
    )