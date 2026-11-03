"""Opt-in live ingestion of fictional evaluation documents.

Requires a configured PostgreSQL, Qdrant, and embedding provider. Run one case
at a time to avoid unrelated fictional policies changing retrieval outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_agent.config import get_settings
from legal_agent.db import ensure_collections
from legal_agent.db.client import client_from_settings
from legal_agent.db.manifest import DocumentManifest
from legal_agent.scraping.items import RegulatoryDocumentItem
from legal_agent.scraping.pipelines import (
    ChunkingPipeline,
    ContentValidationPipeline,
    QdrantPipeline,
)
from legal_agent.utils.identity import content_hash, revision_id
from legal_agent.utils.loader import _build_point
from legal_agent.utils.models import compute_vectors

FIXTURE = Path(__file__).with_name("synthetic_cases.json")


def load_cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def seed_case(case: dict) -> tuple[int, int]:
    settings = get_settings()
    manifest = DocumentManifest(settings.database_url)
    client = client_from_settings(settings)
    ensure_collections(client, settings)
    policy_count = 0
    if case["policy_text"]:
        policy_id = f"fictional-{case['id']}"
        policy_hash = content_hash(case["policy_text"])
        document_id = revision_id(f"policy::{policy_id}", case["policy_text"])
        chunk = {
            "text": case["policy_text"], "header_path": "Fictional policy",
            "chunk_id": f"{document_id}:0", "chunk_index": 0, "chunk_count": 1,
            "document_id": document_id, "policy_id": policy_id, "department": "Fictional",
            "compliance_domain": "Fictional privacy",
            "topic_tags": ["fictional", case["id"]],
        }
        vectors = compute_vectors(
            [chunk["text"]], settings,
            dense_name=settings.qdrant_policies_dense_name,
            sparse_name=settings.qdrant_sparse_name,
        )
        client.upsert(
            collection_name=settings.qdrant_policies_collection,
            points=[_build_point(chunk, vectors[0], policy_hash)],
            wait=True,
        )
        policy_count = 1

    # These are Scrapy's own validation, chunking, and Qdrant pipelines.
    item = RegulatoryDocumentItem(
        title="FICTIONAL regulatory notice",
        full_text=case["regulation_quote"],
        jurisdiction="Fictional",
        source_url=case["source"],
        source_id="fictional-evaluation",
        document_type="fictional_notice",
        is_pdf=False,
        raw_pdf_bytes=b"",
    )
    validation = ContentValidationPipeline()
    validation._settings = settings.model_copy(update={
        "content_validation_min_chars": 1, "content_validation_llm_enabled": False
    })
    validation._manifest = manifest
    try:
        item = validation.process_item(item, spider=None)
    except Exception as exc:
        # An unchanged current revision is an expected idempotent seed result.
        if "unchanged document" in str(exc):
            return (0, policy_count)
        raise
    chunker = ChunkingPipeline(
        tokenizer_model=settings.chunking_tokenizer_model,
        max_tokens=settings.chunk_max_tokens,
    )
    chunker.open_spider(None)
    chunks = chunker.process_item(item, spider=None)
    upsert = QdrantPipeline()
    upsert.open_spider(None)
    upsert.process_item(chunks, spider=None)
    upsert.close_spider(None)
    return (len(chunks), policy_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed one fictional evaluation case")
    parser.add_argument("--case", required=True, help="Case ID from synthetic_cases.json")
    args = parser.parse_args()
    cases = {case["id"]: case for case in load_cases()}
    if args.case == "multiple-obligations":
        first = cases["multiple-obligations-a"]
        second = cases["multiple-obligations-b"]
        combined = {
            **first,
            "id": "multiple-obligations",
            "regulation_quote": first["regulation_quote"] + "\n\n" + second["regulation_quote"],
        }
        regulation_chunks, policy_chunks = seed_case(combined)
        print(f"Seeded multiple-obligations: {regulation_chunks} regulation, {policy_chunks} policy chunks")
        return
    if args.case not in cases:
        parser.error(f"unknown case {args.case!r}; choose from {', '.join(cases)}")
    regulation_chunks, policy_chunks = seed_case(cases[args.case])
    print(f"Seeded {args.case}: {regulation_chunks} regulation, {policy_chunks} policy chunks")


if __name__ == "__main__":
    main()
