"""Seed the opt-in GDPR Article 33 evaluation pair into the *_eval collections.

The external document is a test paraphrase of the actual GDPR; the internal
policy belongs to a fictional company. No collections are cleared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg
from scrapy.exceptions import DropItem

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
from legal_agent.utils.identity import content_hash, revision_id, source_key
from legal_agent.utils.loader import _build_point
from legal_agent.utils.models import compute_vectors

FIXTURE = Path(__file__).with_name("realistic_breach_case.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed or requeue the Article 33 evaluation")
    parser.add_argument(
        "--requeue",
        action="store_true",
        help="Reset only this unreviewed test document so run-workflow can process it again",
    )
    args = parser.parse_args()
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))
    settings = get_settings()
    if not (
        settings.qdrant_regulatory_collection.endswith("_eval")
        and settings.qdrant_policies_collection.endswith("_eval")
    ):
        raise SystemExit("This seed is restricted to the *_eval Qdrant collections.")
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is required for the regulatory document manifest.")
    if not case.get("test_only") or len(case["regulation_text"]) < 500:
        raise SystemExit("The evaluation fixture is missing its test label or full source text.")

    if args.requeue:
        document_id = revision_id(
            source_key(case["source_id"], case["official_source_url"]),
            case["regulation_text"],
        )
        with psycopg.connect(settings.database_url) as connection:
            row = connection.execute(
                """UPDATE document_manifest
                   SET workflow_status = 'pending', report = NULL,
                       claim_token = NULL, claim_until = NULL, updated_at = now()
                   WHERE revision_id = %s AND source_id = %s
                     AND status = 'stored' AND is_current AND review IS NULL
                     AND claim_token IS NULL
                     AND workflow_status IN ('review_required', 'pending_human_review')
                   RETURNING revision_id""",
                (document_id, case["source_id"]),
            ).fetchone()
        if row is None:
            raise SystemExit("The current unreviewed test document is not ready to requeue.")
        print(f"Requeued evaluation revision {document_id}. Its prior test report was cleared.")
        return

    client = client_from_settings(settings)
    ensure_collections(client, settings)

    # Upsert the stable fictional policy point; reruns replace the same point.
    policy_text = case["policy_text"]
    policy_id = case["policy_id"]
    policy_document_id = revision_id(f"policy::{policy_id}", policy_text)
    chunk = {
        "text": policy_text,
        "header_path": case["policy_title"],
        "chunk_id": f"{policy_document_id}:0",
        "chunk_index": 0,
        "chunk_count": 1,
        "document_id": policy_document_id,
        "policy_id": policy_id,
        "department": case["policy_department"],
        "compliance_domain": "Data protection",
        "topic_tags": ["GDPR", "personal data breach", "supervisory authority notification"],
        "obligation_type": "incident response",
    }
    vectors = compute_vectors(
        [policy_text],
        settings,
        dense_name=settings.qdrant_policies_dense_name,
        sparse_name=settings.qdrant_sparse_name,
    )
    client.upsert(
        collection_name=settings.qdrant_policies_collection,
        points=[_build_point(chunk, vectors[0], content_hash(policy_text))],
        wait=True,
    )

    item = RegulatoryDocumentItem(
        title=case["source_title"],
        full_text=case["regulation_text"],
        jurisdiction=case["jurisdiction"],
        source_url=case["official_source_url"],
        source_id=case["source_id"],
        document_type="regulation",
        legal_identifier=case["legal_identifier"],
        language="en",
        is_pdf=False,
        raw_pdf_bytes=b"",
    )
    validation = ContentValidationPipeline()
    validation._settings = settings
    validation._manifest = DocumentManifest(settings.database_url)
    try:
        item = validation.process_item(item, spider=None)
    except DropItem as exc:
        if "unchanged document" not in str(exc):
            raise
        print("Regulatory revision already present; policy point refreshed.")
        return

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
    print(
        f"Seeded {len(chunks)} regulatory chunk(s) and 1 policy point into "
        f"{settings.qdrant_regulatory_collection} and {settings.qdrant_policies_collection}."
    )
    print(f"Regulatory revision: {item['revision_id']}")


if __name__ == "__main__":
    main()
