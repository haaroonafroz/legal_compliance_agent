"""PostgreSQL transitions and best-effort Qdrant search-payload synchronization."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from legal_agent.config import Settings
from legal_agent.db.client import client_from_settings
from legal_agent.db.manifest import DocumentManifest
from legal_agent.utils.report import save_report

logger = logging.getLogger(__name__)


def mirror_document_state(settings: Settings, document_id: str, status: str) -> None:
    """Update retrieval metadata after the authoritative DB transaction commits."""
    qdrant = client_from_settings(settings)
    qdrant.set_payload(
        collection_name=settings.qdrant_regulatory_collection,
        payload={"is_processed": True, "workflow_status": status},
        points=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        ]),
        wait=True,
    )


def persist_workflow_result(settings: Settings, result: dict[str, Any],
                            claim: dict[str, Any]) -> str:
    """Commit the report and transition together, then export a local copy."""
    manifest = DocumentManifest(settings.database_url)
    manifest.complete_workflow(
        claim["revision_id"], claim["document_hash"], claim["claim_token"], result
    )
    try:
        path = str(save_report(result))
    except Exception:
        logger.warning("Durable report committed; local export failed", exc_info=True)
        path = f"postgresql:document_manifest/{claim['revision_id']}"
    try:
        mirror_document_state(settings, claim["revision_id"], result["status"])
    except Exception:
        logger.warning("Qdrant status mirror failed; PostgreSQL state is intact", exc_info=True)
    return path
