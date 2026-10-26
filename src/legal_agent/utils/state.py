"""Durable workflow state transitions for regulatory documents."""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchValue

from legal_agent.config import Settings
from legal_agent.db.client import client_from_settings
from legal_agent.utils.report import save_report


def mark_document_state(
    settings: Settings,
    document_id: str,
    status: str,
    *,
    client=None,
    expected_status: str | None = None,
    expected_hash: str | None = None,
) -> None:
    """Mark a document processed only after its report has been saved."""
    if not document_id or status not in {
        "irrelevant",
        "review_required",
        "pending_human_review",
        "approved",
        "rejected",
    }:
        raise ValueError("Invalid document ID or workflow status")
    qdrant = client or client_from_settings(settings)
    conditions = [FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    if expected_status is not None:
        points = []
        offset = None
        while True:
            page, offset = qdrant.scroll(
                collection_name=settings.qdrant_regulatory_collection,
                scroll_filter=Filter(must=conditions),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(page)
            if offset is None:
                break
        if not points:
            raise ValueError("Document is absent from Qdrant")
        expected_count = (points[0].payload or {}).get("chunk_count", len(points))
        if len(points) != expected_count:
            raise ValueError("Document is incomplete in Qdrant")
        for point in points:
            payload = point.payload or {}
            if payload.get("workflow_status") != expected_status or payload.get("is_current") is False:
                raise ValueError("Report no longer matches the current document state")
            if expected_hash is not None and payload.get("document_hash") != expected_hash:
                raise ValueError("Report document hash does not match Qdrant")
        conditions.append(
            FieldCondition(key="workflow_status", match=MatchValue(value=expected_status))
        )
    qdrant.set_payload(
        collection_name=settings.qdrant_regulatory_collection,
        payload={"is_processed": True, "workflow_status": status},
        points=Filter(must=conditions),
    )


def persist_workflow_result(settings: Settings, result: dict):
    """Save the reviewable report before removing a document from the pending queue."""
    path = save_report(result)
    mark_document_state(
        settings,
        result["document_id"],
        result["status"],
        expected_status="pending",
        expected_hash=result.get("document_hash"),
    )
    return path
