"""Qdrant collection definitions for regulatory_updates and internal_policies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from legal_agent.config import Settings

logger = logging.getLogger(__name__)

REGULATORY_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "is_processed": PayloadSchemaType.BOOL,
    "jurisdiction": PayloadSchemaType.KEYWORD,
    "source_url": PayloadSchemaType.KEYWORD,
}

POLICIES_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "department": PayloadSchemaType.KEYWORD,
    "policy_id": PayloadSchemaType.KEYWORD,
}


def ensure_collections(client: "QdrantClient", settings: "Settings") -> None:
    """Create Qdrant collections if they don't already exist, then add payload indexes."""
    vector_params = VectorParams(
        size=settings.openai_embedding_dim,
        distance=Distance.COSINE,
    )

    _ensure_one(
        client,
        name=settings.qdrant_regulatory_collection,
        vector_params=vector_params,
        indexes=REGULATORY_PAYLOAD_INDEXES,
    )

    _ensure_one(
        client,
        name=settings.qdrant_policies_collection,
        vector_params=vector_params,
        indexes=POLICIES_PAYLOAD_INDEXES,
    )


def _ensure_one(
    client: "QdrantClient",
    *,
    name: str,
    vector_params: VectorParams,
    indexes: dict[str, PayloadSchemaType],
) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(collection_name=name, vectors_config=vector_params)
        logger.info("Created Qdrant collection '%s'.", name)
    else:
        logger.info("Qdrant collection '%s' already exists.", name)

    for field, schema_type in indexes.items():
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema_type,
        )
    logger.info("Payload indexes ensured for '%s'.", name)
