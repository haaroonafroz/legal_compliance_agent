"""Qdrant collection definitions for regulatory_updates and internal_policies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseVectorParams,
    SparseIndexParams,
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
    "topic_tags": PayloadSchemaType.KEYWORD,
    "compliance_domain": PayloadSchemaType.KEYWORD,
    "applies_to_departments": PayloadSchemaType.KEYWORD,
    "obligation_type": PayloadSchemaType.KEYWORD,
}

POLICIES_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "department": PayloadSchemaType.KEYWORD,
    "policy_id": PayloadSchemaType.KEYWORD,
    "topic_tags": PayloadSchemaType.KEYWORD,
    "compliance_domain": PayloadSchemaType.KEYWORD,
    "obligation_type": PayloadSchemaType.KEYWORD,
}


def ensure_collections(client, settings):
    dim = settings.legal_embedding_dim if settings.use_legal_embeddings else settings.openai_embedding_dim
    # regulatory_updates: dense="compliance", sparse="legal_clause"
    _ensure_one(
        client,
        name=settings.qdrant_regulatory_collection,
        vectors_config={
            settings.qdrant_regulatory_dense_name: VectorParams(
                size=dim, distance=Distance.COSINE
            ),
        },
        sparse_vectors_config={
            settings.qdrant_sparse_name: SparseVectorParams(
                index=SparseIndexParams(on_disk=True),
                modifier="idf",
            ),
        },
        indexes=REGULATORY_PAYLOAD_INDEXES,
    )
    # internal_policies: dense="internal_policy", sparse="legal_clause"
    _ensure_one(
        client,
        name=settings.qdrant_policies_collection,
        vectors_config={
            settings.qdrant_policies_dense_name: VectorParams(
                size=dim, distance=Distance.COSINE
            ),
        },
        sparse_vectors_config={
            settings.qdrant_sparse_name: SparseVectorParams(
                index=SparseIndexParams(on_disk=True),
                modifier="idf",
            ),
        },
        indexes=POLICIES_PAYLOAD_INDEXES,
    )



def _ensure_one(client, *, name, vectors_config, sparse_vectors_config, indexes):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        logger.info("Created Qdrant collection '%s'.", name)
    else:
        logger.info("Qdrant collection '%s' already exists.", name)
    for field, schema_type in indexes.items():
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema_type,
        )