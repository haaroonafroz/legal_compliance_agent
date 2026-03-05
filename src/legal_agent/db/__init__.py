"""Qdrant database layer – collection schemas and client helpers."""

from legal_agent.db.client import get_qdrant_client
from legal_agent.db.schemas import ensure_collections

__all__ = ["get_qdrant_client", "ensure_collections"]
