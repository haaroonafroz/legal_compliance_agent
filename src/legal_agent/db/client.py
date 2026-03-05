"""Singleton Qdrant client."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

if TYPE_CHECKING:
    from legal_agent.config import Settings


@lru_cache(maxsize=1)
def get_qdrant_client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    """Return a cached Qdrant client instance."""
    return QdrantClient(url=url, api_key=api_key or None)


def client_from_settings(settings: "Settings") -> QdrantClient:
    return get_qdrant_client(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
