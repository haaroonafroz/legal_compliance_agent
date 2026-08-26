"""Source-specific discovery and document acquisition adapters."""

from legal_agent.scraping.adapters.base import (
    DiscoveredDocument,
    SourceAdapter,
    SourceTarget,
)
from legal_agent.scraping.adapters.registry import get_source_adapter

__all__ = [
    "DiscoveredDocument",
    "SourceAdapter",
    "SourceTarget",
    "get_source_adapter",
]
