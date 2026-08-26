"""Registry for source-specific acquisition adapters."""

from __future__ import annotations

from legal_agent.scraping.adapters.base import SourceAdapter
from legal_agent.scraping.adapters.edpb import EDPBAdapter
from legal_agent.scraping.adapters.eurlex import EURLexAdapter

_ADAPTERS: dict[str, SourceAdapter] = {
    "edpb": EDPBAdapter(),
    "eurlex": EURLexAdapter(),
    "cellar": EURLexAdapter(),
}


def get_source_adapter(name: str) -> SourceAdapter | None:
    """Return a registered adapter, or ``None`` for the generic spider path."""
    return _ADAPTERS.get(name.strip().lower())
