"""Contracts and normalized records shared by source adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from scrapy import Request
from scrapy.http import Response


@dataclass(frozen=True)
class SourceTarget:
    """Validated configuration for one regulatory source."""

    source_id: str
    jurisdiction: str
    start_urls: tuple[str, ...] = ()
    feed_urls: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    allow_patterns: tuple[str, ...] = ()
    deny_patterns: tuple[str, ...] = ()
    document_path_patterns: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    language: str = "en"
    adapter: str = "generic"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_id: str) -> SourceTarget:
        """Build a target and fail early when its required fields are absent."""
        source_id = str(raw.get("source_id") or default_id).strip()
        jurisdiction = str(raw.get("jurisdiction", "")).strip()
        if not source_id:
            raise ValueError("source_id must not be empty")
        if not jurisdiction:
            raise ValueError(f"source {source_id!r} is missing jurisdiction")

        start_urls = tuple(str(url).strip() for url in raw.get("start_urls", []) if str(url).strip())
        feed_urls = tuple(str(url).strip() for url in raw.get("feed_urls", []) if str(url).strip())
        if not start_urls and not feed_urls:
            raise ValueError(f"source {source_id!r} must define start_urls or feed_urls")

        options = {
            "follow_links": raw.get("follow_links", False),
            "follow_pdf": raw.get("follow_pdf", False),
            "max_depth": raw.get("max_depth", 2),
            "max_pages": raw.get("max_pages", 0),
            "max_pdfs": raw.get("max_pdfs", 0),
            **dict(raw.get("options", {})),
        }
        return cls(
            source_id=source_id,
            jurisdiction=jurisdiction,
            start_urls=start_urls,
            feed_urls=feed_urls,
            allowed_domains=tuple(raw.get("allowed_domains", [])),
            allow_patterns=tuple(raw.get("allow_patterns", [])),
            deny_patterns=tuple(raw.get("deny_patterns", [])),
            document_path_patterns=tuple(raw.get("document_path_patterns", [])),
            document_types=tuple(raw.get("document_types", [])),
            language=str(raw.get("language", "en")),
            adapter=str(raw.get("adapter", "generic")),
            options=options,
        )


@dataclass(frozen=True)
class DiscoveredDocument:
    """A source item normalized before its content is fetched."""

    source_id: str
    url: str
    title: str = ""
    document_type: str = ""
    published_at: str = ""
    effective_date: str = ""
    legal_identifier: str = ""
    language: str = "en"
    discovery_url: str = ""
    external_id: str = ""
    mime_type: str = ""

    def as_meta(self) -> dict[str, str]:
        """Return values safe to place in Scrapy request metadata."""
        return {
            key: value
            for key, value in {
                "source_id": self.source_id,
                "title": self.title,
                "document_type": self.document_type,
                "published_at": self.published_at,
                "effective_date": self.effective_date,
                "legal_identifier": self.legal_identifier,
                "language": self.language,
                "discovery_url": self.discovery_url,
                "external_id": self.external_id,
                "mime_type": self.mime_type,
            }.items()
            if value
        }


class SourceAdapter(Protocol):
    """Protocol implemented by source-specific discovery adapters."""

    name: str

    def discovery_requests(self, target: SourceTarget) -> Iterable[Request]:
        """Return feed/API/listing requests used to discover documents."""

    def parse_discovery(
        self, response: Response, target: SourceTarget
    ) -> Iterable[DiscoveredDocument]:
        """Extract normalized document records from a discovery response."""
