"""EDPB RSS/HTML discovery adapter."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from urllib.parse import urlparse
from xml.etree import ElementTree

import scrapy
from scrapy.http import Response

from legal_agent.scraping.adapters.base import DiscoveredDocument, SourceTarget
from legal_agent.utils.identity import canonicalize_url

logger = logging.getLogger(__name__)

_LEGAL_TYPES = {
    "guideline": "guidelines",
    "guidelines": "guidelines",
    "recommendation": "recommendations",
    "recommendations": "recommendations",
    "opinion": "opinions",
    "opinions": "opinions",
    "decision": "decisions",
    "decisions": "decisions",
}
_LEGAL_LINK_RE = re.compile(
    r"\b(guidelines?|recommendations?|opinions?|decisions?)\b", re.IGNORECASE
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _first_text(element: ElementTree.Element, names: set[str]) -> str:
    for child in element.iter():
        if _local_name(child.tag) in names and child.text:
            return " ".join(child.text.split())
    return ""


class EDPBAdapter:
    """Discover EDPB publications from feeds and controlled listing pages."""

    name = "edpb"

    def discovery_requests(self, target: SourceTarget) -> Iterable[scrapy.Request]:
        for url in (*target.feed_urls, *target.start_urls):
            kind = "feed" if url in target.feed_urls else "html"
            yield scrapy.Request(
                url,
                dont_filter=True,
                meta={
                    "discovery": True,
                    "discovery_kind": kind,
                    "source_id": target.source_id,
                    "handle_httpstatus_list": [403, 404, 410, 429],
                },
            )

    def parse_discovery(
        self, response: Response, target: SourceTarget
    ) -> Iterable[DiscoveredDocument]:
        if response.status >= 400:
            logger.warning("EDPB discovery request failed with HTTP %d: %s", response.status, response.url)
            return
        if self._is_xml(response):
            yield from self._parse_feed(response, target)
        else:
            yield from self._parse_listing(response, target)

    def _parse_feed(
        self, response: Response, target: SourceTarget
    ) -> Iterable[DiscoveredDocument]:
        try:
            root = ElementTree.fromstring(response.body)
        except ElementTree.ParseError:
            return

        seen: set[str] = set()
        for entry in root.iter():
            if _local_name(entry.tag) not in {"item", "entry"}:
                continue

            title = _first_text(entry, {"title"})
            published_at = _first_text(entry, {"pubdate", "published", "updated", "date"})
            external_id = _first_text(entry, {"guid", "id"})
            links = self._entry_links(entry)
            if not links:
                continue

            for link in links:
                url = canonicalize_url(response.urljoin(link))
                if url in seen or not self._allowed_url(url, target):
                    continue
                seen.add(url)
                document_type = self._document_type(title, url)
                if target.document_types and document_type not in target.document_types:
                    continue
                yield DiscoveredDocument(
                    source_id=target.source_id,
                    url=url,
                    title=title,
                    document_type=document_type,
                    published_at=published_at,
                    language=target.language,
                    discovery_url=response.url,
                    external_id=external_id,
                    mime_type="application/pdf" if url.lower().endswith(".pdf") else "",
                )

    @staticmethod
    def _entry_links(entry: ElementTree.Element) -> list[str]:
        links: list[str] = []
        for child in entry:
            name = _local_name(child.tag)
            if name == "link":
                href = child.attrib.get("href") or (child.text or "")
                if href.strip():
                    links.append(href.strip())
            elif name == "enclosure":
                href = child.attrib.get("url") or ""
                if href.strip():
                    links.append(href.strip())
        return links

    def _parse_listing(
        self, response: Response, target: SourceTarget
    ) -> Iterable[DiscoveredDocument]:
        seen: set[str] = set()
        links = response.css("main a, article a, .view-content a, a")
        for link in links:
            href = link.attrib.get("href", "")
            if not href:
                continue
            url = canonicalize_url(response.urljoin(href))
            title = " ".join(link.css("::text").getall()).strip()
            if url in seen or not self._allowed_url(url, target):
                continue
            if not self._looks_like_document(url, title, target):
                continue
            seen.add(url)
            document_type = self._document_type(title, url)
            if target.document_types and document_type not in target.document_types:
                continue
            yield DiscoveredDocument(
                source_id=target.source_id,
                url=url,
                title=title,
                document_type=document_type,
                language=target.language,
                discovery_url=response.url,
                mime_type="application/pdf" if url.lower().endswith(".pdf") else "",
            )

    @staticmethod
    def _is_xml(response: Response) -> bool:
        content_type = response.headers.get("Content-Type", b"").decode(errors="ignore").lower()
        body = response.body.lstrip().lower()
        return "xml" in content_type or body.startswith((b"<?xml", b"<rss", b"<feed"))

    @staticmethod
    def _allowed_url(url: str, target: SourceTarget) -> bool:
        host = (urlparse(url).hostname or "").lower()
        domains = {domain.lstrip("*.").lower() for domain in target.allowed_domains}
        if domains and not any(host == domain or host.endswith(f".{domain}") for domain in domains):
            return False
        path = urlparse(url).path
        for pattern in target.deny_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return False
        if target.allow_patterns:
            return any(
                re.search(pattern, path, re.IGNORECASE) for pattern in target.allow_patterns
            )
        return True

    @classmethod
    def _looks_like_document(cls, url: str, title: str, target: SourceTarget) -> bool:
        path = urlparse(url).path.lower()
        if path.endswith(".pdf"):
            return True
        if target.document_path_patterns and not any(
            re.search(pattern, path, re.IGNORECASE) for pattern in target.document_path_patterns
        ):
            return False
        return bool(_LEGAL_LINK_RE.search(title) or _LEGAL_LINK_RE.search(path))

    @staticmethod
    def _document_type(title: str, url: str) -> str:
        match = _LEGAL_LINK_RE.search(f"{title} {url}")
        return _LEGAL_TYPES.get(match.group(1).lower(), "") if match else ""
