"""Bounded EUR-Lex/CELLAR acquisition adapter."""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlparse
from xml.etree import ElementTree

import scrapy
from scrapy.http import Response

from legal_agent.scraping.adapters.base import DiscoveredDocument, SourceTarget
from legal_agent.utils.identity import canonicalize_url

_CELEX_RE = re.compile(r"\b(?:CELEX\s*:?\s*)?([0-9A-Z]{4,5}:[0-9A-Z()./-]+)\b", re.IGNORECASE)
_TYPE_RE = re.compile(r"\b(regulation|directive|decision|guidance|recommendation)\b", re.IGNORECASE)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _text_for(root: ElementTree.Element, names: set[str]) -> str:
    for element in root.iter():
        if _local_name(element.tag) in names and element.text:
            return " ".join(element.text.split())
    return ""


class EURLexAdapter:
    """Fetch explicitly configured EUR-Lex/CELLAR documents, never the whole site."""

    name = "eurlex"

    def discovery_requests(self, target: SourceTarget) -> Iterable[scrapy.Request]:
        options = target.options
        urls = [
            *options.get("document_urls", []),
            *options.get("cellar_urls", []),
        ]
        urls.extend(
            f"https://eur-lex.europa.eu/legal-content/{target.language.upper()}/TXT/?uri=CELEX:{celex}"
            for celex in options.get("celex_ids", [])
        )
        urls.extend(target.start_urls)
        for url in dict.fromkeys(str(url).strip() for url in urls if str(url).strip()):
            if not self._allowed_url(url, target):
                continue
            yield scrapy.Request(
                url,
                dont_filter=True,
                meta={
                    "discovery": True,
                    "discovery_kind": "document",
                    "source_id": target.source_id,
                },
            )

    def parse_discovery(
        self, response: Response, target: SourceTarget
    ) -> Iterable[DiscoveredDocument]:
        title = response.css("title::text").get(default="").strip()
        published_at = ""
        legal_identifier = ""
        if self._is_xml(response):
            try:
                root = ElementTree.fromstring(response.body)
            except ElementTree.ParseError:
                root = None
            if root is not None:
                title = _text_for(root, {"title", "doctitle"}) or title
                published_at = _text_for(
                    root,
                    {"date", "datedocument", "datepublication", "publicationdate"},
                )
                legal_identifier = _text_for(
                    root,
                    {"celex", "resource_legal_id_celex", "identifier"},
                )

        legal_identifier = legal_identifier or self._celex_from_url(response.url)
        yield DiscoveredDocument(
            source_id=target.source_id,
            url=canonicalize_url(response.url),
            title=title,
            document_type=self._document_type(title, response.text),
            published_at=published_at,
            legal_identifier=legal_identifier,
            language=target.language,
            discovery_url=response.url,
            external_id=legal_identifier,
            mime_type=response.headers.get("Content-Type", b"").decode(errors="ignore"),
        )

    @staticmethod
    def _is_xml(response: Response) -> bool:
        content_type = response.headers.get("Content-Type", b"").decode(errors="ignore").lower()
        return "xml" in content_type or response.body.lstrip().startswith(b"<?xml")

    @staticmethod
    def _celex_from_url(url: str) -> str:
        match = _CELEX_RE.search(url)
        return match.group(1).upper() if match else ""

    @staticmethod
    def _document_type(title: str, body: str) -> str:
        match = _TYPE_RE.search(f"{title} {body[:4000]}")
        return match.group(1).lower() if match else ""

    @staticmethod
    def _allowed_url(url: str, target: SourceTarget) -> bool:
        host = (urlparse(url).hostname or "").lower()
        domains = {domain.lstrip("*.").lower() for domain in target.allowed_domains}
        return not domains or any(host == domain or host.endswith(f".{domain}") for domain in domains)
