"""Generic regulatory spider driven by the targets.json configuration.

Crawl model
-----------
Each entry in targets.json defines one independent crawl "job".  The spider
tracks state per-job (keyed by start_url) so limits apply individually:

    {
      "name": "European Data Protection Board",
      "jurisdiction": "EU",
      "start_urls": ["https://www.edpb.europa.eu/our-work-tools/our-documents/publication-type/guidelines_en"],
      "allowed_domains": ["edpb.europa.eu"],
      "follow_links": true,
      "follow_pdf":   true,
      "max_depth":    2,
      "max_pages":    50,
      "max_pdfs":     20
    }

Per-target field reference
--------------------------
follow_links    bool  Follow <a> href links on HTML pages.   default: false
follow_pdf      bool  Download .pdf links found on pages.    default: false
max_depth       int   Max hops away from start_url.          default: 2
max_pages       int   HTML pages to visit per job. 0 = ∞.    default: 0
max_pdfs        int   PDFs to download per job.    0 = ∞.    default: 0
allowed_domains list  Only follow links within these domains. default: []

Connection hardening
--------------------
- Browser-grade headers + rotating User-Agent
- Accept-Encoding: br only when brotlicffi/brotli is installed
- Forced HTTP/1.1 (avoids twisted h2 ConnectionLost)
- AutoThrottle + base DOWNLOAD_DELAY
- Extended retry list (ConnectionLost, TCPTimedOutError, …)
- NotSupported guard on response.text with UTF-8 raw-bytes fallback
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import scrapy
import trafilatura
from scrapy.exceptions import NotSupported
from scrapy.http import Response

from legal_agent.scraping.items import RegulatoryDocumentItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brotli availability
# ---------------------------------------------------------------------------

def _brotli_available() -> bool:
    for mod in ("brotlicffi", "brotli"):
        try:
            __import__(mod)
            return True
        except ImportError:
            pass
    return False


_BROTLI_OK = _brotli_available()
if not _BROTLI_OK:
    logger.warning(
        "brotlicffi/brotli not installed – 'br' excluded from Accept-Encoding. "
        "Install with: pip install brotlicffi"
    )

_ACCEPT_ENCODING = "gzip, deflate, br" if _BROTLI_OK else "gzip, deflate"

# ---------------------------------------------------------------------------
# Browser-grade headers
# ---------------------------------------------------------------------------

_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.3 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) "
        "Gecko/20100101 Firefox/124.0"
    ),
]

_BASE_HEADERS: dict[str, str] = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept-Encoding": _ACCEPT_ENCODING,
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
}


def _browser_headers(referer: str | None = None) -> dict[str, str]:
    headers = dict(_BASE_HEADERS)
    headers["User-Agent"] = random.choice(_USER_AGENTS)
    if referer:
        headers["Referer"] = referer
        headers["Sec-Fetch-Site"] = "same-origin"
    return headers


# ---------------------------------------------------------------------------
# Per-job crawl state
# ---------------------------------------------------------------------------

class _JobState:
    """Tracks quotas and config for a single targets.json start_url entry."""

    __slots__ = (
        "jurisdiction", "allowed_domains",
        "follow_links", "follow_pdf",
        "max_depth", "max_pages", "max_pdfs",
        "pages_seen", "pdfs_seen",
    )

    def __init__(
        self,
        *,
        jurisdiction: str,
        allowed_domains: list[str],
        follow_links: bool,
        follow_pdf: bool,
        max_depth: int,
        max_pages: int,
        max_pdfs: int,
    ) -> None:
        self.jurisdiction = jurisdiction
        # Strip leading wildcard so "*.europa.eu" → "europa.eu"
        self.allowed_domains = [d.lstrip("*.") for d in allowed_domains]
        self.follow_links = follow_links
        self.follow_pdf = follow_pdf
        self.max_depth = max_depth   # 0 = unlimited
        self.max_pages = max_pages   # 0 = unlimited
        self.max_pdfs = max_pdfs     # 0 = unlimited
        self.pages_seen = 0
        self.pdfs_seen = 0

    # -- quota helpers --------------------------------------------------

    def page_allowed(self) -> bool:
        return self.max_pages == 0 or self.pages_seen < self.max_pages

    def pdf_allowed(self) -> bool:
        return self.max_pdfs == 0 or self.pdfs_seen < self.max_pdfs

    def depth_allowed(self, depth: int) -> bool:
        """True if *depth* is within the configured limit (0 = unlimited)."""
        return self.max_depth == 0 or depth <= self.max_depth

    def domain_allowed(self, url: str) -> bool:
        """True if *url* is within one of the allowed_domains (empty = allow all)."""
        if not self.allowed_domains:
            return True
        host = urlparse(url).hostname or ""
        return any(
            host == d or host.endswith("." + d)
            for d in self.allowed_domains
        )

    def claim_page(self) -> None:
        self.pages_seen += 1

    def claim_pdf(self) -> None:
        self.pdfs_seen += 1


# ---------------------------------------------------------------------------
# Spider
# ---------------------------------------------------------------------------

class RegulatorySpider(scrapy.Spider):
    """Scrape regulatory/legal documents driven by targets.json."""

    name = "regulatory"

    custom_settings: dict[str, Any] = {
        # --- connection robustness ---
        "DOWNLOAD_HANDLERS": {
            "https": "scrapy.core.downloader.handlers.http11.HTTP11DownloadHandler",
            "http": "scrapy.core.downloader.handlers.http11.HTTP11DownloadHandler",
        },
        "DOWNLOAD_TIMEOUT": 45,
        "REACTOR_THREADPOOL_MAXSIZE": 20,

        # --- retry ---
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 5,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        "RETRY_EXCEPTIONS": [
            "twisted.internet.error.ConnectionLost",
            "twisted.internet.error.ConnectionRefusedError",
            "twisted.internet.error.TCPTimedOutError",
            "twisted.internet.defer.TimeoutError",
            "scrapy.core.downloader.handlers.http11.TunnelError",
            "builtins.IOError",
        ],

        # --- politeness ---
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 2.0,
        "AUTOTHROTTLE_MAX_DELAY": 30.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "AUTOTHROTTLE_DEBUG": False,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "DOWNLOAD_DELAY": 2,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,

        # --- misc ---
        "COOKIES_ENABLED": True,
        "ROBOTSTXT_OBEY": False,
        "REDIRECT_ENABLED": True,
        "REDIRECT_MAX_TIMES": 5,
        "LOG_LEVEL": "ERROR",
    }

    # ------------------------------------------------------------------

    def __init__(
        self,
        sources_file: str = "data/sources/targets.json",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sources_file = sources_file
        # job_key (= start_url) → _JobState
        self._jobs: dict[str, _JobState] = {}

    # ------------------------------------------------------------------
    # Request generation
    # ------------------------------------------------------------------

    def start_requests(self) -> Iterator[scrapy.Request]:
        path = Path(self.sources_file)
        if not path.exists():
            logger.error("Sources file not found: %s", path)
            return

        targets: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

        for target in targets:
            jurisdiction: str = target["jurisdiction"]
            allowed_domains: list[str] = target.get("allowed_domains", [])
            follow_links: bool = bool(target.get("follow_links", False))
            follow_pdf: bool = bool(target.get("follow_pdf", False))
            max_depth: int = int(target.get("max_depth", 2))
            max_pages: int = int(target.get("max_pages", 0))
            max_pdfs: int = int(target.get("max_pdfs", 0))

            for url in target["start_urls"]:
                # Use the start URL as the job key so each entry point gets
                # its own independent quota counters.
                job_key = url
                self._jobs[job_key] = _JobState(
                    jurisdiction=jurisdiction,
                    allowed_domains=allowed_domains,
                    follow_links=follow_links,
                    follow_pdf=follow_pdf,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    max_pdfs=max_pdfs,
                )
                logger.info(
                    "Job queued: %s | depth≤%s  pages≤%s  pdfs≤%s",
                    url,
                    max_depth or "∞",
                    max_pages or "∞",
                    max_pdfs or "∞",
                )
                yield scrapy.Request(
                    url,
                    callback=self.parse,
                    headers=_browser_headers(),
                    dont_filter=False,
                    meta={
                        "job_key": job_key,
                        "depth": 0,
                        "download_timeout": 45,
                        "handle_httpstatus_list": [403, 404, 410, 429],
                    },
                )

    # ------------------------------------------------------------------
    # Main dispatcher
    # ------------------------------------------------------------------

    def parse(self, response: Response) -> Iterator[RegulatoryDocumentItem | scrapy.Request]:
        """Route each response to the PDF or HTML handler."""
        job_key: str = response.meta["job_key"]
        depth: int = response.meta.get("depth", 0)
        job: _JobState = self._jobs[job_key]

        if response.status == 429:
            logger.warning("Rate-limited (429) on %s – retrying.", response.url)
            return
        if response.status in (403, 410):
            logger.warning("Blocked/gone (%s) on %s – skipping.", response.status, response.url)
            return

        content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore")

        if "application/pdf" in content_type:
            yield from self._handle_pdf(response, job)
        else:
            yield from self._handle_html(response, job, depth)

    # ------------------------------------------------------------------
    # PDF handler
    # ------------------------------------------------------------------

    def _handle_pdf(
        self,
        response: Response,
        job: _JobState,
    ) -> Iterator[RegulatoryDocumentItem]:
        if not job.pdf_allowed():
            logger.info(
                "PDF quota (%d) reached – skipping %s",
                job.max_pdfs, response.url,
            )
            return

        job.claim_pdf()
        logger.info(
            "[pdf %d/%s] %s",
            job.pdfs_seen, job.max_pdfs or "∞", response.url,
        )
        yield RegulatoryDocumentItem(
            title=response.url.split("/")[-1],
            full_text="",
            jurisdiction=job.jurisdiction,
            effective_date="",
            source_url=response.url,
            is_pdf=True,
            raw_pdf_bytes=response.body,
        )

    # ------------------------------------------------------------------
    # HTML handler
    # ------------------------------------------------------------------

    def _get_response_text(self, response: Response) -> str | None:
        """Decode response body, gracefully handling missing brotli support."""
        try:
            return response.text
        except NotSupported:
            encoding = response.headers.get(
                "Content-Encoding", b""
            ).decode("utf-8", errors="ignore")
            logger.error(
                "Cannot decode %s (Content-Encoding: %s). "
                "Install brotli support: pip install brotlicffi",
                response.url, encoding or "unknown",
            )
            try:
                return response.body.decode("utf-8", errors="replace")
            except Exception:
                return None

    def _handle_html(
        self,
        response: Response,
        job: _JobState,
        depth: int,
    ) -> Iterator[RegulatoryDocumentItem | scrapy.Request]:
        job_key: str = response.meta["job_key"]

        # -- page quota -------------------------------------------------
        if not job.page_allowed():
            logger.info(
                "Page quota (%d) reached – skipping %s",
                job.max_pages, response.url,
            )
            return

        job.claim_page()
        logger.info(
            "[page %d/%s  depth %d/%s] %s",
            job.pages_seen, job.max_pages or "∞",
            depth, job.max_depth or "∞",
            response.url,
        )

        response_text = self._get_response_text(response)
        if not response_text:
            logger.warning("Skipping %s – could not decode body.", response.url)
            return

        # -- content extraction -----------------------------------------
        title = response.css("title::text").get(default="").strip()

        extracted: str | None = None
        try:
            extracted = trafilatura.extract(
                response_text,
                output_format="markdown",
                include_links=True,
                include_tables=True,
                favor_recall=True,
                no_fallback=False,
            )
        except Exception:
            logger.exception("trafilatura failed for %s", response.url)

        if not extracted:
            logger.warning("trafilatura empty for %s – CSS fallback.", response.url)
            paragraphs = response.css(
                "p::text, li::text, h1::text, h2::text, h3::text"
            ).getall()
            extracted = "\n".join(p.strip() for p in paragraphs if p.strip())

        if not title:
            try:
                meta = trafilatura.extract_metadata(response_text)
                if meta and meta.title:
                    title = meta.title
            except Exception:
                pass
        if not title:
            title = response.url.split("/")[-1] or response.url

        if extracted:
            logger.info("Extracted %d chars from %s", len(extracted), response.url)
            yield RegulatoryDocumentItem(
                title=title,
                full_text=extracted,
                jurisdiction=job.jurisdiction,
                effective_date="",
                source_url=response.url,
                is_pdf=False,
                raw_pdf_bytes=b"",
            )
        else:
            logger.warning("No content extracted from %s.", response.url)

        # -- link following ---------------------------------------------
        # Don't descend further if already at the depth ceiling.
        next_depth = depth + 1
        if not job.depth_allowed(next_depth):
            logger.debug("Depth limit at %s – not following links.", response.url)
            return

        # PDF links: highest value for compliance document sites.
        if job.follow_pdf:
            for href in response.css("a[href$='.pdf']::attr(href)").getall():
                abs_url = response.urljoin(href)
                if not job.domain_allowed(abs_url):
                    continue
                if not job.pdf_allowed():
                    logger.info("PDF quota reached – stopping PDF link collection.")
                    break
                yield scrapy.Request(
                    abs_url,
                    callback=self.parse,
                    headers=_browser_headers(referer=response.url),
                    meta={
                        "job_key": job_key,
                        "depth": next_depth,
                        "download_timeout": 45,
                        "handle_httpstatus_list": [403, 404, 410, 429],
                    },
                )

        # HTML page links: walk index → individual guideline pages.
        if job.follow_links:
            for href in response.css("a::attr(href)").getall():
                if href.lower().endswith(".pdf"):
                    continue  # already handled above
                abs_url = response.urljoin(href)
                if not abs_url.startswith(("http://", "https://")):
                    continue  # skip anchors, mailto, javascript:, etc.
                if not job.domain_allowed(abs_url):
                    continue
                if not job.page_allowed():
                    logger.info("Page quota reached – stopping HTML link collection.")
                    break
                yield scrapy.Request(
                    abs_url,
                    callback=self.parse,
                    headers=_browser_headers(referer=response.url),
                    dont_filter=False,
                    meta={
                        "job_key": job_key,
                        "depth": next_depth,
                        "download_timeout": 45,
                        "handle_httpstatus_list": [403, 404, 410, 429],
                    },
                )