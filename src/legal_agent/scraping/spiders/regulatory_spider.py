"""Generic regulatory spider driven by the targets.json configuration.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator

import scrapy
import trafilatura
from scrapy.exceptions import NotSupported
from scrapy.http import Response

from legal_agent.scraping.items import RegulatoryDocumentItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect brotli support at import time so Accept-Encoding is accurate.
# Scrapy's HttpCompressionMiddleware raises a WARNING (and returns garbled
# bytes) when it receives 'br'-encoded content without brotlicffi/brotli.
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
# Browser-grade headers that prevent most "connection lost" rejections.
# Rotate User-Agent strings to reduce fingerprinting.
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
    # Set dynamically based on whether brotli is available (see above).
    "Accept-Encoding": _ACCEPT_ENCODING,
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    # Sec-Fetch-* headers are checked by Cloudflare and similar WAFs.
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
}


def _browser_headers(referer: str | None = None) -> dict[str, str]:
    """Return a headers dict with a random User-Agent and optional Referer."""
    headers = dict(_BASE_HEADERS)
    headers["User-Agent"] = random.choice(_USER_AGENTS)
    if referer:
        headers["Referer"] = referer
        headers["Sec-Fetch-Site"] = "same-origin"
    return headers


class RegulatorySpider(scrapy.Spider):
    """Scrape regulatory/legal documents from a targets.json config file.

    Spider-level custom_settings override the project-wide Scrapy settings
    for everything connection/retry related so this spider can be hardened
    independently of other spiders in the project.
    """

    name = "regulatory"

    # ------------------------------------------------------------------
    # Spider-scoped Scrapy settings (merged on top of project settings).
    # ------------------------------------------------------------------
    custom_settings: dict[str, Any] = {
        # --- connection robustness ---
        # Force HTTP/1.1.  Many twisted builds negotiate h2 poorly with sites
        # that do early connection termination, producing ConnectionLost.
        "DOWNLOAD_HANDLERS": {
            "https": "scrapy.core.downloader.handlers.http11.HTTP11DownloadHandler",
            "http": "scrapy.core.downloader.handlers.http11.HTTP11DownloadHandler",
        },
        "DOWNLOAD_TIMEOUT": 45,          # seconds; slow gov sites need headroom
        "REACTOR_THREADPOOL_MAXSIZE": 20,

        # --- retry ---
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 5,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        # Also retry on the twisted connection exceptions that show up as
        # ConnectionLost / ConnectionRefusedError / TCPTimedOutError.
        "RETRY_EXCEPTIONS": [
            "twisted.internet.error.ConnectionLost",
            "twisted.internet.error.ConnectionRefusedError",
            "twisted.internet.error.TCPTimedOutError",
            "twisted.internet.defer.TimeoutError",
            "scrapy.core.downloader.handlers.http11.TunnelError",
            "builtins.IOError",
        ],

        # --- politeness / throttle ---
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 2.0,
        "AUTOTHROTTLE_MAX_DELAY": 30.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,  # one request in-flight per domain
        "AUTOTHROTTLE_DEBUG": False,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "DOWNLOAD_DELAY": 2,             # base delay between requests (seconds)
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,

        # --- misc ---
        "COOKIES_ENABLED": True,         # keep session cookies; some sites require it
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
            follow_pdf: bool = target.get("follow_pdf", False)
            for url in target["start_urls"]:
                # Jitter the start of each domain crawl so requests are not
                # all fired at t=0, which looks like a burst to WAFs.
                yield scrapy.Request(
                    url,
                    callback=self.parse,
                    headers=_browser_headers(),
                    cb_kwargs={"jurisdiction": jurisdiction, "follow_pdf": follow_pdf},
                    # Don't let Scrapy override our hand-crafted headers with the
                    # project-level DEFAULT_REQUEST_HEADERS.
                    dont_filter=False,
                    meta={
                        "download_timeout": 45,
                        "handle_httpstatus_list": [403, 404, 410, 429],
                    },
                )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(
        self,
        response: Response,
        jurisdiction: str = "",
        follow_pdf: bool = False,
    ) -> Iterator[RegulatoryDocumentItem | scrapy.Request]:
        """Dispatch to PDF or HTML handler based on Content-Type."""

        # ---- status-code guard ----------------------------------------
        if response.status == 429:
            logger.warning(
                "Rate-limited (429) on %s – will be retried by middleware.", response.url
            )
            return
        if response.status in (403, 410):
            logger.warning("Blocked/gone (%s) on %s – skipping.", response.status, response.url)
            return

        content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore")

        if "application/pdf" in content_type:
            yield from self._handle_pdf(response, jurisdiction)
            return

        yield from self._handle_html(response, jurisdiction, follow_pdf)

    # ------------------------------------------------------------------
    # PDF handler
    # ------------------------------------------------------------------

    def _handle_pdf(
        self,
        response: Response,
        jurisdiction: str,
    ) -> Iterator[RegulatoryDocumentItem]:
        logger.info("PDF document: %s", response.url)
        yield RegulatoryDocumentItem(
            title=response.url.split("/")[-1],
            full_text="",           # pipeline handles PDF text extraction
            jurisdiction=jurisdiction,
            effective_date="",
            source_url=response.url,
            is_pdf=True,
            raw_pdf_bytes=response.body,
        )

    # ------------------------------------------------------------------
    # HTML handler
    # ------------------------------------------------------------------

    def _get_response_text(self, response: Response) -> str | None:
        """Return response body as a Unicode string, handling all failure modes.

        Scrapy raises ``scrapy.exceptions.NotSupported`` with the message
        "Response content isn't text" when HttpCompressionMiddleware could not
        decode the body (e.g. 'br' encoding without brotlicffi installed).
        Rather than crashing the spider we attempt a graceful decode from raw
        bytes and log a clear action-item for the operator.
        """
        try:
            return response.text
        except NotSupported:
            encoding = response.headers.get("Content-Encoding", b"").decode("utf-8", errors="ignore")
            logger.error(
                "Cannot decode response from %s (Content-Encoding: %s). "
                "Install brotli support with: pip install brotlicffi",
                response.url,
                encoding or "unknown",
            )
            # Last-ditch attempt: try decoding raw bytes as UTF-8/latin-1.
            try:
                return response.body.decode("utf-8", errors="replace")
            except Exception:
                return None

    def _handle_html(
        self,
        response: Response,
        jurisdiction: str,
        follow_pdf: bool,
    ) -> Iterator[RegulatoryDocumentItem | scrapy.Request]:
        # --- safely obtain response text ---------------------------------
        response_text = self._get_response_text(response)
        if not response_text:
            logger.warning("Skipping %s – could not decode response body.", response.url)
            return

        # --- title -------------------------------------------------------
        title = response.css("title::text").get(default="").strip()

        # --- main-content extraction with trafilatura --------------------
        extracted: str | None = None
        try:
            extracted = trafilatura.extract(
                response_text,
                output_format="markdown",
                include_links=True,
                include_tables=True,
                favor_recall=True,
                no_fallback=False,      # allow trafilatura's own fallback heuristics
            )
        except Exception:
            logger.exception("trafilatura extraction failed for %s", response.url)

        # Fallback: grab all visible paragraph text so we never lose a page
        # even when trafilatura cannot parse the DOM.
        if not extracted:
            logger.warning(
                "trafilatura returned nothing for %s – using CSS fallback.", response.url
            )
            paragraphs = response.css("p::text, li::text, h1::text, h2::text, h3::text").getall()
            extracted = "\n".join(p.strip() for p in paragraphs if p.strip())

        # --- title fallback via trafilatura metadata ---------------------
        if not title:
            try:
                meta = trafilatura.extract_metadata(response_text)
                if meta and meta.title:
                    title = meta.title
            except Exception:
                pass

        if not title:
            title = response.url.split("/")[-1] or response.url

        # --- yield item --------------------------------------------------
        if extracted:
            logger.info("Extracted %d chars from %s", len(extracted), response.url)
            yield RegulatoryDocumentItem(
                title=title,
                full_text=extracted,
                jurisdiction=jurisdiction,
                effective_date="",
                source_url=response.url,
                is_pdf=False,
                raw_pdf_bytes=b"",
            )
        else:
            logger.warning("No content extracted from %s – skipping item.", response.url)

        # --- follow PDF links on the page --------------------------------
        if follow_pdf:
            for href in response.css("a[href$='.pdf']::attr(href)").getall():
                yield response.follow(
                    href,
                    callback=self.parse,
                    headers=_browser_headers(referer=response.url),
                    cb_kwargs={"jurisdiction": jurisdiction, "follow_pdf": False},
                    meta={"download_timeout": 45},
                )