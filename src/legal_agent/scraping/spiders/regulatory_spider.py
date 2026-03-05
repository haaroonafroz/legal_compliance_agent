"""Generic regulatory spider driven by the targets.json configuration."""

from __future__ import annotations
import trafilatura

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import scrapy
from scrapy.http import Response

from legal_agent.scraping.items import RegulatoryDocumentItem

logger = logging.getLogger(__name__)


class RegulatorySpider(scrapy.Spider):
    name = "regulatory"

    def __init__(self, sources_file: str = "data/sources/targets.json", **kwargs: Any):
        super().__init__(**kwargs)
        self.sources_file = sources_file

    def start_requests(self) -> Iterator[scrapy.Request]:
        path = Path(self.sources_file)
        if not path.exists():
            logger.error("Sources file not found: %s", path)
            return

        targets = json.loads(path.read_text(encoding="utf-8"))
        for target in targets:
            jurisdiction = target["jurisdiction"]
            follow_pdf = target.get("follow_pdf", False)
            for url in target["start_urls"]:
                yield scrapy.Request(
                    url,
                    callback=self.parse,
                    cb_kwargs={"jurisdiction": jurisdiction, "follow_pdf": follow_pdf},
                )

    def parse(
        self,
        response: Response,
        jurisdiction: str = "",
        follow_pdf: bool = False,
    ) -> Iterator[RegulatoryDocumentItem | scrapy.Request]:
        content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore")

        if "application/pdf" in content_type:
            yield RegulatoryDocumentItem(
                title=response.url.split("/")[-1],
                full_text="",
                jurisdiction=jurisdiction,
                effective_date="",
                source_url=response.url,
                is_pdf=True,
                raw_pdf_bytes=response.body,
            )
            return

        title = response.css("title::text").get(default="").strip()
        body_text = trafilatura.extract(
            response.text,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            favor_recall=True,
        )

        if body_text:
            yield RegulatoryDocumentItem(
                title=title or trafilatura.extract_metadata(response.text).title or "",
                full_text=body_text,
                jurisdiction=jurisdiction,
                effective_date="",
                source_url=response.url,
                is_pdf=False,
                raw_pdf_bytes=b"",
            )

        if follow_pdf:
            for href in response.css("a[href$='.pdf']::attr(href)").getall():
                yield response.follow(
                    href,
                    callback=self.parse,
                    cb_kwargs={"jurisdiction": jurisdiction, "follow_pdf": False},
                )
