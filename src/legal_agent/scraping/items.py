"""Scrapy item definitions for harvested regulatory documents."""

from __future__ import annotations

import scrapy


class RegulatoryDocumentItem(scrapy.Item):
    title = scrapy.Field()
    full_text = scrapy.Field()
    jurisdiction = scrapy.Field()
    effective_date = scrapy.Field()
    source_url = scrapy.Field()
    is_pdf = scrapy.Field()
    raw_pdf_bytes = scrapy.Field()


class ChunkedRegulationItem(scrapy.Item):
    """Produced after chunking – one item per chunk."""
    text = scrapy.Field()
    header_path = scrapy.Field()
    source_url = scrapy.Field()
    jurisdiction = scrapy.Field()
    effective_date = scrapy.Field()
    is_processed = scrapy.Field()
    chunk_index = scrapy.Field()
