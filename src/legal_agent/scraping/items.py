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
    # Document-level metadata enrichment
    topic_tags = scrapy.Field()
    compliance_domain = scrapy.Field()
    applies_to_departments = scrapy.Field()
    obligation_type = scrapy.Field()



class ChunkedRegulationItem(scrapy.Item):
    text = scrapy.Field()
    header_path = scrapy.Field()
    source_url = scrapy.Field()
    jurisdiction = scrapy.Field()
    effective_date = scrapy.Field()
    is_processed = scrapy.Field()
    # Document linkage
    document_id = scrapy.Field()
    document_hash = scrapy.Field()
    chunk_id = scrapy.Field()
    prev_chunk_id = scrapy.Field()
    next_chunk_id = scrapy.Field()
    chunk_index = scrapy.Field()
    chunk_count = scrapy.Field()
    token_count = scrapy.Field()
    # Copied document-level metadata
    topic_tags = scrapy.Field()
    compliance_domain = scrapy.Field()
    applies_to_departments = scrapy.Field()
    obligation_type = scrapy.Field()