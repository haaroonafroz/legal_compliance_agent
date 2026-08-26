"""Tests for regulatory spider request generation."""

import json

import pytest

from legal_agent.scraping.spiders.regulatory_spider import RegulatorySpider


@pytest.mark.asyncio
async def test_start_yields_requests_from_sources_file(tmp_path):
    sources_file = tmp_path / "targets.json"
    sources_file.write_text(
        json.dumps(
            [
                {
                    "jurisdiction": "EU",
                    "start_urls": ["https://example.com/guidance"],
                }
            ]
        ),
        encoding="utf-8",
    )

    spider = RegulatorySpider(sources_file=str(sources_file))
    requests = [request async for request in spider.start()]

    assert [request.url for request in requests] == ["https://example.com/guidance"]
