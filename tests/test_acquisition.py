"""Tests for source discovery and pre-ingestion content validation."""

from types import SimpleNamespace

from scrapy import Request
from scrapy.http import TextResponse

from legal_agent.scraping.adapters.base import SourceTarget
from legal_agent.scraping.adapters.edpb import EDPBAdapter
from legal_agent.scraping.adapters.eurlex import EURLexAdapter
from legal_agent.scraping.content_validation import (
    classify_borderline,
    validate_deterministic,
)


def _target(**kwargs):
    values = {
        "source_id": "edpb-publications",
        "jurisdiction": "EU",
        "allowed_domains": ["edpb.europa.eu"],
        "document_path_patterns": [r"/our-work-tools/our-documents/"],
        "document_types": ["guidelines", "recommendations", "opinions"],
    }
    values.update(kwargs)
    return SourceTarget(**values)


def test_edpb_feed_extracts_only_allowed_document_types():
    body = b"""<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>Guidelines 01/2026 on testing</title>
        <guid>guideline-1</guid>
        <pubDate>Wed, 01 Jan 2026 00:00:00 GMT</pubDate>
        <link>https://edpb.europa.eu/our-work-tools/our-documents/guidelines-01_en</link>
      </item>
      <item>
        <title>EDPB news announcement</title>
        <guid>news-1</guid>
        <link>https://edpb.europa.eu/news/news_en</link>
      </item>
    </channel></rss>"""
    response = TextResponse(
        url="https://edpb.europa.eu/our-work-tools/our-documents/feed",
        body=body,
        encoding="utf-8",
        request=None,
    )

    documents = list(EDPBAdapter().parse_discovery(response, _target(feed_urls=(response.url,))))

    assert len(documents) == 1
    assert documents[0].document_type == "guidelines"
    assert documents[0].external_id == "guideline-1"


def test_edpb_listing_ignores_unrelated_links():
    response = TextResponse(
        url="https://edpb.europa.eu/our-work-tools/our-documents/publication-type/guidelines_en",
        body=b"""
        <main>
          <a href="/our-work-tools/our-documents/guidelines-01_en">Guidelines 01/2026</a>
          <a href="/news/news_en">News</a>
          <a href="/contact/">Contact</a>
        </main>
        """,
        encoding="utf-8",
        request=None,
    )

    documents = list(EDPBAdapter().parse_discovery(response, _target()))

    assert [document.document_type for document in documents] == ["guidelines"]


def test_edpb_redirected_feed_falls_back_to_html_listing():
    request = Request(
        "https://edpb.europa.eu/news/latest-publications/rss.xml",
        meta={"discovery": True, "discovery_kind": "feed"},
    )
    response = TextResponse(
        url="https://www.edpb.europa.eu/news_en",
        body=b"""
        <main>
          <a href="/our-work-tools/our-documents/guidelines-01_en">
            Guidelines 01/2026
          </a>
        </main>
        """,
        encoding="utf-8",
        request=request,
    )

    documents = list(EDPBAdapter().parse_discovery(response, _target()))

    assert len(documents) == 1
    assert documents[0].document_type == "guidelines"


def test_eurlex_adapter_is_configuration_driven():
    target = SourceTarget(
        source_id="gdpr",
        jurisdiction="EU",
        allowed_domains=("eur-lex.europa.eu",),
        language="en",
        adapter="eurlex",
        options={"celex_ids": ["32016R0679"]},
    )

    requests = list(EURLexAdapter().discovery_requests(target))

    assert len(requests) == 1
    assert "CELEX:32016R0679" in requests[0].url


def test_content_validation_accepts_substantive_legal_text():
    result = validate_deterministic(
        title="Guidelines on data protection",
        text=("These guidelines explain obligations for controllers and processors. " * 20),
    )

    assert result.accepted is True
    assert result.borderline is False


def test_content_validation_marks_unknown_content_borderline():
    result = validate_deterministic(
        title="Document",
        text=("This page contains a substantive document with detailed analysis. " * 20),
    )

    assert result.accepted is False
    assert result.borderline is True


def test_borderline_classifier_requires_confidence_threshold(monkeypatch):
    class _Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"relevant": true, "confidence": 0.7, "reason": "uncertain"}'
                        )
                    )
                ]
            )

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr("legal_agent.scraping.content_validation.OpenAI", _Client)
    settings = SimpleNamespace(
        use_local_llm=False,
        openai_api_key="test",
        content_validation_llm_model="gpt-4o-mini",
        content_validation_confidence_threshold=0.8,
    )

    result = classify_borderline(title="Document", text="Content", settings=settings)

    assert result.accepted is False
    assert result.confidence == 0.7
