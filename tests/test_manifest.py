"""Tests for durable document revision state."""

import pytest
from scrapy.exceptions import DropItem

from legal_agent.db.manifest import DocumentManifest
from legal_agent.scraping.items import RegulatoryDocumentItem
from legal_agent.scraping.pipelines import ContentValidationPipeline


def test_manifest_skips_unchanged_current_revision(tmp_path):
    manifest = DocumentManifest(tmp_path / "manifest.sqlite3")
    values = {
        "source_id": "edpb",
        "source_key": "edpb::guideline-1",
        "source_url": "https://edpb.europa.eu/guideline-1",
        "document_hash": "hash-1",
        "revision_id": "revision-1",
        "title": "Guideline 1",
    }

    assert manifest.prepare(**values)["action"] == "ingest"
    manifest.set_expected_chunks("revision-1", 1)
    assert manifest.record_chunks("revision-1", 1) is True
    assert manifest.mark_stored("revision-1") == ""
    assert manifest.prepare(**values)["action"] == "unchanged"


def test_manifest_supersedes_previous_revision(tmp_path):
    manifest = DocumentManifest(tmp_path / "manifest.sqlite3")
    common = {
        "source_id": "edpb",
        "source_key": "edpb::guideline-1",
        "source_url": "https://edpb.europa.eu/guideline-1",
        "title": "Guideline 1",
    }

    first = manifest.prepare(**common, document_hash="hash-1", revision_id="revision-1")
    assert first["supersedes_revision_id"] == ""
    manifest.set_expected_chunks("revision-1", 1)
    manifest.record_chunks("revision-1", 1)
    manifest.mark_stored("revision-1")

    second = manifest.prepare(**common, document_hash="hash-2", revision_id="revision-2")
    assert second["supersedes_revision_id"] == "revision-1"
    manifest.set_expected_chunks("revision-2", 1)
    manifest.record_chunks("revision-2", 1)
    assert manifest.mark_stored("revision-2") == "revision-1"


def test_content_pipeline_registers_revision_and_skips_it_next_time(tmp_path):
    settings = type(
        "Settings",
        (),
        {
            "content_validation_min_chars": 100,
            "content_validation_llm_enabled": False,
            "manifest_file": str(tmp_path / "manifest.sqlite3"),
            "quarantine_file": str(tmp_path / "quarantine.jsonl"),
        },
    )()
    item = RegulatoryDocumentItem(
        title="Guidelines on data protection",
        full_text="These guidelines describe obligations for controllers. " * 10,
        jurisdiction="EU",
        effective_date="",
        source_url="https://edpb.europa.eu/guidelines/1",
        is_pdf=False,
        raw_pdf_bytes=b"",
    )

    pipeline = ContentValidationPipeline()
    pipeline._settings = settings
    accepted = pipeline.process_item(item, spider=None)
    assert accepted["revision_id"]
    assert accepted["source_key"].startswith("generic::")
    manifest = DocumentManifest(settings.manifest_file)
    manifest.set_expected_chunks(accepted["revision_id"], 1)
    manifest.record_chunks(accepted["revision_id"], 1)
    manifest.mark_stored(accepted["revision_id"])

    with pytest.raises(DropItem, match="unchanged document"):
        pipeline.process_item(item, spider=None)
