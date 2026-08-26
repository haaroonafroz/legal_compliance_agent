"""Tests for the hierarchical chunking pipeline."""

from legal_agent.scraping.items import RegulatoryDocumentItem
from legal_agent.scraping.pipelines import ChunkingPipeline


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join("word" for _ in token_ids)


def _pipeline(max_tokens=512):
    pipeline = ChunkingPipeline(tokenizer_model="test-tokenizer", max_tokens=max_tokens)
    pipeline.tokenizer = _WhitespaceTokenizer()
    return pipeline


def _make_item(**kwargs):
    defaults = {
        "title": "Test",
        "full_text": "",
        "jurisdiction": "EU",
        "effective_date": "",
        "source_url": "https://example.com",
        "is_pdf": False,
        "raw_pdf_bytes": b"",
    }
    defaults.update(kwargs)
    return RegulatoryDocumentItem(**defaults)


def test_simple_chunking():
    text = "# Section 1\n\nSome content here.\n\n## Subsection A\n\nMore detail."
    pipeline = _pipeline()
    item = _make_item(full_text=text)
    chunks = pipeline.process_item(item, spider=None)
    assert len(chunks) >= 2
    assert chunks[0]["header_path"]


def test_empty_text_returns_no_chunks():
    pipeline = _pipeline()
    item = _make_item(full_text="")
    result = pipeline.process_item(item, spider=None)
    assert result == []


def test_large_section_is_split():
    text = " ".join(["word"] * 1200)
    pipeline = _pipeline()
    item = _make_item(full_text=text)
    chunks = pipeline.process_item(item, spider=None)
    assert len(chunks) >= 2
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["token_count"] <= 512 for chunk in chunks)
