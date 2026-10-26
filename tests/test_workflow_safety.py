"""Tests for fail-safe extraction, analysis, and durable state ordering."""

import json
from types import SimpleNamespace

import pytest
from llama_index.core.workflow import StopEvent

from legal_agent.config import Settings
from legal_agent.utils import state
from legal_agent.workflow.events import NewLawEvent, RelevantContextEvent
from legal_agent.workflow.workflow import ComplianceWorkflow


class FakeStore:
    def __init__(self):
        self.values = {}

    async def set(self, key, value):
        self.values[key] = value

    async def get(self, key, default=None):
        return self.values.get(key, default)


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)

    async def achat(self, messages):
        return SimpleNamespace(message=SimpleNamespace(content=next(self.replies)))


def _law():
    return NewLawEvent(
        document_id="doc-1",
        document_hash="hash-1",
        regulation_text="First duty. Second duty.",
        regulation_chunks=["First duty.", "Second duty."],
        header_path="",
        jurisdiction="EU",
        source_url="https://example.com/law",
        effective_date="",
    )


def _workflow(monkeypatch, replies):
    monkeypatch.setattr("legal_agent.workflow.workflow.get_llm_for_step", lambda *_: FakeLLM(replies))
    monkeypatch.setattr("legal_agent.workflow.workflow.client_from_settings", lambda *_: object())
    return ComplianceWorkflow(Settings(_env_file=None), timeout=10)


@pytest.mark.asyncio
async def test_extraction_reads_every_chunk_and_deduplicates(monkeypatch):
    replies = [
        json.dumps({"obligations": [{"quote": "First duty.", "action": "First"}]}),
        json.dumps({"obligations": [{"quote": "Second duty.", "action": "Second"}]}),
    ]
    workflow = _workflow(monkeypatch, replies)
    result = await workflow.obligation_extractor(SimpleNamespace(store=FakeStore()), _law())
    assert [item.quote for item in result.obligations] == ["First duty.", "Second duty."]
    assert [item.obligation_id for item in result.obligations] == ["obligation-1", "obligation-2"]


@pytest.mark.asyncio
async def test_invalid_extraction_stops_for_review(monkeypatch):
    workflow = _workflow(monkeypatch, ['{"obligations": []}', "not JSON"])
    result = await workflow.obligation_extractor(SimpleNamespace(store=FakeStore()), _law())
    assert isinstance(result, StopEvent)
    assert result.result["status"] == "review_required"


@pytest.mark.asyncio
async def test_incomplete_analysis_stops_for_review(monkeypatch):
    workflow = _workflow(monkeypatch, ['{"assessments": []}'])
    bundle = {
        "obligation_id": "obligation-1",
        "obligation": {"quote": "First duty.", "action": "First"},
        "matched_policies": [{"chunk_id": "p1", "text": "Policy text"}],
        "sufficient": True,
    }
    event = RelevantContextEvent(
        regulation=_law(),
        matched_policies=bundle["matched_policies"],
        retrieval_scores=[0.5],
        obligation_evidence=[bundle],
    )
    result = await workflow.analyst(SimpleNamespace(store=FakeStore()), event)
    assert isinstance(result, StopEvent)
    assert result.result["reason"].startswith("invalid_analysis")


def test_result_is_not_marked_processed_if_report_write_fails(monkeypatch):
    calls = []

    def fail_save(result):
        calls.append("save")
        raise OSError("disk full")

    monkeypatch.setattr(state, "save_report", fail_save)
    monkeypatch.setattr(state, "mark_document_state", lambda *_args, **_kwargs: calls.append("mark"))
    with pytest.raises(OSError, match="disk full"):
        state.persist_workflow_result(
            Settings(_env_file=None), {"document_id": "doc-1", "status": "review_required"}
        )
    assert calls == ["save"]


def test_result_is_marked_only_after_report_is_saved(monkeypatch, tmp_path):
    calls = []
    report = tmp_path / "doc-1.md"

    def save(result):
        calls.append("save")
        return report

    def mark(_settings, document_id, status, **kwargs):
        calls.append("mark")
        assert (document_id, status) == ("doc-1", "pending_human_review")
        assert kwargs["expected_status"] == "pending"

    monkeypatch.setattr(state, "save_report", save)
    monkeypatch.setattr(state, "mark_document_state", mark)
    path = state.persist_workflow_result(
        Settings(_env_file=None),
        {"document_id": "doc-1", "document_hash": "hash-1", "status": "pending_human_review"},
    )
    assert path == report
    assert calls == ["save", "mark"]


def test_review_rejects_superseded_document():
    client = SimpleNamespace(
        scroll=lambda **_kwargs: (
            [SimpleNamespace(payload={"workflow_status": "pending_human_review", "is_current": False})],
            None,
        ),
        set_payload=lambda **_kwargs: pytest.fail("Superseded document was updated"),
    )
    with pytest.raises(ValueError, match="current document state"):
        state.mark_document_state(
            Settings(_env_file=None),
            "doc-1",
            "approved",
            client=client,
            expected_status="pending_human_review",
        )
