"""Tests for fail-safe extraction, analysis, and durable state ordering."""

import json
from types import SimpleNamespace

import pytest
from llama_index.core.workflow import StopEvent

from legal_agent.config import Settings
from legal_agent.utils import state
from legal_agent.workflow.events import NewLawEvent, RelevantContextEvent, RetrievedContextEvent
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
    monkeypatch.setattr(
        "legal_agent.workflow.workflow.get_llm_for_step",
        lambda _settings, step: FakeLLM(["RELEVANT"] if step == "relevance_check" else replies),
    )
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


def test_local_export_failure_does_not_lose_durable_report(monkeypatch):
    calls = []

    class FakeManifest:
        def __init__(self, _url):
            pass

        def complete_workflow(self, revision_id, document_hash, token, report):
            calls.append(("commit", revision_id, document_hash, token, report["status"]))

    def fail_save(_result):
        raise OSError("disk full")

    monkeypatch.setattr(state, "DocumentManifest", FakeManifest)
    monkeypatch.setattr(state, "save_report", fail_save)
    monkeypatch.setattr(state, "mirror_document_state", lambda *_: None)
    result = {"document_id": "doc-1", "document_hash": "hash-1", "status": "review_required"}
    path = state.persist_workflow_result(
        Settings(_env_file=None, database_url="postgresql://test"),
        result,
        {"revision_id": "doc-1", "document_hash": "hash-1", "claim_token": "token"},
    )
    assert path.startswith("postgresql:")
    assert calls == [("commit", "doc-1", "hash-1", "token", "review_required")]


def test_failed_db_commit_prevents_local_export(monkeypatch):
    calls = []

    class FakeManifest:
        def __init__(self, _url):
            pass

        def complete_workflow(self, *_):
            raise ValueError("stale claim")

    monkeypatch.setattr(state, "DocumentManifest", FakeManifest)
    monkeypatch.setattr(state, "save_report", lambda *_: calls.append("export"))
    with pytest.raises(ValueError, match="stale claim"):
        state.persist_workflow_result(
            Settings(_env_file=None, database_url="postgresql://test"),
            {"document_id": "doc-1", "document_hash": "hash-1", "status": "review_required"},
            {"revision_id": "doc-1", "document_hash": "hash-1", "claim_token": "token"},
        )
    assert calls == []


@pytest.mark.asyncio
async def test_irrelevant_document_stops_before_extraction(monkeypatch):
    monkeypatch.setattr(
        "legal_agent.workflow.workflow.get_llm_for_step",
        lambda _settings, step: FakeLLM(["IRRELEVANT"] if step == "relevance_check" else []),
    )
    monkeypatch.setattr("legal_agent.workflow.workflow.client_from_settings", lambda *_: object())
    workflow = ComplianceWorkflow(Settings(_env_file=None), timeout=10)
    result = await workflow.obligation_extractor(SimpleNamespace(store=FakeStore()), _law())
    assert result.result["status"] == "irrelevant"


@pytest.mark.asyncio
async def test_missing_policy_is_a_gap_when_corpus_exists(monkeypatch):
    workflow = _workflow(monkeypatch, [])
    workflow.qdrant = SimpleNamespace(count=lambda **_: SimpleNamespace(count=1))
    bundle = {
        "obligation_id": "obligation-1",
        "obligation": {"quote": "First duty.", "action": "First"},
        "matched_policies": [],
        "sufficient": False,
    }
    event = RetrievedContextEvent(
        regulation=_law(), matched_policies=[], retrieval_scores=[],
        obligation_evidence=[bundle],
    )
    relevant = await workflow.relevance_check(SimpleNamespace(store=FakeStore()), event)
    assert not isinstance(relevant, StopEvent)
    result = await workflow.analyst(SimpleNamespace(store=FakeStore()), relevant)
    assert "No matching internal policy" in result.gap_analysis


@pytest.mark.asyncio
async def test_empty_policy_corpus_requires_review(monkeypatch):
    workflow = _workflow(monkeypatch, [])
    workflow.qdrant = SimpleNamespace(count=lambda **_: SimpleNamespace(count=0))
    event = RetrievedContextEvent(
        regulation=_law(), matched_policies=[], retrieval_scores=[],
        obligation_evidence=[{
            "obligation_id": "obligation-1",
            "obligation": {"quote": "First duty.", "action": "First"},
            "matched_policies": [], "sufficient": False,
        }],
    )
    result = await workflow.relevance_check(SimpleNamespace(store=FakeStore()), event)
    assert result.result["reason"] == "policy_corpus_empty"
