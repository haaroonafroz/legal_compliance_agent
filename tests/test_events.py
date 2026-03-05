"""Smoke tests for workflow event construction."""

from legal_agent.workflow.events import FinalReportEvent, NewLawEvent, RetrievedContextEvent


def test_new_law_event_construction():
    ev = NewLawEvent(
        regulation_text="Test regulation",
        header_path="Section 1 > Article 2",
        jurisdiction="EU",
        source_url="https://example.com/reg",
        effective_date="2026-01-01",
        chunk_ids=[1, 2, 3],
    )
    assert ev.jurisdiction == "EU"
    assert len(ev.chunk_ids) == 3


def test_retrieved_context_event():
    law = NewLawEvent(
        regulation_text="Test",
        header_path="",
        jurisdiction="California",
        source_url="https://example.com",
        effective_date="",
    )
    ev = RetrievedContextEvent(
        regulation=law,
        matched_policies=[{"text": "policy A", "policy_id": "POL-001"}],
        retrieval_scores=[0.92],
    )
    assert len(ev.matched_policies) == 1


def test_final_report_event():
    ev = FinalReportEvent(
        jurisdiction="EU",
        source_url="https://example.com",
        gap_analysis="Gaps found.",
        proposed_updates="Update section 3.",
        audit_notes="PASS – all clear.",
        passed=True,
    )
    assert ev.passed is True
