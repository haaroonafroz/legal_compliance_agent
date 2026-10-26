"""Tests for provenance-bearing report rendering."""

from legal_agent.utils.report import _render_markdown


def test_report_includes_workflow_and_retrieval_provenance():
    rendered = _render_markdown(
        {
            "status": "review_required",
            "passed": False,
            "jurisdiction": "EU",
            "source_url": "https://example.com/regulation",
            "gap_analysis": "A gap",
            "proposed_updates": "An update",
            "audit_notes": "FAIL: verify deadline",
            "audit_retries": 2,
            "reranker_used": True,
            "matched_policies": [
                {
                    "policy_id": "POL-1",
                    "chunk_id": "chunk-1",
                    "header_path": "Retention",
                }
            ],
            "retrieval_scores": [0.91],
        }
    )

    assert "REVIEW_REQUIRED" in rendered
    assert "POL-1" in rendered
    assert "0.9100" in rendered
    assert "Reranker Used | True" in rendered


def test_review_required_markdown_exposes_obligation_evidence():
    rendered = _render_markdown(
        {
            "document_id": "doc-1",
            "status": "review_required",
            "skipped": True,
            "reason": "uncertain_policy_coverage",
            "obligation_evidence": [
                {
                    "obligation_id": "obligation-1",
                    "obligation": {"quote": "The controller shall notify."},
                    "sufficient": True,
                    "matched_policies": [{"chunk_id": "policy-1"}],
                }
            ],
            "assessments": [
                {
                    "obligation_id": "obligation-1",
                    "outcome": "uncertain",
                    "explanation": "Deadline missing from policy",
                }
            ],
        }
    )
    assert "uncertain_policy_coverage" in rendered
    assert "policy-1" in rendered
    assert "Deadline missing from policy" in rendered
    assert "NOT RUN" in rendered
