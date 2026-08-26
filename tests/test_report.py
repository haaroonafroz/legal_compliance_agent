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
