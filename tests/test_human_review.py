"""A model PASS never becomes human approval without an explicit review action."""

import json

from click.testing import CliRunner

from legal_agent.cli import cli
from legal_agent.utils.report import write_report


def test_human_approval_records_reviewer_and_updates_state(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    write_report(
        report,
        {
            "document_id": "doc-1",
            "status": "pending_human_review",
            "passed": True,
            "source_url": "https://example.com/law",
        },
    )
    recorded = []
    monkeypatch.setattr(
        "legal_agent.utils.state.mark_document_state",
        lambda *_args, **_kwargs: recorded.append((_args[1], _args[2])),
    )
    # The command imports the function directly, so patch its defining module before invoke.
    result = CliRunner().invoke(
        cli,
        [
            "review-report",
            str(report.with_suffix(".md")),
            "--decision",
            "approve",
            "--reviewer",
            "reviewer-1",
            "--reason",
            "Checked against the source and policy owners",
        ],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["status"] == "approved"
    assert saved["review"]["reviewer"] == "reviewer-1"
    assert saved["review"]["reviewed_at"]
    assert recorded == [("doc-1", "approved")]


def test_failed_automated_report_cannot_be_approved(tmp_path):
    report = tmp_path / "report.json"
    write_report(report, {"document_id": "doc-2", "status": "review_required"})
    result = CliRunner().invoke(
        cli,
        [
            "review-report",
            str(report),
            "--decision",
            "approve",
            "--reviewer",
            "reviewer-1",
            "--reason",
            "Test",
        ],
    )
    assert result.exit_code != 0
    assert "Only a report that passed automated checks" in result.output
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "review_required"


def test_failed_state_update_restores_unreviewed_report(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    original = {
        "document_id": "doc-3",
        "document_hash": "hash-3",
        "status": "pending_human_review",
    }
    write_report(report, original)

    def fail_state(*_args, **_kwargs):
        raise OSError("Qdrant unavailable")

    monkeypatch.setattr("legal_agent.utils.state.mark_document_state", fail_state)
    result = CliRunner().invoke(
        cli,
        [
            "review-report",
            str(report),
            "--decision",
            "approve",
            "--reviewer",
            "reviewer-1",
            "--reason",
            "Checked evidence",
        ],
    )
    assert result.exit_code != 0
    assert json.loads(report.read_text(encoding="utf-8")) == original
