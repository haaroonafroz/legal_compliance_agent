"""Human review is an accountable PostgreSQL transition."""

import json

from click.testing import CliRunner

from legal_agent.cli import cli
from legal_agent.utils.report import write_report


def _invoke(report, decision="approve"):
    return CliRunner().invoke(cli, [
        "review-report", str(report), "--decision", decision,
        "--reviewer", "reviewer-1", "--reason", "Checked source evidence",
    ])


def test_human_approval_updates_durable_report_and_local_export(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    initial = {
        "document_id": "doc-1", "document_hash": "hash-1",
        "status": "pending_human_review", "passed": True,
    }
    write_report(path, initial)
    calls = []

    class FakeManifest:
        def __init__(self, _url):
            pass

        def get_report(self, _document_id):
            return initial

        def review_report(self, document_id, document_hash, expected, decision, reviewer, reason):
            calls.append((document_id, document_hash, expected, decision, reviewer, reason))
            return {**initial, "status": "approved", "review": {"reviewer": reviewer}}

    monkeypatch.setattr("legal_agent.db.manifest.DocumentManifest", FakeManifest)
    monkeypatch.setattr("legal_agent.utils.state.mirror_document_state", lambda *_: None)
    result = _invoke(path.with_suffix(".md"))
    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text())["status"] == "approved"
    assert calls[0][:4] == ("doc-1", "hash-1", "pending_human_review", "approve")


def test_failed_automated_report_cannot_be_approved(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    initial = {
        "document_id": "doc-2", "document_hash": "hash-2", "status": "review_required"
    }
    write_report(path, initial)

    class FakeManifest:
        def __init__(self, _url):
            pass

        def get_report(self, _document_id):
            return initial

    monkeypatch.setattr("legal_agent.db.manifest.DocumentManifest", FakeManifest)
    result = _invoke(path)
    assert result.exit_code != 0
    assert "Only a report that passed automated checks" in result.output
    assert json.loads(path.read_text()) == initial


def test_stale_review_does_not_change_local_export(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    initial = {
        "document_id": "doc-3", "document_hash": "hash-3",
        "status": "pending_human_review",
    }
    write_report(path, initial)

    class FakeManifest:
        def __init__(self, _url):
            pass

        def get_report(self, _document_id):
            return initial

        def review_report(self, *_):
            raise ValueError("Report is stale")

    monkeypatch.setattr("legal_agent.db.manifest.DocumentManifest", FakeManifest)
    result = _invoke(path)
    assert result.exit_code != 0
    assert "Report is stale" in result.output
    assert json.loads(path.read_text()) == initial
