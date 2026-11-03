"""Explicit, non-destructive import of pre-PostgreSQL local state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

from legal_agent.db.manifest import DocumentManifest

_ALLOWED = {"pending", "irrelevant", "review_required", "pending_human_review", "approved", "rejected"}


def preview_legacy(sqlite_path: Path, reports_dir: Path) -> tuple[list[dict], dict[str, dict], list[str]]:
    """Read legacy files without writing to or removing them."""
    if not sqlite_path.is_file():
        raise FileNotFoundError(sqlite_path)
    uri = f"file:{quote(str(sqlite_path.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM document_manifest")]
    known = {row["revision_id"]: row for row in rows}
    reports: dict[str, dict] = {}
    warnings: list[str] = []
    for path in sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            document_id = report.get("document_id")
            if document_id not in known:
                warnings.append(f"Unmatched report: {path.name}")
                continue
            if report.get("document_hash") and report["document_hash"] != known[document_id]["document_hash"]:
                warnings.append(f"Hash mismatch: {path.name}")
                continue
            if report.get("status") not in _ALLOWED:
                warnings.append(f"Unknown report status: {path.name}")
                continue
            if report["status"] in {"approved", "rejected"} and not (report.get("review") or {}).get("reviewer"):
                warnings.append(f"Unattributed review requires re-review: {path.name}")
                report = {**report, "status": "review_required"}
            reports[document_id] = report
        except (OSError, ValueError, TypeError):
            warnings.append(f"Unreadable report: {path.name}")
    return rows, reports, warnings


def import_legacy(store: DocumentManifest, sqlite_path: Path, reports_dir: Path) -> tuple[int, int, list[str]]:
    """Import into an empty PostgreSQL manifest in one transaction."""
    rows, reports, warnings = preview_legacy(sqlite_path, reports_dir)
    with store._connect() as connection:
        connection.execute("SELECT pg_advisory_xact_lock(474159166216016)")
        if connection.execute("SELECT count(*) AS n FROM document_manifest").fetchone()["n"]:
            raise ValueError("PostgreSQL manifest is not empty; import into a fresh database")
        for row in rows:
            report = reports.get(row["revision_id"])
            workflow_status = report["status"] if report and row["is_current"] else (
                "superseded" if not row["is_current"] and row["status"] == "superseded" else "pending"
            )
            connection.execute(
                """INSERT INTO document_manifest (
                   revision_id, source_id, source_key, source_url, document_hash, title,
                   status, is_current, supersedes_revision_id, expected_chunks,
                   indexed_chunks, attempts, error, workflow_status, report, review,
                   created_at, updated_at
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (row["revision_id"], row["source_id"], row["source_key"], row["source_url"],
                 row["document_hash"], row["title"], row["status"], bool(row["is_current"]),
                 row["supersedes_revision_id"], row["expected_chunks"],
                 row["indexed_chunks"] if row["status"] == "stored" else 0,
                 row["attempts"], row["error"], workflow_status,
                 json.dumps(report) if report else None,
                 json.dumps(report.get("review")) if report and report.get("review") else None,
                 row["created_at"], row["updated_at"]),
            )
    return len(rows), len(reports), warnings
