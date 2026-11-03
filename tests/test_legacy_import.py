"""Non-destructive migration from a legacy SQLite manifest and JSON report."""

import json
import os
import sqlite3
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from legal_agent.db.legacy import import_legacy, preview_legacy
from legal_agent.db.manifest import DocumentManifest


def test_legacy_preview_and_import(tmp_path):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to an isolated PostgreSQL test database")
    schema = f"legacy_test_{uuid4().hex}"
    sqlite_file = tmp_path / "manifest.sqlite3"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    revision = uuid4().hex
    with sqlite3.connect(sqlite_file) as connection:
        connection.execute("""CREATE TABLE document_manifest (
            revision_id TEXT, source_id TEXT, source_key TEXT, source_url TEXT,
            document_hash TEXT, title TEXT, status TEXT, is_current INTEGER,
            supersedes_revision_id TEXT, expected_chunks INTEGER, indexed_chunks INTEGER,
            attempts INTEGER, error TEXT, created_at TEXT, updated_at TEXT)""")
        connection.execute(
            "INSERT INTO document_manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (revision, "fictional", "fictional::one", "https://fiction.invalid/one", "hash-1",
             "Fictional notice", "stored", 1, "", 1, 1, 0, "", "2026-01-01T00:00:00Z",
             "2026-01-01T00:00:00Z"),
        )
    report = {
        "document_id": revision, "document_hash": "hash-1",
        "status": "pending_human_review", "passed": True,
    }
    (reports_dir / "report.json").write_text(json.dumps(report))
    rows, reports, warnings = preview_legacy(sqlite_file, reports_dir)
    assert len(rows) == 1 and reports[revision] == report and warnings == []
    with psycopg.connect(url) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    sep = "&" if "?" in url else "?"
    scoped_url = f"{url}{sep}options=-csearch_path%3D{schema}"
    try:
        store = DocumentManifest(scoped_url)
        store.migrate()
        assert import_legacy(store, sqlite_file, reports_dir)[:2] == (1, 1)
        assert store.get_report(revision) == report
        with pytest.raises(ValueError, match="not empty"):
            import_legacy(store, sqlite_file, reports_dir)
        assert sqlite_file.exists() and (reports_dir / "report.json").exists()
    finally:
        with psycopg.connect(url) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
