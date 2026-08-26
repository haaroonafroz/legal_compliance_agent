"""Durable document-level ingestion state and dead-letter tracking."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DocumentManifest:
    """Small SQLite manifest for idempotent revisions and ingestion failures."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_manifest (
                    revision_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    is_current INTEGER NOT NULL DEFAULT 0,
                    supersedes_revision_id TEXT NOT NULL DEFAULT '',
                    expected_chunks INTEGER NOT NULL DEFAULT 0,
                    indexed_chunks INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_manifest_source_current
                    ON document_manifest(source_key, is_current);
                CREATE INDEX IF NOT EXISTS idx_manifest_status
                    ON document_manifest(status);
                """
            )

    def prepare(
        self,
        *,
        source_id: str,
        source_key: str,
        source_url: str,
        document_hash: str,
        revision_id: str,
        title: str = "",
    ) -> dict[str, Any]:
        """Register a revision, returning whether it should be ingested."""
        now = _now()
        with self._connect() as connection:
            current = connection.execute(
                """
                SELECT revision_id, status
                FROM document_manifest
                WHERE source_key = ? AND is_current = 1
                """,
                (source_key,),
            ).fetchone()
            if current and current["revision_id"] == revision_id:
                return {"action": "unchanged", "revision_id": revision_id}

            supersedes = current["revision_id"] if current else ""
            connection.execute(
                """
                INSERT INTO document_manifest (
                    revision_id, source_id, source_key, source_url, document_hash,
                    title, status, supersedes_revision_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(revision_id) DO UPDATE SET
                    status = 'pending',
                    error = '',
                    attempts = attempts + 1,
                    supersedes_revision_id = excluded.supersedes_revision_id,
                    updated_at = excluded.updated_at
                """,
                (
                    revision_id,
                    source_id,
                    source_key,
                    source_url,
                    document_hash,
                    title,
                    supersedes,
                    now,
                    now,
                ),
            )
            return {
                "action": "ingest",
                "revision_id": revision_id,
                "supersedes_revision_id": supersedes,
            }

    def set_expected_chunks(self, revision_id: str, count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_manifest
                SET expected_chunks = ?, updated_at = ?
                WHERE revision_id = ?
                """,
                (count, _now(), revision_id),
            )

    def record_chunks(self, revision_id: str, count: int) -> bool:
        """Record indexed chunks and return whether the revision is complete."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_manifest
                SET indexed_chunks = indexed_chunks + ?, updated_at = ?
                WHERE revision_id = ?
                """,
                (count, _now(), revision_id),
            )
            row = connection.execute(
                """
                SELECT indexed_chunks, expected_chunks
                FROM document_manifest
                WHERE revision_id = ?
                """,
                (revision_id,),
            ).fetchone()
            return bool(row and row["expected_chunks"] and row["indexed_chunks"] >= row["expected_chunks"])

    def mark_stored(self, revision_id: str) -> str:
        """Make a revision current and supersede its predecessor atomically."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_key, supersedes_revision_id
                FROM document_manifest
                WHERE revision_id = ?
                """,
                (revision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown revision: {revision_id}")
            connection.execute(
                """
                UPDATE document_manifest
                SET is_current = 0, status = 'superseded', updated_at = ?
                WHERE source_key = ? AND revision_id != ? AND is_current = 1
                """,
                (_now(), row["source_key"], revision_id),
            )
            connection.execute(
                """
                UPDATE document_manifest
                SET is_current = 1, status = 'stored', updated_at = ?
                WHERE revision_id = ?
                """,
                (_now(), revision_id),
            )
            return row["supersedes_revision_id"]

    def mark_failed(self, revision_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_manifest
                SET status = 'dead_letter', error = ?, updated_at = ?
                WHERE revision_id = ?
                """,
                (error[:2000], _now(), revision_id),
            )

    def retryable(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM document_manifest
                WHERE status = 'dead_letter'
                ORDER BY updated_at
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def reset_for_retry(self, revision_id: str) -> None:
        """Move one dead-lettered revision back to pending for the next scrape."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_manifest
                SET status = 'pending', error = '', attempts = attempts + 1, updated_at = ?
                WHERE revision_id = ? AND status = 'dead_letter'
                """,
                (_now(), revision_id),
            )

    def counts(self) -> dict[str, int]:
        """Return document counts grouped by lifecycle status."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM document_manifest GROUP BY status"
            ).fetchall()
            return {row["status"]: row["count"] for row in rows}
