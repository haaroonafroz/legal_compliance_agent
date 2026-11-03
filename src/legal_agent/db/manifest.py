"""PostgreSQL is the authoritative store for document and workflow state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row


class DocumentManifest:
    """Transactional document registry shared by scraper and workflow workers."""

    def __init__(self, database_url: str | None) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required; run PostgreSQL and legal-agent migrate-db")
        self.database_url = database_url

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=10)

    def migrate(self) -> None:
        sql = (Path(__file__).parent / "migrations" / "001_initial.sql").read_text()
        with self._connect() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(474159166216015)")
            connection.execute(sql)
            connection.execute(
                "INSERT INTO schema_migrations(version) VALUES (%s) ON CONFLICT DO NOTHING",
                ("001_initial",),
            )

    def prepare(self, *, source_id: str, source_key: str, source_url: str,
                document_hash: str, revision_id: str, title: str = "") -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (source_key,))
            current = connection.execute(
                "SELECT revision_id FROM document_manifest WHERE source_key = %s AND is_current",
                (source_key,),
            ).fetchone()
            if current and current["revision_id"] == revision_id:
                return {"action": "unchanged", "revision_id": revision_id}
            supersedes = current["revision_id"] if current else ""
            existing = connection.execute(
                "SELECT status FROM document_manifest WHERE revision_id = %s FOR UPDATE",
                (revision_id,),
            ).fetchone()
            if existing and existing["status"] == "stored":
                return {"action": "unchanged", "revision_id": revision_id}
            connection.execute(
                """INSERT INTO document_manifest
                   (revision_id, source_id, source_key, source_url, document_hash,
                    title, status, supersedes_revision_id)
                   VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s)
                   ON CONFLICT (revision_id) DO UPDATE SET status = 'pending',
                   error = '', attempts = document_manifest.attempts + 1,
                   supersedes_revision_id = EXCLUDED.supersedes_revision_id,
                   updated_at = now()""",
                (revision_id, source_id, source_key, source_url, document_hash, title, supersedes),
            )
            return {"action": "ingest", "revision_id": revision_id,
                    "supersedes_revision_id": supersedes}

    def set_expected_chunks(self, revision_id: str, count: int) -> None:
        if count <= 0:
            raise ValueError("A revision must have at least one chunk")
        with self._connect() as connection:
            row = connection.execute(
                """UPDATE document_manifest SET expected_chunks = %s, updated_at = now()
                   WHERE revision_id = %s AND status = 'pending' RETURNING revision_id""",
                (count, revision_id),
            ).fetchone()
            if not row:
                raise ValueError("Revision is not pending")

    def record_chunks(self, revision_id: str, indices: list[int]) -> bool:
        """Record successful upserts once, including worker retries."""
        if not indices or any(index < 0 for index in indices):
            raise ValueError("Chunk indices must be nonempty and nonnegative")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT expected_chunks, status FROM document_manifest WHERE revision_id = %s FOR UPDATE",
                (revision_id,),
            ).fetchone()
            if not row or row["status"] != "pending":
                raise ValueError("Revision is not pending")
            if any(index >= row["expected_chunks"] for index in indices):
                raise ValueError("Chunk index exceeds expected count")
            with connection.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO indexed_chunks(revision_id, chunk_index) VALUES (%s, %s)
                       ON CONFLICT DO NOTHING""",
                    [(revision_id, index) for index in indices],
                )
            count = connection.execute(
                "SELECT count(*) AS n FROM indexed_chunks WHERE revision_id = %s", (revision_id,)
            ).fetchone()["n"]
            connection.execute(
                "UPDATE document_manifest SET indexed_chunks = %s, updated_at = now() WHERE revision_id = %s",
                (count, revision_id),
            )
            return count == row["expected_chunks"]

    def mark_stored(self, revision_id: str) -> str:
        """Promote only a complete revision; stale workers cannot supersede a newer revision."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT source_key FROM document_manifest WHERE revision_id = %s", (revision_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown revision: {revision_id}")
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                               (row["source_key"],))
            pending = connection.execute(
                """SELECT revision_id, expected_chunks, indexed_chunks, status
                   FROM document_manifest WHERE source_key = %s
                   ORDER BY created_at DESC, revision_id DESC LIMIT 1""",
                (row["source_key"],),
            ).fetchone()
            if pending["revision_id"] != revision_id:
                raise ValueError("A newer revision has already been registered")
            if pending["status"] != "pending" or pending["expected_chunks"] == 0 or (
                pending["indexed_chunks"] != pending["expected_chunks"]
            ):
                raise ValueError("Cannot promote an incomplete or failed revision")
            current = connection.execute(
                "SELECT revision_id FROM document_manifest WHERE source_key = %s AND is_current",
                (row["source_key"],),
            ).fetchone()
            superseded = current["revision_id"] if current else ""
            connection.execute(
                """UPDATE document_manifest SET is_current = false, status = 'superseded',
                   workflow_status = 'superseded', claim_token = NULL, claim_until = NULL,
                   updated_at = now() WHERE source_key = %s AND is_current""",
                (row["source_key"],),
            )
            connection.execute(
                """UPDATE document_manifest SET is_current = true, status = 'stored',
                   updated_at = now() WHERE revision_id = %s""",
                (revision_id,),
            )
            return superseded

    def mark_failed(self, revision_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE document_manifest SET status = 'dead_letter', error = %s,
                   updated_at = now() WHERE revision_id = %s AND status = 'pending'""",
                (error[:2000], revision_id),
            )

    def retryable(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return connection.execute(
                """SELECT * FROM document_manifest WHERE status = 'dead_letter'
                   ORDER BY updated_at LIMIT %s""", (limit,)
            ).fetchall()

    def reset_for_retry(self, revision_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """UPDATE document_manifest SET status = 'pending', error = '',
                   attempts = attempts + 1, updated_at = now()
                   WHERE revision_id = %s AND status = 'dead_letter' RETURNING revision_id""",
                (revision_id,),
            ).fetchone()
            return bool(row)

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, count(*) AS count FROM document_manifest GROUP BY status"
            ).fetchall()
            return {row["status"]: row["count"] for row in rows}

    def workflow_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT workflow_status, count(*) AS count FROM document_manifest
                   WHERE is_current GROUP BY workflow_status"""
            ).fetchall()
            return {row["workflow_status"]: row["count"] for row in rows}

    def claim_next(self, lease_seconds: int = 1800,
                   revision_id: str | None = None) -> dict[str, Any] | None:
        """Claim a current pending document with a bounded DB lease."""
        token = uuid4()
        with self._connect() as connection:
            return connection.execute(
                """WITH candidate AS (
                     SELECT revision_id FROM document_manifest
                     WHERE is_current AND status = 'stored' AND workflow_status = 'pending'
                       AND (claim_until IS NULL OR claim_until < now())
                       AND (%s::text IS NULL OR revision_id = %s)
                     ORDER BY updated_at, revision_id FOR UPDATE SKIP LOCKED LIMIT 1
                   )
                   UPDATE document_manifest d SET claim_token = %s,
                     claim_until = now() + (%s * interval '1 second'), updated_at = now()
                   FROM candidate WHERE d.revision_id = candidate.revision_id
                   RETURNING d.revision_id, d.document_hash, d.claim_token, d.claim_until""",
                (revision_id, revision_id, token, lease_seconds),
            ).fetchone()

    def release_claim(self, revision_id: str, token: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE document_manifest SET claim_token = NULL, claim_until = NULL,
                   updated_at = now() WHERE revision_id = %s AND claim_token = %s
                   AND workflow_status = 'pending'""", (revision_id, token)
            )

    def complete_workflow(self, revision_id: str, document_hash: str, token: Any,
                          report: dict[str, Any]) -> None:
        status = report.get("status")
        if status not in {"irrelevant", "review_required", "pending_human_review"}:
            raise ValueError("Invalid automated workflow status")
        if report.get("document_id") != revision_id or report.get("document_hash") != document_hash:
            raise ValueError("Report identity does not match claim")
        with self._connect() as connection:
            row = connection.execute(
                """UPDATE document_manifest SET workflow_status = %s, report = %s,
                   claim_token = NULL, claim_until = NULL, updated_at = now()
                   WHERE revision_id = %s AND document_hash = %s AND claim_token = %s
                   AND claim_until > now() AND status = 'stored' AND is_current
                   AND workflow_status = 'pending' RETURNING revision_id""",
                (status, json.dumps(report), revision_id, document_hash, token),
            ).fetchone()
            if not row:
                raise ValueError("Claim expired, document changed, or report already completed")

    def get_report(self, revision_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report FROM document_manifest WHERE revision_id = %s", (revision_id,)
            ).fetchone()
            if not row or row["report"] is None:
                raise KeyError(f"No durable report for {revision_id}")
            return row["report"]

    def review_report(self, revision_id: str, expected_hash: str, expected_status: str,
                      decision: str, reviewer: str, reason: str) -> dict[str, Any]:
        from datetime import UTC, datetime

        if decision not in {"approve", "reject"} or not reviewer.strip() or not reason.strip():
            raise ValueError("Decision, reviewer, and reason are required")
        status = "approved" if decision == "approve" else "rejected"
        with self._connect() as connection:
            row = connection.execute(
                """SELECT report FROM document_manifest WHERE revision_id = %s
                   AND document_hash = %s AND is_current AND status = 'stored'
                   AND workflow_status = %s FOR UPDATE""",
                (revision_id, expected_hash, expected_status),
            ).fetchone()
            if not row or row["report"] is None:
                raise ValueError("Report is stale or document is no longer current")
            if expected_status not in {"pending_human_review", "review_required"} or (
                decision == "approve" and expected_status != "pending_human_review"
            ):
                raise ValueError("Only a passing report can be approved")
            report = row["report"]
            review = {"decision": decision, "reviewer": reviewer.strip(),
                      "reason": reason.strip(), "reviewed_at": datetime.now(UTC).isoformat()}
            report.update({"status": status, "review": review})
            connection.execute(
                """UPDATE document_manifest SET workflow_status = %s, report = %s,
                   review = %s, updated_at = now() WHERE revision_id = %s""",
                (status, json.dumps(report), json.dumps(review), revision_id),
            )
            return report
