"""Write final compliance reports to disk."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")


def save_report(result: dict[str, Any]) -> Path:
    """Persist a workflow result dict as a timestamped JSON + Markdown pair."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = (result.get("source_url", "unknown").split("/")[-1] or "report")[:40]
    document_suffix = str(result.get("document_id", "unknown"))[:12]
    base = f"{ts}_{slug}_{document_suffix}"

    json_path = REPORTS_DIR / f"{base}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    md_path = REPORTS_DIR / f"{base}.md"
    md_path.write_text(_render_markdown(result), encoding="utf-8")

    logger.info("Report saved: %s", md_path)
    return md_path


def _render_markdown(result: dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    if result.get("skipped"):
        return f"""\
# Compliance Workflow Report

- Status: **{status.upper()}**
- Jurisdiction: {result.get("jurisdiction", "N/A")}
- Source URL: {result.get("source_url", "N/A")}
- Reason: {result.get("reason", "N/A")}
- Document ID: {result.get("document_id", "N/A")}
- Revision ID: {result.get("revision_id", "N/A")}
"""

    passed = "PASS" if result.get("passed") else "FAIL"
    retrieved = result.get("matched_policies", [])
    scores = result.get("retrieval_scores", [])
    retrieval_lines = (
        "\n".join(
            (
                f"- `{policy.get('policy_id') or policy.get('document_id') or 'unknown'}` "
                f"/ `{policy.get('chunk_id', 'unknown')}` "
                f"({policy.get('header_path', 'section unknown')}) — score "
                f"`{scores[index]:.4f}`"
            )
            for index, policy in enumerate(retrieved)
            if index < len(scores)
        )
        or "_No retrieval provenance recorded._"
    )

    return f"""\
# Compliance Report

| Field | Value |
|---|---|
| Workflow Status | **{status.upper()}** |
| Jurisdiction | {result.get("jurisdiction", "N/A")} |
| Source URL | {result.get("source_url", "N/A")} |
| Source Key | {result.get("source_key", "N/A")} |
| Revision ID | {result.get("revision_id", "N/A")} |
| Document Type | {result.get("document_type", "N/A")} |
| Audit Result | **{passed}** |
| Audit Retries | {result.get("audit_retries", 0)} |
| Reranker Used | {result.get("reranker_used", False)} |

---

## Gap Analysis

{result.get("gap_analysis", "_No analysis generated._")}

---

## Proposed Policy Updates

{result.get("proposed_updates", "_No updates proposed._")}

---

## Auditor Notes

{result.get("audit_notes", "_No notes._")}

---

## Retrieved Policy Evidence

{retrieval_lines}
"""
