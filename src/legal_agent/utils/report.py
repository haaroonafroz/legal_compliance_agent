"""Write final compliance reports to disk."""

from __future__ import annotations

import json
import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")


def save_report(result: dict[str, Any]) -> Path:
    """Persist a workflow result dict as a timestamped JSON + Markdown pair."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    document_id = str(result.get("document_id", ""))
    if not document_id:
        raise ValueError("Cannot save a report without document_id")
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", document_id)[:64]
    base = f"{safe_id}_{sha256(document_id.encode()).hexdigest()[:12]}"

    json_path = REPORTS_DIR / f"{base}.json"
    write_report(json_path, result)
    md_path = json_path.with_suffix(".md")

    logger.info("Report saved: %s", md_path)
    return md_path


def load_report(path: Path) -> dict[str, Any]:
    """Load the JSON paired with a JSON or Markdown report path."""
    json_path = path.with_suffix(".json")
    result = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not result.get("document_id"):
        raise ValueError("Report has no document ID")
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    """Write both report formats before the workflow changes durable state."""
    json_path = path.with_suffix(".json")
    md_path = path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_suffix(".json.tmp")
    md_tmp = md_path.with_suffix(".md.tmp")
    json_tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md_tmp.write_text(_render_markdown(result), encoding="utf-8")
    json_tmp.replace(json_path)
    md_tmp.replace(md_path)


def _render_markdown(result: dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    review = result.get("review") or {}
    review_lines = (
        f"- Human decision: {review.get('decision', 'pending')}\n"
        f"- Reviewer: {review.get('reviewer', 'N/A')}\n"
        f"- Reviewed at: {review.get('reviewed_at', 'N/A')}\n"
        f"- Review rationale: {review.get('reason', 'N/A')}"
    )
    passed = "NOT RUN" if result.get("skipped") else ("PASS" if result.get("passed") else "FAIL")
    retrieved = result.get("matched_policies", [])
    scores = result.get("retrieval_scores", [])
    obligation_evidence = result.get("obligation_evidence", [])
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
    obligation_lines = (
        "\n\n".join(
            (
                f"### {bundle.get('obligation_id', 'unknown')} "
                f"({'sufficient' if bundle.get('sufficient') else 'insufficient'})\n"
                f"Quote: {bundle.get('obligation', {}).get('quote', 'N/A')}\n"
                "Evidence chunks: "
                + ", ".join(
                    str(policy.get("chunk_id") or policy.get("document_id") or "unknown")
                    for policy in bundle.get("matched_policies", [])
                )
            )
            for bundle in obligation_evidence
        )
        or "_No obligation evidence recorded._"
    )
    assessment_lines = (
        "\n".join(
            f"- `{item.get('obligation_id', 'unknown')}`: **{item.get('outcome', 'unknown')}** — "
            f"{item.get('explanation', '')}"
            for item in result.get("assessments", [])
        )
        or "_No structured assessments recorded._"
    )

    return f"""\
# Policy Coverage Review

| Field | Value |
|---|---|
| Workflow Status | **{status.upper()}** |
| Reason | {result.get("reason", "N/A")} |
| Jurisdiction | {result.get("jurisdiction", "N/A")} |
| Source URL | {result.get("source_url", "N/A")} |
| Source Key | {result.get("source_key", "N/A")} |
| Revision ID | {result.get("revision_id", "N/A")} |
| Document Type | {result.get("document_type", "N/A")} |
| Model Audit Result | **{passed}** |
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

---

## Obligation Evidence

{obligation_lines}

## Structured Assessments

{assessment_lines}

---

## Human Review

{review_lines}
"""
