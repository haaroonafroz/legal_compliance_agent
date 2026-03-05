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
    base = f"{ts}_{slug}"

    json_path = REPORTS_DIR / f"{base}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    md_path = REPORTS_DIR / f"{base}.md"
    md_path.write_text(_render_markdown(result), encoding="utf-8")

    logger.info("Report saved: %s", md_path)
    return md_path


def _render_markdown(result: dict[str, Any]) -> str:
    passed = "PASS" if result.get("passed") else "FAIL"
    return f"""\
# Compliance Report

| Field | Value |
|---|---|
| Jurisdiction | {result.get('jurisdiction', 'N/A')} |
| Source URL | {result.get('source_url', 'N/A')} |
| Audit Result | **{passed}** |

---

## Gap Analysis

{result.get('gap_analysis', '_No analysis generated._')}

---

## Proposed Policy Updates

{result.get('proposed_updates', '_No updates proposed._')}

---

## Auditor Notes

{result.get('audit_notes', '_No notes._')}
"""
