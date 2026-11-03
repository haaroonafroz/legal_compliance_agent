"""Opt-in live LLM evaluation of the fictional policy-coverage cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from legal_agent.config import get_settings
from legal_agent.utils.assessments import parse_assessments
from legal_agent.workflow.llm_provider import get_llm_for_step
from legal_agent.workflow.prompts import ANALYST_SYSTEM, ANALYST_USER
from legal_agent.workflow.workflow import _format_obligation_evidence, _format_policy_evidence

CASES = Path(__file__).with_name("synthetic_cases.json")


async def run(case_id: str | None = None) -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    if case_id:
        cases = [case for case in cases if case["id"] == case_id]
        if not cases:
            raise ValueError(f"Unknown case ID: {case_id}")
    llm = get_llm_for_step(get_settings(), "analyst")
    results = []
    for case in cases:
        policies = []
        if case["policy_text"]:
            policies = [{
                "policy_id": f"fictional-{case['id']}",
                "chunk_id": f"fictional-{case['id']}:0",
                "text": case["policy_text"],
            }]
        bundle = {
            "obligation_id": case["id"],
            "obligation": {"quote": case["regulation_quote"], "action": "Assess the duty"},
            "matched_policies": policies,
            "sufficient": bool(policies),
        }
        prompt = ANALYST_USER.format(
            jurisdiction="Fictional",
            source_url=case["source"],
            effective_date="",
            regulation_text=case["regulation_quote"],
            policies_text=_format_policy_evidence(policies),
            obligation_evidence=_format_obligation_evidence([bundle]),
        )
        start = time.monotonic()
        try:
            response = await llm.achat(messages=[
                ChatMessage(role=MessageRole.SYSTEM, content=ANALYST_SYSTEM),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ])
            assessment = parse_assessments(response.message.content, [bundle])[0]
            actual = assessment.outcome
            error = None
        except Exception as exc:  # noqa: BLE001 - provider failures are measured
            actual = "invalid"
            error = f"{type(exc).__name__}: {exc}"
        results.append({
            "id": case["id"],
            "scenario": case["scenario"],
            "expected": case["expected_outcome"],
            "actual": actual,
            "correct": actual == case["expected_outcome"],
            "latency_seconds": round(time.monotonic() - start, 3),
            "error": error,
        })
    counts = Counter((item["expected"], item["actual"]) for item in results)
    return {
        "notice": "FICTIONAL TEST DATA ONLY. Not legal advice or actual GDPR/EDPB text.",
        "total": len(results),
        "accuracy": sum(item["correct"] for item in results) / len(results),
        "invalid_rate": sum(item["actual"] == "invalid" for item in results) / len(results),
        "mean_latency_seconds": round(sum(item["latency_seconds"] for item in results) / len(results), 3),
        "confusion": {f"{expected}->{actual}": count for (expected, actual), count in counts.items()},
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Run one case ID")
    parser.add_argument("--output", type=Path, default=Path("evals/results.json"))
    args = parser.parse_args()
    metrics = asyncio.run(run(args.case))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in (
        "total", "accuracy", "invalid_rate", "mean_latency_seconds", "confusion"
    )}, indent=2))
    print(f"Detailed results: {args.output}")
    return 0 if metrics["accuracy"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
