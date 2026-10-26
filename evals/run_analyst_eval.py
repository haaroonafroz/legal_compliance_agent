"""Run a small labeled analyst evaluation with the configured LLM provider."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from legal_agent.config import get_settings
from legal_agent.utils.assessments import parse_assessments
from legal_agent.workflow.llm_provider import get_llm_for_step
from legal_agent.workflow.prompts import ANALYST_SYSTEM, ANALYST_USER
from legal_agent.workflow.workflow import _format_obligation_evidence, _format_policy_evidence


async def main() -> int:
    cases = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    llm = get_llm_for_step(get_settings(), "analyst")
    correct = 0
    for case in cases:
        bundle = {
            "obligation_id": case["id"],
            "obligation": {"quote": case["regulation_quote"], "action": "Assess the duty"},
            "matched_policies": [
                {"policy_id": "synthetic-policy", "chunk_id": case["id"], "text": case["policy_text"]}
            ],
            "sufficient": True,
        }
        prompt = ANALYST_USER.format(
            jurisdiction="Synthetic",
            source_url=f"eval:{case['id']}",
            effective_date="",
            regulation_text=case["regulation_quote"],
            policies_text=_format_policy_evidence(bundle["matched_policies"]),
            obligation_evidence=_format_obligation_evidence([bundle]),
        )
        try:
            response = await llm.achat(
                messages=[
                    ChatMessage(role=MessageRole.SYSTEM, content=ANALYST_SYSTEM),
                    ChatMessage(role=MessageRole.USER, content=prompt),
                ]
            )
            assessment = parse_assessments(response.message.content, [bundle])[0]
            actual = assessment.outcome
        except ValueError as exc:
            actual = f"invalid: {exc}"
        correct += actual == case["expected_outcome"]
        print(f"{case['id']}: expected={case['expected_outcome']} actual={actual}")
    print(f"Outcome accuracy: {correct}/{len(cases)}")
    return 0 if correct == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
