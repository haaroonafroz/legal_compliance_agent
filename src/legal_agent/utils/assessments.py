"""Validate model assessments against the obligations and retrieved policy text."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field


class PolicyCitation(BaseModel):
    chunk_id: str
    quote: str


class ObligationAssessment(BaseModel):
    obligation_id: str
    regulation_quote: str
    outcome: Literal["covered", "gap", "uncertain"]
    explanation: str
    severity: Literal["high", "medium", "low", "none"]
    policy_citations: list[PolicyCitation] = Field(default_factory=list)


def parse_assessments(content: str | None, evidence_bundles: list[dict]) -> list[ObligationAssessment]:
    """Require one source-backed assessment for every extracted obligation."""
    if not content:
        raise ValueError("Analysis returned no content")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    payload = json.loads(cleaned)
    raw = payload.get("assessments") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        raise ValueError("Analysis must return an assessments list")
    expected = {bundle["obligation_id"]: bundle for bundle in evidence_bundles}
    if len(expected) != len(evidence_bundles) or len(raw) != len(expected):
        raise ValueError("Analysis must contain exactly one result per obligation")

    assessments: list[ObligationAssessment] = []
    seen: set[str] = set()
    for item in raw:
        assessment = ObligationAssessment.model_validate(item)
        bundle = expected.get(assessment.obligation_id)
        if bundle is None or assessment.obligation_id in seen:
            raise ValueError("Analysis has an unknown or duplicate obligation ID")
        seen.add(assessment.obligation_id)
        if assessment.regulation_quote != bundle["obligation"]["quote"]:
            raise ValueError("Analysis regulation quote does not match the extracted obligation")
        if not assessment.explanation.strip():
            raise ValueError("Analysis explanation is empty")
        if assessment.outcome == "covered" and not assessment.policy_citations:
            raise ValueError("Covered outcome requires a policy citation")
        if assessment.outcome == "gap" and assessment.severity == "none":
            raise ValueError("Gap outcome requires a severity")
        policies = {
            str(policy.get("chunk_id")): policy.get("text", "")
            for policy in bundle.get("matched_policies", [])
            if policy.get("chunk_id")
        }
        for citation in assessment.policy_citations:
            if citation.chunk_id not in policies or not citation.quote.strip():
                raise ValueError("Policy citation does not identify retrieved evidence")
            if citation.quote not in policies[citation.chunk_id]:
                raise ValueError("Policy quote is not present in the cited chunk")
        assessments.append(assessment)
    return assessments


def render_assessments(assessments: list[ObligationAssessment]) -> str:
    """Give reviewers a readable view of the validated, structured assessment."""
    lines = [
        "| Obligation | Regulation quote | Policy coverage | Policy citations | Explanation | Severity |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for assessment in assessments:
        citations = "; ".join(
            f"{citation.chunk_id}: {citation.quote}" for citation in assessment.policy_citations
        ) or "No policy citation"
        values = (
            assessment.obligation_id,
            assessment.regulation_quote,
            assessment.outcome,
            citations,
            assessment.explanation,
            assessment.severity,
        )
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    return "\n".join(lines)
