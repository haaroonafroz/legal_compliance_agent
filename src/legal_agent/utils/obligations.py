"""Obligation models and conservative parsing helpers for obligation-centric RAG."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class LegalObligation(BaseModel):
    """One atomic requirement extracted from a regulatory document."""

    obligation_id: str
    quote: str
    actor: str = ""
    action: str
    object: str = ""
    conditions: str = ""
    deadline: str = ""
    responsible_departments: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ObligationEvidence(BaseModel):
    """Policy evidence and provenance collected for one obligation."""

    obligation_id: str
    obligation: LegalObligation
    matched_policies: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    reranker_used: bool = False
    sufficient: bool = False
    insufficiency_reason: str = ""


_OBLIGATION_RE = re.compile(
    r"(?P<sentence>[^.!?\n]{0,500}\b(?:shall|must|required to|prohibited from|"
    r"may not|is obliged to|are obliged to)\b[^.!?\n]*[.!?]?)",
    re.IGNORECASE,
)


def _clean_json(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _fallback_obligations(text: str) -> list[LegalObligation]:
    """Extract obvious modal-verb sentences when structured output is unavailable."""
    matches = [match.group("sentence").strip() for match in _OBLIGATION_RE.finditer(text)]
    if not matches:
        excerpt = " ".join(text.split())[:1000]
        matches = [excerpt] if excerpt else []
    return [
        LegalObligation(
            obligation_id=f"obligation-{index}",
            quote=quote,
            action="Comply with the stated requirement",
            confidence=0.25,
        )
        for index, quote in enumerate(dict.fromkeys(matches), start=1)
    ]


def parse_obligations(
    content: str | None,
    source_text: str,
    *,
    maximum: int = 30,
) -> list[LegalObligation]:
    """Parse model output and retain only obligations with verifiable source quotes."""
    if not content:
        return _fallback_obligations(source_text)[:maximum]

    try:
        payload = json.loads(_clean_json(content))
        raw_items = payload.get("obligations", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list):
            return _fallback_obligations(source_text)[:maximum]
        obligations: list[LegalObligation] = []
        for index, raw_item in enumerate(raw_items[:maximum], start=1):
            if not isinstance(raw_item, dict):
                continue
            raw_item = dict(raw_item)
            raw_item.setdefault("obligation_id", f"obligation-{index}")
            raw_item.setdefault("action", "Comply with the stated requirement")
            obligation = LegalObligation.model_validate(raw_item)
            if obligation.quote and obligation.quote in source_text:
                obligations.append(obligation)
        if obligations:
            return obligations
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    return _fallback_obligations(source_text)[:maximum]


def obligation_query(obligation: LegalObligation) -> str:
    """Build a focused retrieval query from structured obligation components."""
    parts = [
        obligation.actor,
        obligation.action,
        obligation.object,
        obligation.conditions,
        obligation.deadline,
        " ".join(obligation.topic_tags),
        obligation.quote,
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())
