"""Cheap content gates and a borderline-only relevance classifier."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from legal_agent.utils.identity import canonicalize_url
from legal_agent.workflow.llm_provider import local_chat_template_kwargs

_LEGAL_SIGNAL_RE = re.compile(
    r"\b("
    r"guidelines?|recommendations?|opinions?|decisions?|regulation|directive|"
    r"article|personal data|data protection|controller|processor|shall|obligation"
    r")\b",
    re.IGNORECASE,
)
_NAVIGATION_WORDS = {
    "about",
    "contact",
    "home",
    "login",
    "menu",
    "search",
    "share",
    "subscribe",
}


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of deterministic or model-assisted content validation."""

    accepted: bool
    borderline: bool = False
    reason: str = ""
    confidence: float | None = None


def validate_deterministic(
    *,
    title: str,
    text: str,
    document_type: str = "",
    min_chars: int = 500,
) -> ValidationResult:
    """Reject obvious garbage and identify content needing a second-stage check."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ValidationResult(False, reason="empty extracted text")
    if len(cleaned) < min_chars:
        return ValidationResult(False, reason=f"extracted text is shorter than {min_chars} characters")

    words = cleaned.lower().split()
    if len(words) >= 40:
        navigation_ratio = sum(word in _NAVIGATION_WORDS for word in words) / len(words)
        if navigation_ratio > 0.15:
            return ValidationResult(False, reason="content is dominated by navigation terms")

    if document_type or _LEGAL_SIGNAL_RE.search(f"{title} {cleaned}"):
        return ValidationResult(True, reason="deterministic legal-content signal")
    return ValidationResult(
        accepted=False,
        borderline=True,
        reason="no source or legal-content signal",
    )


def classify_borderline(
    *,
    title: str,
    text: str,
    settings: Any,
) -> ValidationResult:
    """Use a small configured model only after deterministic validation is borderline."""
    if getattr(settings, "use_local_llm", False):
        client = OpenAI(
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key,
            timeout=settings.local_llm_timeout_seconds,
        )
        model = settings.local_llm_model
        extra_body = {"chat_template_kwargs": local_chat_template_kwargs("relevance_check")}
    else:
        client = OpenAI(api_key=settings.openai_api_key)
        model = getattr(settings, "content_validation_llm_model", None) or getattr(
            settings, "openai_llm_model_relevance_check", "gpt-4o-mini"
        )
        extra_body = None

    prompt = (
        "Classify whether this is a substantive legal or regulatory document. "
        "Return JSON only: {\"relevant\": true|false, \"confidence\": 0.0-1.0, "
        "\"reason\": \"short explanation\"}.\n\n"
        f"Title: {title}\n\nContent:\n{text[:8000]}"
    )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a conservative legal-content classifier."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    payload = json.loads(response.choices[0].message.content or "{}")
    confidence = float(payload.get("confidence", 0))
    threshold = float(getattr(settings, "content_validation_confidence_threshold", 0.8))
    relevant = bool(payload.get("relevant")) and confidence >= threshold
    return ValidationResult(
        accepted=relevant,
        borderline=False,
        reason=str(payload.get("reason", "LLM relevance decision")),
        confidence=confidence,
    )


def quarantine_record(item: dict[str, Any], result: ValidationResult) -> dict[str, Any]:
    """Create a minimal, non-content-heavy record for rejected documents."""
    return {
        "source_url": canonicalize_url(str(item.get("source_url", ""))),
        "source_id": item.get("source_id", ""),
        "title": item.get("title", ""),
        "reason": result.reason,
        "confidence": result.confidence,
        "retrieved_at": item.get("retrieved_at", ""),
    }
