"""Contract tests for obligation-level decisions and provenance."""

import json

import pytest

from legal_agent.utils.assessments import parse_assessments


def _evidence():
    return [
        {
            "obligation_id": "obligation-1",
            "obligation": {"quote": "The controller shall notify within 72 hours."},
            "matched_policies": [
                {"chunk_id": "policy-1", "text": "The team notifies the authority within 72 hours."}
            ],
        },
        {
            "obligation_id": "obligation-2",
            "obligation": {"quote": "The controller shall keep records."},
            "matched_policies": [],
        },
    ]


def _assessment(obligation_id, quote, outcome, citations=None):
    return {
        "obligation_id": obligation_id,
        "regulation_quote": quote,
        "outcome": outcome,
        "explanation": "Policy coverage assessment",
        "severity": "none" if outcome == "covered" else "medium",
        "policy_citations": citations or [],
    }


def test_assessment_requires_complete_obligation_coverage():
    first = _assessment(
        "obligation-1",
        "The controller shall notify within 72 hours.",
        "covered",
        [{"chunk_id": "policy-1", "quote": "notifies the authority within 72 hours"}],
    )
    second = _assessment("obligation-2", "The controller shall keep records.", "gap")
    parsed = parse_assessments(json.dumps({"assessments": [first, second]}), _evidence())
    assert [item.outcome for item in parsed] == ["covered", "gap"]

    with pytest.raises(ValueError, match="exactly one"):
        parse_assessments(json.dumps({"assessments": [first]}), _evidence())


def test_assessment_rejects_unsupported_quotes_and_uncited_coverage():
    bundle = _evidence()[:1]
    covered = _assessment(
        "obligation-1", "The controller shall notify within 72 hours.", "covered"
    )
    with pytest.raises(ValueError, match="citation"):
        parse_assessments(json.dumps({"assessments": [covered]}), bundle)

    covered["policy_citations"] = [{"chunk_id": "policy-1", "quote": "a process not in policy"}]
    with pytest.raises(ValueError, match="not present"):
        parse_assessments(json.dumps({"assessments": [covered]}), bundle)

    covered["regulation_quote"] = "Fabricated regulatory quote"
    with pytest.raises(ValueError, match="regulation quote"):
        parse_assessments(json.dumps({"assessments": [covered]}), bundle)


def test_assessment_rejects_duplicate_obligation_ids():
    first = _assessment(
        "obligation-1",
        "The controller shall notify within 72 hours.",
        "covered",
        [{"chunk_id": "policy-1", "quote": "notifies the authority within 72 hours"}],
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_assessments(json.dumps({"assessments": [first, first]}), _evidence())
