"""Offline checks for fictional evaluation cases and source integrity."""

import json
from pathlib import Path

import pytest

from legal_agent.utils.assessments import parse_assessments
from legal_agent.utils.identity import revision_id, source_key

FIXTURE = Path(__file__).parents[1] / "evals" / "synthetic_cases.json"


def test_fixtures_cover_required_failure_modes():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert "FICTIONAL" in payload["notice"]
    cases = payload["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["source"].startswith("https://fiction.invalid/") for case in cases)
    assert {"covered", "gap", "uncertain"} <= {
        case["expected_outcome"] for case in cases
    }
    scenarios = {case["scenario"] for case in cases}
    assert any("injection" in scenario for scenario in scenarios)
    assert any("citation" in scenario for scenario in scenarios)
    assert any("missing" in scenario for scenario in scenarios)
    assert any("multiple obligations" in scenario for scenario in scenarios)
    old = next(case for case in cases if case["id"] == "source-revision-old")
    new = next(case for case in cases if case["id"] == "source-revision-new")
    assert old["source"] == new["source"]
    key = source_key("fictional-evaluation", old["source"])
    assert revision_id(key, old["regulation_quote"]) != revision_id(
        key, new["regulation_quote"]
    )


def test_fabricated_policy_citation_is_rejected():
    case = next(
        case for case in json.loads(FIXTURE.read_text())["cases"]
        if case["id"] == "fabricated-citation"
    )
    bundle = {
        "obligation_id": case["id"],
        "obligation": {"quote": case["regulation_quote"]},
        "matched_policies": [{"chunk_id": "real:0", "text": case["policy_text"]}],
    }
    response = {"assessments": [{
        "obligation_id": case["id"], "regulation_quote": case["regulation_quote"],
        "outcome": "covered", "explanation": "Claims coverage",
        "severity": "none",
        "policy_citations": [{"chunk_id": "real:0", "quote": "Invented policy clause"}],
    }]}
    with pytest.raises(ValueError, match="not present"):
        parse_assessments(json.dumps(response), [bundle])
