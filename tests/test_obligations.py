"""Tests for obligation extraction and retrieval query preparation."""

from legal_agent.utils.obligations import (
    LegalObligation,
    obligation_query,
    parse_obligations,
)


def test_parse_obligations_keeps_only_verifiable_quotes():
    source = "The controller shall notify the authority within 72 hours."
    content = (
        '{"obligations": ['
        '{"obligation_id": "o1", '
        '"quote": "The controller shall notify the authority within 72 hours.", '
        '"action": "Notify the authority", "deadline": "72 hours", "confidence": 0.95},'
        '{"obligation_id": "o2", "quote": "This quote is not in the source", '
        '"action": "Ignore", "confidence": 0.99}'
        "]}"
    )

    obligations = parse_obligations(content, source)

    assert len(obligations) == 1
    assert obligations[0].obligation_id == "o1"
    assert obligations[0].quote == source


def test_parse_obligations_falls_back_to_modal_sentences():
    source = (
        "The controller shall notify the authority within 72 hours. "
        "The processor must maintain records."
    )

    obligations = parse_obligations(None, source)

    assert len(obligations) == 2
    assert all(obligation.confidence == 0.25 for obligation in obligations)


def test_obligation_query_contains_structured_fields():
    obligation = LegalObligation(
        obligation_id="o1",
        quote="The controller shall notify the authority.",
        actor="controller",
        action="notify",
        object="authority",
        deadline="72 hours",
        topic_tags=["breach_notification"],
    )

    query = obligation_query(obligation)

    assert "controller" in query
    assert "72 hours" in query
    assert "breach_notification" in query
