"""Tests for canonical document and revision identities."""

from legal_agent.utils.identity import canonicalize_url, content_hash, revision_id, source_key


def test_canonicalize_url_removes_tracking_and_fragment():
    assert (
        canonicalize_url(
            "HTTPS://EDPB.EUROPA.EU/guide/?utm_source=mail&b=2&a=1#section"
        )
        == "https://edpb.europa.eu/guide?a=1&b=2"
    )


def test_source_identity_is_stable_for_equivalent_urls():
    first = source_key("edpb", "https://edpb.europa.eu/guide/?utm_campaign=x")
    second = source_key("edpb", "https://EDPB.EUROPA.EU/guide")

    assert first == second


def test_revision_changes_with_content_but_ignores_whitespace():
    source = "edpb::guide"

    assert content_hash("Article 1\n requires controls.") == content_hash(
        "Article 1 requires  controls."
    )
    assert revision_id(source, "version one") != revision_id(source, "version two")
