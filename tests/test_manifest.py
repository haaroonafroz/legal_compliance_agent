"""PostgreSQL integration tests; TEST_DATABASE_URL must name an isolated test DB."""

import os
from uuid import uuid4

import pytest

from legal_agent.db.manifest import DocumentManifest


@pytest.fixture
def manifest():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to an isolated PostgreSQL test database")
    store = DocumentManifest(url)
    store.migrate()
    return store


def _prepare(store, source, revision, text_hash):
    return store.prepare(
        source_id="fictional", source_key=source,
        source_url=f"https://fiction.invalid/{source}",
        document_hash=text_hash, revision_id=revision, title="Fictional test",
    )


def _stored(store, source, revision, text_hash):
    result = _prepare(store, source, revision, text_hash)
    store.set_expected_chunks(revision, 2)
    assert not store.record_chunks(revision, [0])
    assert not store.record_chunks(revision, [0])  # retried upsert must not overcount
    assert store.record_chunks(revision, [1])
    store.mark_stored(revision)
    return result


def test_manifest_idempotency_and_revision_order(manifest):
    source = str(uuid4())
    first, second = str(uuid4()), str(uuid4())
    _stored(manifest, source, first, "hash-1")
    assert _prepare(manifest, source, first, "hash-1")["action"] == "unchanged"
    result = _prepare(manifest, source, second, "hash-2")
    assert result["supersedes_revision_id"] == first
    manifest.set_expected_chunks(second, 2)
    with pytest.raises(ValueError, match="incomplete"):
        manifest.mark_stored(second)
    manifest.record_chunks(second, [0, 1])
    assert manifest.mark_stored(second) == first
    with pytest.raises(ValueError):
        manifest.mark_stored(first)


def test_claim_is_exclusive_and_report_review_is_durable(manifest):
    revision = str(uuid4())
    _stored(manifest, str(uuid4()), revision, "hash-1")
    claim = manifest.claim_next(revision_id=revision)
    assert claim["revision_id"] == revision
    # No second worker can take the same document while the lease is valid.
    assert manifest.claim_next(revision_id=revision) is None
    report = {
        "document_id": revision, "document_hash": "hash-1",
        "status": "pending_human_review", "passed": True,
    }
    with pytest.raises(ValueError, match="Claim expired"):
        manifest.complete_workflow(revision, "hash-1", uuid4(), report)
    manifest.complete_workflow(revision, "hash-1", claim["claim_token"], report)
    assert manifest.get_report(revision) == report
    approved = manifest.review_report(
        revision, "hash-1", "pending_human_review", "approve", "reviewer", "Checked evidence"
    )
    assert approved["status"] == "approved"
    assert manifest.get_report(revision)["review"]["reviewer"] == "reviewer"
    with pytest.raises(ValueError, match="stale"):
        manifest.review_report(
            revision, "hash-1", "pending_human_review", "reject", "other", "Stale"
        )


def test_failed_and_superseded_revisions_cannot_complete(manifest):
    source = str(uuid4())
    first, second = str(uuid4()), str(uuid4())
    _stored(manifest, source, first, "hash-1")
    claim = manifest.claim_next(revision_id=first)
    _stored(manifest, source, second, "hash-2")
    with pytest.raises(ValueError, match="Claim expired"):
        manifest.complete_workflow(first, "hash-1", claim["claim_token"], {
            "document_id": first, "document_hash": "hash-1", "status": "review_required"
        })
    failed = str(uuid4())
    _prepare(manifest, str(uuid4()), failed, "hash-f")
    manifest.mark_failed(failed, "embedding failed")
    with pytest.raises(ValueError, match="incomplete or failed"):
        manifest.mark_stored(failed)
    assert manifest.reset_for_retry(failed)
