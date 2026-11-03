CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS document_manifest (
    revision_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_key TEXT NOT NULL,
    source_url TEXT NOT NULL,
    document_hash TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','stored','dead_letter','superseded')),
    is_current BOOLEAN NOT NULL DEFAULT false,
    supersedes_revision_id TEXT NOT NULL DEFAULT '',
    expected_chunks INTEGER NOT NULL DEFAULT 0 CHECK (expected_chunks >= 0),
    indexed_chunks INTEGER NOT NULL DEFAULT 0 CHECK (indexed_chunks >= 0),
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    workflow_status TEXT NOT NULL DEFAULT 'pending' CHECK (workflow_status IN ('pending','irrelevant','review_required','pending_human_review','approved','rejected','superseded')),
    claim_token UUID,
    claim_until TIMESTAMPTZ,
    report JSONB,
    review JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_manifest_current_source ON document_manifest(source_key) WHERE is_current;
CREATE INDEX IF NOT EXISTS idx_manifest_retry ON document_manifest(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_manifest_pending ON document_manifest(updated_at) WHERE is_current AND status='stored' AND workflow_status='pending';
CREATE TABLE IF NOT EXISTS indexed_chunks (
    revision_id TEXT NOT NULL REFERENCES document_manifest(revision_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    PRIMARY KEY (revision_id, chunk_index)
);
