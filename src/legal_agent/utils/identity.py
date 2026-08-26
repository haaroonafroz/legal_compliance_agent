"""Stable identities for acquired documents and their revisions."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def canonicalize_url(url: str) -> str:
    """Normalize a source URL while preserving meaningful query parameters."""
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if not scheme or not hostname:
        return url.strip()

    port = parsed.port
    netloc = hostname
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{netloc}:{port}"

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in _TRACKING_QUERY_KEYS
        and not any(key.startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES)
    ]
    query.sort()
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


def source_key(source_id: str, url: str, *, external_id: str = "") -> str:
    """Return the stable logical identity of a source document."""
    identity = external_id.strip() or canonicalize_url(url)
    return f"{source_id.strip()}::{identity}"


def content_hash(text: str) -> str:
    """Hash normalized content so formatting-only changes do not create revisions."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def revision_id(source: str, text: str) -> str:
    """Return a deterministic revision identifier for a logical source document."""
    return hashlib.sha256(f"{source}::{content_hash(text)}".encode()).hexdigest()[:32]


def point_id(revision: str, chunk_index: int) -> int:
    """Return a stable Qdrant-compatible integer point ID."""
    return int(hashlib.sha256(f"{revision}::{chunk_index}".encode()).hexdigest()[:32], 16) % (
        2**63
    )
