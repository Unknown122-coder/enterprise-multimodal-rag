"""Deterministic identifier derivation.

Identifiers must be stable across re-ingestion so that artifacts remain the source
of truth and a vector store can be rebuilt with identical point IDs.
"""

from __future__ import annotations

import hashlib
import re
import uuid

POINT_NAMESPACE = uuid.UUID("6f1d9b2a-7c3e-4e1a-9d2b-0e4a8c000001")

DOC_ID_RE = re.compile(r"^d_[0-9a-f]{16}$")
CHUNK_ID_RE = re.compile(r"^d_[0-9a-f]{16}:p[0-9]{3,4}:c[0-9]{4}$")
PARENT_ID_RE = re.compile(r"^d_[0-9a-f]{16}:s[0-9]{4}(:[0-9]+)?$")
TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


class IdError(ValueError):
    """Raised when an identifier is malformed."""


def doc_id_for_bytes(data: bytes) -> str:
    """Return the content-addressed document id for raw file bytes."""
    return "d_" + hashlib.sha256(data).hexdigest()[:16]


def chunk_id(doc_id: str, page: int, seq: int) -> str:
    """Build ``{doc_id}:p{page:03d}:c{seq:04d}``."""
    if not DOC_ID_RE.match(doc_id):
        raise IdError(f"invalid doc_id: {doc_id!r}")
    if page < 1:
        raise IdError(f"page must be 1-based, got {page}")
    if not 0 <= seq <= 9999:
        raise IdError(f"chunk sequence out of range: {seq}")
    return f"{doc_id}:p{page:03d}:c{seq:04d}"


def parent_id(doc_id: str, section_seq: int, part: int | None = None) -> str:
    """Build ``{doc_id}:s{section_seq:04d}[:{part}]``."""
    if not DOC_ID_RE.match(doc_id):
        raise IdError(f"invalid doc_id: {doc_id!r}")
    base = f"{doc_id}:s{section_seq:04d}"
    return base if part is None else f"{base}:{part}"


def point_id(chunk_identifier: str) -> str:
    """Return the vector-store primary key (UUIDv5) for a chunk id."""
    if not CHUNK_ID_RE.match(chunk_identifier):
        raise IdError(f"invalid chunk_id: {chunk_identifier!r}")
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_identifier))


def doc_id_of_chunk(chunk_identifier: str) -> str:
    """Extract the document id from a chunk id."""
    if not CHUNK_ID_RE.match(chunk_identifier):
        raise IdError(f"invalid chunk_id: {chunk_identifier!r}")
    return chunk_identifier.split(":", 1)[0]


def page_of_chunk(chunk_identifier: str) -> int:
    """Extract the 1-based page number from a chunk id."""
    if not CHUNK_ID_RE.match(chunk_identifier):
        raise IdError(f"invalid chunk_id: {chunk_identifier!r}")
    return int(chunk_identifier.split(":")[1][1:])


def table_id(doc_id: str, page: int, seq: int) -> str:
    """Build a stable table identifier."""
    return f"t_{doc_id[2:]}_{page:03d}_{seq:02d}"


def figure_id(doc_id: str, page: int, seq: int) -> str:
    """Build a stable figure identifier."""
    return f"f_{doc_id[2:]}_{page:03d}_{seq:02d}"


def validate_tenant_id(tenant_id: str) -> str:
    """Return ``tenant_id`` if it is a valid slug, else raise."""
    if not TENANT_ID_RE.match(tenant_id):
        raise IdError(f"invalid tenant_id: {tenant_id!r}")
    return tenant_id


def new_job_id() -> str:
    """Return a fresh job identifier."""
    return "job_" + uuid.uuid4().hex[:20]


def new_request_id() -> str:
    """Return a fresh request identifier."""
    return "req_" + uuid.uuid4().hex[:20]
