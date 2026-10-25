"""Access control helpers.

The retrieval filter is the enforcement point: a principal can only ever match
chunks in its own tenant, sharing at least one ACL group, at or below its
clearance. Nothing above this module is allowed to construct a
:class:`RetrievalFilter` without a :class:`Principal`.
"""

from __future__ import annotations

from emrag.core.errors import Forbidden
from emrag.core.types import (
    CLASSIFICATION_ORDER,
    Classification,
    DocumentRecord,
    Modality,
    Principal,
    RetrievalFilter,
)


def build_retrieval_filter(
    principal: Principal,
    *,
    doc_ids: list[str] | None = None,
    page_range: tuple[int, int] | None = None,
    modalities: list[Modality] | None = None,
    tags: list[str] | None = None,
    section_contains: str | None = None,
) -> RetrievalFilter:
    """Build the only filter the retriever will accept.

    The tenant and ACL clauses come from the credential, never from the request
    body, which is what makes cross-tenant retrieval impossible by construction.
    """
    groups = list(principal.groups)
    if "all" not in groups:
        groups.append("all")
    return RetrievalFilter(
        tenant_id=principal.tenant_id,
        acl_groups=groups,
        max_classification=principal.clearance,
        doc_ids=doc_ids or None,
        page_range=page_range,
        modalities=modalities or None,
        tags=tags or None,
        section_contains=section_contains,
    )


def can_read_document(principal: Principal, record: DocumentRecord) -> bool:
    """Return True when the principal may see this document's metadata."""
    if record.tenant_id != principal.tenant_id:
        return False
    if principal.is_admin():
        return True
    if not _classification_allowed(principal.clearance, record.classification):
        return False
    groups = {*principal.groups, "all"}
    return bool(groups & set(record.acl_groups))


def require_read(principal: Principal, record: DocumentRecord) -> DocumentRecord:
    """Return the record or raise :class:`Forbidden`."""
    if not can_read_document(principal, record):
        raise Forbidden(f"document {record.doc_id} is not accessible to this principal")
    return record


def require_write(principal: Principal) -> None:
    """Raise unless the principal may create or delete documents."""
    if not principal.can_write():
        raise Forbidden("this credential is read-only")


def _classification_allowed(clearance: Classification, level: Classification) -> bool:
    return CLASSIFICATION_ORDER[level] <= CLASSIFICATION_ORDER[clearance]
