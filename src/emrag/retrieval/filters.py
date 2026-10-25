"""Request filter translation.

Converts the user-supplied part of a query request into the provider-neutral
:class:`~emrag.core.types.RetrievalFilter`, always composed on top of the
principal's tenant and ACL clauses. User input can narrow a search, never widen
it.
"""

from __future__ import annotations

from typing import cast

from emrag.core.errors import ValidationFailed
from emrag.core.ids import DOC_ID_RE
from emrag.core.types import Modality, Principal, RetrievalFilter
from emrag.security.acl import build_retrieval_filter

_VALID_MODALITIES = {"text", "table", "figure", "ocr_text"}
_MAX_DOC_IDS = 50
_MAX_TAGS = 20


def from_request(
    principal: Principal,
    *,
    doc_ids: list[str] | None = None,
    page_range: list[int] | tuple[int, int] | None = None,
    modalities: list[str] | None = None,
    tags: list[str] | None = None,
    section_contains: str | None = None,
) -> RetrievalFilter:
    """Build a retrieval filter from a validated request payload."""
    if doc_ids:
        if len(doc_ids) > _MAX_DOC_IDS:
            raise ValidationFailed(f"at most {_MAX_DOC_IDS} doc_ids may be supplied")
        for doc_id in doc_ids:
            if not DOC_ID_RE.match(doc_id):
                raise ValidationFailed(f"malformed doc_id: {doc_id!r}")

    if modalities:
        unknown = sorted(set(modalities) - _VALID_MODALITIES)
        if unknown:
            raise ValidationFailed(f"unknown modalities: {unknown}")

    if tags and len(tags) > _MAX_TAGS:
        raise ValidationFailed(f"at most {_MAX_TAGS} tags may be supplied")

    page_tuple: tuple[int, int] | None = None
    if page_range is not None:
        if len(page_range) != 2:
            raise ValidationFailed("page_range must be [first, last]")
        first, last = int(page_range[0]), int(page_range[1])
        if first < 1 or last < first:
            raise ValidationFailed("page_range must satisfy 1 <= first <= last")
        page_tuple = (first, last)

    if section_contains is not None and len(section_contains) > 200:
        raise ValidationFailed("section_contains must be 200 characters or fewer")

    return build_retrieval_filter(
        principal,
        doc_ids=doc_ids,
        page_range=page_tuple,
        modalities=list(modalities or []) or None,  # type: ignore[arg-type]
        tags=tags,
        section_contains=section_contains,
    )


def describe(filters: RetrievalFilter) -> dict[str, object]:
    """Compact representation for logs and the retrieval debug block."""
    return {
        "tenant_id": filters.tenant_id,
        "acl_groups": filters.acl_groups,
        "max_classification": filters.max_classification,
        "doc_ids": len(filters.doc_ids or []),
        "modalities": filters.modalities,
        "page_range": filters.page_range,
        "tags": filters.tags,
    }


def modality_list(values: list[str] | None) -> list[Modality] | None:
    """Narrow a list of strings to the modality literal type, dropping unknowns."""
    if not values:
        return None
    out: list[Modality] = []
    for value in values:
        if value in _VALID_MODALITIES:
            out.append(cast("Modality", value))
    return out or None
