"""Chunk metadata contract enforcement.

The contract is defined once by the :class:`~emrag.core.types.Chunk` model. This
module exports it as JSON Schema (for documentation and cross-language
consumers) and provides the batch validation the ingestion quality gate runs
before anything is written or upserted.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from emrag.core.types import Chunk
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

SCHEMA_ID = "https://emrag.dev/schemas/chunk-1.0.json"


def chunk_json_schema() -> dict[str, Any]:
    """Return the chunk contract as a JSON Schema document."""
    schema = Chunk.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = SCHEMA_ID
    schema["title"] = "Chunk"
    return schema


class ContractViolation(Exception):
    """Raised when a batch of chunks breaks the metadata contract."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(f"{len(problems)} chunk contract violation(s)")
        self.problems = problems


def validate_chunks(chunks: list[dict[str, Any]] | list[Chunk]) -> list[Chunk]:
    """Validate chunks and return typed models, raising on any violation."""
    validated: list[Chunk] = []
    problems: list[str] = []
    for index, item in enumerate(chunks):
        if isinstance(item, Chunk):
            validated.append(item)
            continue
        try:
            validated.append(Chunk.model_validate(item))
        except ValidationError as exc:
            problems.append(f"[{index}] {exc.errors()[:3]}")
    if problems:
        raise ContractViolation(problems)
    return validated


def check_invariants(chunks: list[Chunk], *, page_count: int) -> list[str]:
    """Return a list of contract invariant failures (empty means healthy).

    These are the cross-chunk rules a single-model validator cannot express.
    """
    problems: list[str] = []

    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            problems.append(f"duplicate chunk_id {chunk.chunk_id}")
        seen.add(chunk.chunk_id)

    doc_ids = {c.doc_id for c in chunks}
    if len(doc_ids) > 1:
        problems.append(f"chunks span multiple documents: {sorted(doc_ids)}")

    tenants = {c.tenant_id for c in chunks}
    if len(tenants) > 1:
        problems.append(f"chunks span multiple tenants: {sorted(tenants)}")

    for chunk in chunks:
        if chunk.page > page_count:
            problems.append(f"{chunk.chunk_id} page {chunk.page} exceeds page_count {page_count}")
        if not chunk.text.strip():
            problems.append(f"{chunk.chunk_id} has empty text")

    from emrag.core.textnorm import text_hash

    for chunk in chunks:
        if text_hash(chunk.text) != chunk.text_hash:
            problems.append(f"{chunk.chunk_id} text_hash does not match text")

    ids = {c.chunk_id for c in chunks}
    for chunk in chunks:
        if chunk.prev_id and chunk.prev_id not in ids:
            problems.append(f"{chunk.chunk_id} prev_id dangles")
        if chunk.next_id and chunk.next_id not in ids:
            problems.append(f"{chunk.chunk_id} next_id dangles")

    return problems
