"""Quota usage and chunk inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from emrag.api.deps import AdminDep, PrincipalDep, StateDep
from emrag.api.schemas.documents import ChunkPublic
from emrag.api.schemas.query import UsageResponse
from emrag.core.errors import NotFound, ValidationFailed
from emrag.core.ids import CHUNK_ID_RE
from emrag.observability.metrics import METRICS

router = APIRouter(prefix="/v1", tags=["ops"])


@router.get("/usage", response_model=UsageResponse, summary="LLM quota usage")
async def get_usage(state: StateDep, principal: AdminDep) -> UsageResponse:
    """Return the quota ledger snapshot for every routable model."""
    snapshot = await state.router.ledger.snapshot()
    return UsageResponse(window_start=snapshot["window_start"], models=snapshot["models"])


@router.get("/chunks/{chunk_id}", response_model=ChunkPublic, summary="Get one chunk")
async def get_chunk(chunk_id: str, state: StateDep, principal: PrincipalDep) -> ChunkPublic:
    """Return a single chunk by id, for citation inspection."""
    if not CHUNK_ID_RE.match(chunk_id):
        raise ValidationFailed(f"malformed chunk id: {chunk_id!r}")
    chunks = await state.store.get_chunks(principal.tenant_id, [chunk_id])
    if not chunks:
        raise NotFound(f"chunk {chunk_id} not found")
    return ChunkPublic.from_chunk(chunks[0])


@router.get("/metrics", summary="In-process counters and latency histograms")
async def get_metrics(state: StateDep, principal: AdminDep) -> dict[str, object]:
    """Return the in-process metric snapshot."""
    return METRICS.snapshot()
