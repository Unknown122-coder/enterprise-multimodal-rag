"""Retrieval-only endpoint.

Kept in its own module because its audience is different from ``/v1/query``: this
is what evaluation runs, source browsers and debugging sessions call. It spends
no model quota, so it can be hit freely while tuning.
"""

from __future__ import annotations

from fastapi import APIRouter

from emrag.api.deps import PrincipalDep, StateDep
from emrag.api.schemas.query import RetrievedChunk, RetrieveRequest, RetrieveResponse
from emrag.observability.logging import get_logger
from emrag.retrieval import filters as filter_builder
from emrag.retrieval.retriever import RetrievalOptions

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["query"])


@router.post(
    "/retrieve", response_model=RetrieveResponse, summary="Retrieve passages without generating"
)
async def post_retrieve(
    body: RetrieveRequest, state: StateDep, principal: PrincipalDep
) -> RetrieveResponse:
    """Return ranked passages with their text, scores and page links."""
    supplied = body.filters
    retrieval_filter = filter_builder.from_request(
        principal,
        doc_ids=supplied.doc_ids if supplied else None,
        page_range=list(supplied.page_range) if supplied and supplied.page_range else None,
        modalities=list(supplied.modalities) if supplied and supplied.modalities else None,
        tags=supplied.tags if supplied else None,
        section_contains=supplied.section_contains if supplied else None,
    )
    cfg = state.settings
    outcome = await state.retriever.retrieve(
        body.question,
        retrieval_filter,
        RetrievalOptions(
            top_k=body.top_k,
            candidates_k=cfg.candidates_k,
            rewrite="off",
            parent_expansion=False,
            rerank=body.rerank and cfg.rerank_enabled,
            token_budget=cfg.evidence_token_budget,
            gate_tau=cfg.retrieval_gate_tau,
        ),
    )
    candidates = [
        RetrievedChunk(
            chunk_id=c.chunk.chunk_id,
            document_id=c.chunk.doc_id,
            filename=c.chunk.filename,
            page=c.chunk.page,
            bbox=list(c.chunk.bbox),
            modality=c.chunk.modality,
            section_path=list(c.chunk.section_path),
            text=c.chunk.text,
            score=round(c.score, 6),
            rerank_score=round(c.rerank_score, 6) if c.rerank_score is not None else None,
            page_image_url=f"/v1/documents/{c.chunk.doc_id}/pages/{c.chunk.page}",
        )
        for c in outcome.candidates[: body.top_k]
    ]
    return RetrieveResponse(
        candidates=candidates,
        gate_passed=outcome.gate_passed,
        gate_score=outcome.gate_score,
        strategy=outcome.strategy,
        latency_ms=outcome.latency_ms,
    )
