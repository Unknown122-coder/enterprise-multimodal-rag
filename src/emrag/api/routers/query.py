"""Query endpoints: answer and stream. Retrieval-only lives in ``retrieve.py``."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from emrag.api.deps import PrincipalDep, StateDep
from emrag.api.schemas.query import QueryRequest, QueryResponse
from emrag.api.sse import sse_stream
from emrag.core.types import Principal
from emrag.generation.answerer import AnswerOptions
from emrag.observability.logging import get_logger
from emrag.retrieval import filters as filter_builder
from emrag.retrieval.retriever import RetrievalOptions

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["query"])


def _retrieval_options(state: StateDep, request: QueryRequest) -> RetrievalOptions:
    cfg = state.settings
    return RetrievalOptions(
        top_k=request.options.top_k,
        candidates_k=cfg.candidates_k,
        rewrite=request.options.rewrite,
        parent_expansion=request.options.parent_expansion,
        rerank=cfg.rerank_enabled,
        token_budget=cfg.evidence_token_budget,
        gate_tau=cfg.retrieval_gate_tau,
        deadline_ms=request.options.deadline_ms,
    )


def _answer_options(state: StateDep, request: QueryRequest) -> AnswerOptions:
    generation = state.settings.tuning_section("generation")
    return AnswerOptions(
        model_tier=request.options.model_tier,
        repair=request.options.repair and bool(generation.get("repair_enabled", True)),
        include_page_images=request.options.include_page_images,
        deadline_ms=request.options.deadline_ms,
        max_output_tokens=int(generation.get("max_output_tokens", 700)),
        temperature=float(generation.get("temperature", 0.0)),
        debug=state.settings.retrieval_debug,
    )


def _filters(principal: Principal, request: QueryRequest) -> object:
    supplied = request.filters
    return filter_builder.from_request(
        principal,
        doc_ids=supplied.doc_ids if supplied else None,
        page_range=list(supplied.page_range) if supplied and supplied.page_range else None,
        modalities=list(supplied.modalities) if supplied and supplied.modalities else None,
        tags=supplied.tags if supplied else None,
        section_contains=supplied.section_contains if supplied else None,
    )


@router.post("/query", response_model=QueryResponse, summary="Answer a question with citations")
async def post_query(
    body: QueryRequest, state: StateDep, principal: PrincipalDep, request: Request
) -> QueryResponse:
    """Answer a question, returning only claims with verified citations."""
    result = await state.answerer.answer(
        body.question,
        _filters(principal, body),  # type: ignore[arg-type]
        retrieval=_retrieval_options(state, body),
        options=_answer_options(state, body),
        conversation_summary=body.conversation.summary if body.conversation else None,
        request_id=getattr(request.state, "request_id", None),
    )
    return QueryResponse.from_result(result)


@router.post("/query/stream", summary="Answer a question as a server-sent event stream")
async def post_query_stream(
    body: QueryRequest, state: StateDep, principal: PrincipalDep, request: Request
) -> StreamingResponse:
    """Stream an answer, then emit verified citations and any revision."""
    events = state.answerer.stream(
        body.question,
        _filters(principal, body),  # type: ignore[arg-type]
        retrieval=_retrieval_options(state, body),
        options=_answer_options(state, body),
        conversation_summary=body.conversation.summary if body.conversation else None,
        request_id=getattr(request.state, "request_id", None),
    )
    return StreamingResponse(
        sse_stream(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
