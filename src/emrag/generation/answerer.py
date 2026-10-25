"""Answer generation.

The single orchestrator for both query paths. For a non-streaming request:

    refusal check -> retrieve -> gate -> generate (JSON) -> verify -> optional
    repair round -> render -> status policy

For a streaming request the same stages run, but the markdown body is streamed to
the client while the citation trailer is buffered, verified after the stream
ends, and emitted as ``citations`` and ``verification`` events. A ``revision``
event always follows, carrying the final text: the streamed body uses the model's
evidence aliases while the citation list uses dense reference numbers, so clients
must replace what they displayed with the revision payload.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from emrag.citation import policy
from emrag.citation.contract import (
    ANSWER_SCHEMA,
    REPAIR_SCHEMA,
    EvidenceBundle,
    build_answer_messages,
    build_evidence,
    build_repair_messages,
    build_stream_messages,
    markers_in,
    parse_answer_json,
    split_stream_output,
    strip_markers,
)
from emrag.citation.renderer import render, render_nearest
from emrag.citation.verifier import CitationVerifier, VerificationReport, VerifiedClaim
from emrag.core.errors import QuotaExhausted
from emrag.core.ids import new_request_id
from emrag.core.types import (
    Answer,
    Citation,
    ModelTier,
    QueryResult,
    RetrievalFilter,
    RetrievalInfo,
    Usage,
)
from emrag.llm.base import LLMMessage, LLMRequest
from emrag.llm.router import LLMRouter
from emrag.observability.logging import get_logger
from emrag.observability.metrics import METRICS
from emrag.retrieval.retriever import RetrievalOptions, RetrievalOutcome, Retriever
from emrag.security.injection import is_suspicious_question

logger = get_logger(__name__)

MAX_OUTPUT_TOKENS = 700
REPAIR_OUTPUT_TOKENS = 500


@dataclass
class AnswerOptions:
    """Per-request generation knobs."""

    model_tier: ModelTier = "fast"
    repair: bool = True
    include_page_images: bool = True
    deadline_ms: int = 4500
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = 0.0
    debug: bool = False


class Answerer:
    """Produces verified, cited answers."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        router: LLMRouter,
        verifier: CitationVerifier,
    ) -> None:
        self._retriever = retriever
        self._router = router
        self._verifier = verifier

    # ---- non-streaming ------------------------------------------------------
    async def answer(
        self,
        question: str,
        filters: RetrievalFilter,
        *,
        retrieval: RetrievalOptions,
        options: AnswerOptions,
        conversation_summary: str | None = None,
        request_id: str | None = None,
    ) -> QueryResult:
        """Answer one question with verified citations."""
        started = time.perf_counter()
        request_id = request_id or new_request_id()

        refusal_reason = is_suspicious_question(question)
        if refusal_reason:
            decision = policy.refusal(refusal_reason)
            logger.info("query.refused", reason=refusal_reason)
            return QueryResult(
                request_id=request_id,
                status=decision.status,
                reason=decision.reason,
                warnings=decision.warnings,
                retrieval=RetrievalInfo(strategy="refused", candidates=0, used=0),
                latency_ms=_elapsed_ms(started),
            )

        outcome = await self._retriever.retrieve(question, filters, retrieval)
        info = _retrieval_info(outcome)

        if not outcome.gate_passed or outcome.evidence is None:
            decision = policy.decide(
                None,
                gate_passed=outcome.gate_passed,
                model_said_sufficient=False,
                had_candidates=bool(outcome.candidates),
            )
            METRICS.increment("query.total", status=decision.status)
            METRICS.increment("query.gate_blocked")
            return QueryResult(
                request_id=request_id,
                status=decision.status,
                reason=decision.reason,
                warnings=decision.warnings,
                nearest_sources=render_nearest(
                    outcome.candidates, include_page_images=options.include_page_images
                ),
                retrieval=info,
                latency_ms=_elapsed_ms(started),
                debug=_debug(outcome) if options.debug else None,
            )

        bundle = build_evidence(outcome.evidence.blocks)
        system, user = build_answer_messages(
            question, bundle, conversation_summary=conversation_summary
        )
        remaining_ms = options.deadline_ms - _elapsed_ms(started)

        try:
            response = await self._router.complete(
                options.model_tier,
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    max_tokens=options.max_output_tokens,
                    temperature=options.temperature,
                    json_schema=ANSWER_SCHEMA,
                    schema_name="cited_answer",
                ),
                deadline_ms=max(1000, remaining_ms),
            )
        except QuotaExhausted:
            raise
        usage = Usage(
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )

        try:
            payload = parse_answer_json(response.text)
        except ValueError as exc:
            logger.warning("generation.unparsable", error=str(exc)[:200])
            payload = {"sufficient_evidence": False, "answer": []}

        sufficient = bool(payload.get("sufficient_evidence", False))
        raw_claims = payload.get("answer") or []
        report = self._verifier.verify_answer(raw_claims, bundle)

        remaining_ms = options.deadline_ms - _elapsed_ms(started)
        if sufficient and policy.should_repair(
            report, enabled=options.repair, budget_ms=remaining_ms
        ):
            report, repair_usage = await self._repair(question, bundle, report, remaining_ms)
            if repair_usage is not None:
                usage.prompt_tokens += repair_usage.prompt_tokens
                usage.completion_tokens += repair_usage.completion_tokens
                usage.repair_rounds += 1

        decision = policy.decide(
            report,
            gate_passed=True,
            model_said_sufficient=sufficient,
            had_candidates=True,
        )
        answer, citations = render(
            report.supported_claims, include_page_images=options.include_page_images
        )
        warnings = list(decision.warnings)
        for aspect in payload.get("unanswered_aspects") or []:
            text = str(aspect).strip()
            if text:
                warnings.append(f"unanswered:{text[:200]}")

        result = QueryResult(
            request_id=request_id,
            status=decision.status,
            answer=answer if decision.status in ("answered", "partial") else None,
            citations=citations if decision.status in ("answered", "partial") else [],
            nearest_sources=(
                render_nearest(outcome.candidates, include_page_images=options.include_page_images)
                if decision.status == "not_found"
                else []
            ),
            warnings=warnings,
            reason=decision.reason,
            retrieval=info,
            generation=usage,
            latency_ms=_elapsed_ms(started),
            debug=_debug(outcome, report) if options.debug else None,
        )
        METRICS.increment("query.total", status=result.status)
        METRICS.increment("query.claims_dropped", len(report.dropped_claims))
        METRICS.increment("query.citations_rejected", report.rejected_citations)
        METRICS.observe("query.latency_ms", result.latency_ms)
        logger.info(
            "query.answered",
            status=result.status,
            citations=len(result.citations),
            claims=len(answer.claims),
            dropped=len(report.dropped_claims),
            honesty=round(report.honesty_rate, 3),
            latency_ms=result.latency_ms,
        )
        return result

    # ---- streaming ----------------------------------------------------------
    async def stream(
        self,
        question: str,
        filters: RetrievalFilter,
        *,
        retrieval: RetrievalOptions,
        options: AnswerOptions,
        conversation_summary: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield ``(event_name, payload)`` pairs for the SSE endpoint."""
        started = time.perf_counter()
        request_id = request_id or new_request_id()

        refusal_reason = is_suspicious_question(question)
        if refusal_reason:
            yield (
                "verification",
                {"status": "refused", "dropped_claims": 0, "warnings": [refusal_reason]},
            )
            yield "done", {"request_id": request_id, "latency_ms": _elapsed_ms(started)}
            return

        outcome = await self._retriever.retrieve(question, filters, retrieval)
        yield (
            "retrieval",
            {
                "candidates": [
                    c.model_dump(mode="json")
                    for c in render_nearest(outcome.candidates, limit=retrieval.top_k)
                ],
                "latency_ms": outcome.latency_ms,
            },
        )

        if not outcome.gate_passed or outcome.evidence is None:
            decision = policy.decide(
                None,
                gate_passed=outcome.gate_passed,
                model_said_sufficient=False,
                had_candidates=bool(outcome.candidates),
            )
            yield (
                "verification",
                {
                    "status": decision.status,
                    "dropped_claims": 0,
                    "warnings": decision.warnings,
                    "reason": decision.reason,
                },
            )
            yield "done", {"request_id": request_id, "latency_ms": _elapsed_ms(started)}
            return

        bundle = build_evidence(outcome.evidence.blocks)
        system, user = build_stream_messages(
            question, bundle, conversation_summary=conversation_summary
        )
        spec, deltas = await self._router.stream(
            options.model_tier,
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=system),
                    LLMMessage(role="user", content=user),
                ],
                max_tokens=options.max_output_tokens,
                temperature=options.temperature,
            ),
            deadline_ms=options.deadline_ms,
        )

        collected: list[str] = []
        trailer_started = False
        async for delta in deltas:
            collected.append(delta)
            joined = "".join(collected)
            if not trailer_started and "---CITATIONS---" in joined:
                trailer_started = True
                visible, _ = split_stream_output(joined)
                yield "answer_delta", {"text": "", "complete": True, "body": visible}
                continue
            if not trailer_started:
                yield "answer_delta", {"text": delta}

        raw = "".join(collected)
        body, trailer = split_stream_output(raw)
        claims = _claims_from_stream(body, trailer)
        report = self._verifier.verify_answer(claims, bundle)
        sufficient = bool(trailer.get("sufficient_evidence", bool(claims)))

        decision = policy.decide(
            report, gate_passed=True, model_said_sufficient=sufficient, had_candidates=True
        )
        answer, citations = render(
            report.supported_claims, include_page_images=options.include_page_images
        )
        yield "citations", {"citations": [c.model_dump(mode="json") for c in citations]}
        yield (
            "verification",
            {
                "status": decision.status,
                "dropped_claims": len(report.dropped_claims),
                "warnings": decision.warnings,
                "reason": decision.reason,
            },
        )
        # Always revise. The streamed text carries the model's evidence aliases
        # (``[c3]``), which are not the dense reference numbers in `citations`,
        # and verification may also have removed claims. Clients replace the text
        # they displayed with this payload.
        yield (
            "revision",
            {
                "answer": answer.model_dump(mode="json"),
                "reason": (
                    "claims_dropped" if report.dropped_claims else "citation_markers_renumbered"
                ),
            },
        )
        yield (
            "usage",
            {
                "provider": spec.provider,
                "model": spec.model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )
        yield "done", {"request_id": request_id, "latency_ms": _elapsed_ms(started)}

    # ---- repair -------------------------------------------------------------
    async def _repair(
        self,
        question: str,
        bundle: EvidenceBundle,
        report: VerificationReport,
        remaining_ms: int,
    ) -> tuple[VerificationReport, Usage | None]:
        """Give the model one chance to re-cite the claims that failed."""
        failed = [claim.text for claim in report.dropped_claims]
        if not failed:
            return report, None
        system, user = build_repair_messages(question, bundle, failed)
        try:
            response = await self._router.complete(
                "repair",
                LLMRequest(
                    messages=[
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    max_tokens=REPAIR_OUTPUT_TOKENS,
                    temperature=0.0,
                    json_schema=REPAIR_SCHEMA,
                    schema_name="repaired_claims",
                ),
                deadline_ms=max(1000, remaining_ms - 200),
            )
            payload = parse_answer_json(response.text)
        except Exception as exc:  # noqa: BLE001 - repair is best effort
            logger.warning("generation.repair_failed", error=str(exc)[:200])
            return report, None

        repaired = self._verifier.verify_answer(payload.get("answer") or [], bundle)
        merged = VerificationReport(
            claims=[*report.claims, *repaired.supported_claims],
            dropped_claims=repaired.dropped_claims,
            rejected_citations=report.rejected_citations + repaired.rejected_citations,
            total_citations=report.total_citations + repaired.total_citations,
        )
        merged.claims.sort(key=lambda c: c.index)
        logger.info(
            "generation.repaired",
            recovered=len(repaired.supported_claims),
            still_failing=len(repaired.dropped_claims),
        )
        return merged, Usage(
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _claims_from_stream(body: str, trailer: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct claim dictionaries from streamed markdown and its trailer."""
    sentences = _split_sentences(body)
    by_claim: dict[int, list[dict[str, str]]] = {}
    for entry in trailer.get("citations") or []:
        if not isinstance(entry, dict):
            continue
        index = int(entry.get("claim", 0) or 0)
        by_claim.setdefault(index, []).append(
            {"id": str(entry.get("id", "")), "quote": str(entry.get("quote", ""))}
        )

    claims: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        citations = list(by_claim.get(index, []))
        if not citations:
            # Fall back to the inline markers when the trailer omitted this claim.
            citations = [{"id": alias, "quote": ""} for alias in markers_in(sentence)]
        claims.append({"text": strip_markers(sentence), "citations": citations})
    return claims


def _split_sentences(text: str) -> list[str]:
    """Split a streamed answer into fact-bearing sentences."""
    import re

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*\-])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _retrieval_info(outcome: RetrievalOutcome) -> RetrievalInfo:
    return RetrievalInfo(
        strategy=outcome.strategy,
        candidates=len(outcome.candidates),
        used=outcome.used,
        rewrites=list(outcome.rewrites),
        latency_ms=outcome.latency_ms,
        gate_score=round(outcome.gate_score, 4) if outcome.gate_score is not None else None,
    )


def _debug(outcome: RetrievalOutcome, report: VerificationReport | None = None) -> dict[str, Any]:
    """Build the optional debug block (enabled by ``RETRIEVAL_DEBUG``)."""
    data: dict[str, Any] = {
        "candidates": [
            {
                "chunk_id": c.chunk.chunk_id,
                "page": c.chunk.page,
                "modality": c.chunk.modality,
                "score": round(c.score, 4),
                "rerank_score": round(c.rerank_score, 4) if c.rerank_score is not None else None,
                "section": c.chunk.section_path,
            }
            for c in outcome.candidates[:20]
        ],
        "evidence_aliases": [
            b.alias for b in (outcome.evidence.blocks if outcome.evidence else [])
        ],
        "tokens_used": outcome.evidence.tokens_used if outcome.evidence else 0,
        "rewrites": outcome.rewrites,
    }
    if report is not None:
        data["verification"] = {
            "total_citations": report.total_citations,
            "rejected_citations": report.rejected_citations,
            "dropped_claims": [c.text[:120] for c in report.dropped_claims],
            "honesty_rate": round(report.honesty_rate, 3),
        }
    return data


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def citation_alias_map(citations: list[Citation], claims: list[VerifiedClaim]) -> dict[str, int]:
    """Map evidence aliases to rendered reference numbers, for streamed renumbering."""
    out: dict[str, int] = {}
    for claim in claims:
        for citation in claim.citations:
            if citation.candidate is None:
                continue
            for rendered in citations:
                if rendered.chunk_id == citation.candidate.chunk.chunk_id:
                    out[citation.alias] = rendered.ref
                    break
    return out


__all__ = ["Answer", "AnswerOptions", "Answerer", "citation_alias_map"]
