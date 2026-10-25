"""Retrieval orchestration.

The pipeline for one query:

    embed -> hybrid search (dense + BM25, RRF) -> optional rewrite pass and
    re-fusion -> cross-encoder rerank -> retrieval gate -> context budget

The **retrieval gate** is the cheapest guard against hallucination: when the
evidence scores below a calibrated threshold, no generation call is made at all
and the caller returns ``not_found`` with the nearest sources. That saves quota
and removes the most common path to an ungrounded answer. See
:func:`evaluate_gate` for why the gating signal is not the fused retrieval score.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from emrag.core.types import Candidate, Parent, QueryVectors, RetrievalFilter
from emrag.embeddings.base import EmbeddingProvider
from emrag.llm.router import LLMRouter
from emrag.observability.logging import get_logger
from emrag.observability.tracing import span
from emrag.retrieval.budget import BudgetResult, select_evidence
from emrag.retrieval.rerank import CrossEncoderReranker, NoopReranker
from emrag.retrieval.rewrite import RewriteMode, rewrite_query, should_rewrite
from emrag.storage.docstore import DocStoreCache
from emrag.vectorstore.base import VectorStore, reciprocal_rank_fusion

logger = get_logger(__name__)

Reranker = CrossEncoderReranker | NoopReranker


@dataclass
class RetrievalOptions:
    """Per-request retrieval knobs."""

    top_k: int = 8
    candidates_k: int = 40
    rewrite: RewriteMode = "auto"
    parent_expansion: bool = True
    rerank: bool = True
    token_budget: int = 2500
    gate_tau: float = 0.15
    deadline_ms: int | None = None


@dataclass
class RetrievalOutcome:
    """Everything the generator and the response renderer need."""

    candidates: list[Candidate] = field(default_factory=list)
    evidence: BudgetResult | None = None
    rewrites: list[str] = field(default_factory=list)
    gate_passed: bool = False
    gate_score: float | None = None
    strategy: str = "hybrid_rrf"
    latency_ms: int = 0
    reranked: bool = False

    @property
    def used(self) -> int:
        """Number of evidence blocks selected."""
        return len(self.evidence.blocks) if self.evidence else 0


class Retriever:
    """Hybrid retriever with reranking, gating and context budgeting."""

    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        router: LLMRouter | None = None,
        docstores: DocStoreCache | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._router = router
        self._docstores = docstores

    async def retrieve(
        self, question: str, filters: RetrievalFilter, options: RetrievalOptions
    ) -> RetrievalOutcome:
        """Run the full retrieval pipeline for one question."""
        started = time.perf_counter()
        outcome = RetrievalOutcome()

        candidates = await self._search(question, filters, options)
        best_raw = candidates[0].score if candidates else None

        if self._router is not None and should_rewrite(
            options.rewrite, question, best_score=best_raw
        ):
            outcome.rewrites = await rewrite_query(
                self._router, question, deadline_ms=options.deadline_ms
            )
            if outcome.rewrites:
                candidates = await self._search_many(
                    [question, *outcome.rewrites], filters, options, primary=candidates
                )
                outcome.strategy = "hybrid_rrf+rewrite"

        if options.rerank and candidates:
            with span("rerank", candidates=len(candidates)):
                candidates = await asyncio.to_thread(self._reranker.rerank, question, candidates)
            outcome.reranked = getattr(self._reranker, "available", False)
            if outcome.reranked:
                outcome.strategy += "+rerank"
        else:
            candidates.sort(key=lambda c: c.score, reverse=True)

        outcome.candidates = candidates
        outcome.gate_passed, outcome.gate_score = evaluate_gate(
            question, candidates, options.gate_tau, reranked=outcome.reranked
        )

        if outcome.gate_passed:
            parents = await self._load_parents(candidates, filters, options)
            outcome.evidence = select_evidence(
                candidates,
                token_budget=options.token_budget,
                top_k=options.top_k,
                parents=parents,
                parent_expansion=options.parent_expansion,
            )

        outcome.latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "retrieval.done",
            candidates=len(candidates),
            used=outcome.used,
            gate_passed=outcome.gate_passed,
            gate_score=round(outcome.gate_score, 4) if outcome.gate_score is not None else None,
            strategy=outcome.strategy,
            latency_ms=outcome.latency_ms,
        )
        return outcome

    # ---- stages -------------------------------------------------------------
    async def _search(
        self, question: str, filters: RetrievalFilter, options: RetrievalOptions
    ) -> list[Candidate]:
        """Embed one query and run a single hybrid search."""
        with span("embed_query"):
            vectors = await asyncio.to_thread(self._embedder.embed_query, question)
        with span("vector_search", backend=self._store.backend):
            return await self._store.hybrid_search(
                QueryVectors(
                    dense=vectors.dense,
                    sparse_indices=vectors.sparse_indices,
                    sparse_values=vectors.sparse_values,
                ),
                filters,
                k_dense=options.candidates_k,
                k_sparse=options.candidates_k,
                k_out=options.candidates_k,
            )

    async def _search_many(
        self,
        questions: list[str],
        filters: RetrievalFilter,
        options: RetrievalOptions,
        *,
        primary: list[Candidate] | None = None,
    ) -> list[Candidate]:
        """Search every phrasing in parallel and fuse the rankings with RRF."""
        tasks = [
            self._search(q, filters, options)
            for q in (questions[1:] if primary is not None else questions)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        rankings: list[list[Candidate]] = [primary] if primary is not None else []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("retrieval.rewrite_search_failed", error=str(result)[:160])
                continue
            rankings.append(result)

        by_id: dict[str, Candidate] = {}
        id_rankings: list[list[str]] = []
        for ranking in rankings:
            ids: list[str] = []
            for candidate in ranking:
                by_id.setdefault(candidate.chunk.chunk_id, candidate)
                ids.append(candidate.chunk.chunk_id)
            id_rankings.append(ids)

        # The original question is weighted highest so a bad rewrite cannot
        # displace a good direct hit.
        weights = [1.0] + [0.7] * (len(id_rankings) - 1) if id_rankings else []
        fused = reciprocal_rank_fusion(id_rankings, weights=weights)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        out: list[Candidate] = []
        for chunk_id, score in ordered[: options.candidates_k]:
            candidate = by_id[chunk_id]
            candidate.score = score
            out.append(candidate)
        return out

    async def _load_parents(
        self, candidates: list[Candidate], filters: RetrievalFilter, options: RetrievalOptions
    ) -> dict[str, Parent]:
        """Load parent sections for the strongest hits, if a docstore is present."""
        if not options.parent_expansion or self._docstores is None or not candidates:
            return {}
        wanted: dict[str, str] = {}
        for candidate in candidates[: options.top_k]:
            wanted.setdefault(candidate.chunk.parent_id, candidate.chunk.doc_id)
            if len(wanted) >= 4:
                break
        parents: dict[str, Parent] = {}
        for parent_id, doc_id in wanted.items():
            shard = await self._docstores.get(filters.tenant_id, doc_id)
            if shard is None:
                continue
            parent = await asyncio.to_thread(shard.get_parent, parent_id)
            if parent is not None and parent.tenant_id == filters.tenant_id:
                parents[parent_id] = parent
        return parents


# Word and number terms; keeps decimals like 48.2 intact.
_TERM_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
# A whitespace-delimited string rather than a list literal: a 70-element
# vertical list in the middle of this module costs more than it explains.
_STOPWORDS = frozenset(
    """a an and any are as at be been by can did do does for from had has have how in into
    is it its of on or over that the their there these this to under was were what when
    where which who whom why will with would""".split()  # noqa: SIM905
)


def evaluate_gate(
    question: str, candidates: list[Candidate], tau: float, *, reranked: bool
) -> tuple[bool, float | None]:
    """Decide whether the evidence is strong enough to call a model.

    Returns ``(passed, score)`` where the score is comparable to ``tau`` on a
    ``[0, 1]`` scale.

    The signal depends on what is available, because the two score families mean
    different things:

    * **reranked** - the cross-encoder emits a relevance logit, squashed here
      with a sigmoid. This is the intended configuration and the strongest
      signal.
    * **not reranked** - the fused retrieval score from reciprocal rank fusion is
      a function of *rank*, not similarity: the best candidate scores highest
      whether or not it is relevant at all. Gating on it would pass everything.
      So the fallback measures lexical overlap between the question's content
      words and the top passage, which is cheap, local and actually about
      relevance.

    Disabling the reranker therefore weakens this guard. See
    ``docs/adr/0005-citation-verification.md``.
    """
    if not candidates:
        return False, None

    if reranked and candidates[0].rerank_score is not None:
        logit = candidates[0].rerank_score
        score = 1.0 / (1.0 + pow(2.718281828, -logit))
    else:
        score = lexical_overlap(question, candidates[0].chunk.text)

    return score >= tau, score


def lexical_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that appear in the passage.

    A deliberately blunt relevance proxy for when no reranker is available. It
    answers "does this passage even mention what was asked about?", which is the
    question the gate needs answered.

    Tokens are word characters only, so a trailing question mark or an attached
    comma cannot cause a false miss. Decimals and figures like ``48.2`` are kept
    whole, because they are often exactly what a question is about.
    """
    query_terms = _content_terms(question)
    if not query_terms:
        return 0.0
    return len(query_terms & _terms(passage)) / len(query_terms)


def _terms(text: str) -> set[str]:
    """Tokenise into word and number terms, normalised."""
    from emrag.core.textnorm import normalize

    return set(_TERM_RE.findall(normalize(text)))


def _content_terms(text: str) -> set[str]:
    """Tokenise and drop stopwords and single characters."""
    return {t for t in _terms(text) if t not in _STOPWORDS and len(t) > 1}
