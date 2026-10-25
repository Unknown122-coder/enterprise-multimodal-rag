"""Unit tests for the retrieval gate.

The gate is the cheapest defence against an ungrounded answer: below the
threshold, no model is called at all. These tests pin the property that made it
necessary to rewrite once already, namely that fused retrieval scores measure
*rank* rather than relevance and must never be used as the gating signal.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from emrag.core.ids import chunk_id as make_chunk_id
from emrag.core.ids import doc_id_for_bytes
from emrag.core.ids import parent_id as make_parent_id
from emrag.core.textnorm import text_hash
from emrag.core.types import Candidate, Chunk
from emrag.retrieval.retriever import evaluate_gate, lexical_overlap

DOC_ID = doc_id_for_bytes(b"gate-fixture")

PASSAGE = (
    "Segment revenue grew 14 percent year over year, driven by renewals in the enterprise tier."
)


def _candidate(
    text: str = PASSAGE, *, score: float = 1.0, rerank: float | None = None
) -> Candidate:
    chunk = Chunk(
        chunk_id=make_chunk_id(DOC_ID, 1, 1),
        doc_id=DOC_ID,
        tenant_id="demo",
        filename="f.pdf",
        page=1,
        page_count=1,
        bbox=[0.0, 0.0, 1.0, 1.0],
        modality="text",
        text=text,
        text_hash=text_hash(text),
        token_count=20,
        parent_id=make_parent_id(DOC_ID, 1),
        acl_groups=["all"],
        ingest_version="1.0.0",
        pipeline_profile="full",
        embed_model="t",
        sparse_model="t",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return Candidate(chunk=chunk, score=score, rerank_score=rerank)


class TestLexicalOverlap:
    """The fallback signal must be about relevance, not rank."""

    def test_full_overlap(self) -> None:
        assert lexical_overlap("segment revenue", PASSAGE) == 1.0

    def test_partial_overlap(self) -> None:
        assert 0.0 < lexical_overlap("segment revenue margin", PASSAGE) < 1.0

    def test_no_overlap(self) -> None:
        assert lexical_overlap("capital of Peru", PASSAGE) == 0.0

    def test_stopwords_are_ignored(self) -> None:
        # Without stopword removal, "what is the" alone would score above zero.
        assert lexical_overlap("what is the", PASSAGE) == 0.0

    def test_blank_question_scores_zero(self) -> None:
        assert lexical_overlap("   ", PASSAGE) == 0.0

    def test_case_and_punctuation_insensitive(self) -> None:
        assert lexical_overlap("SEGMENT, REVENUE!", PASSAGE) == 1.0


class TestGate:
    """Gating behaviour across both score families."""

    def test_no_candidates_never_passes(self) -> None:
        passed, score = evaluate_gate("anything", [], 0.15, reranked=False)
        assert passed is False
        assert score is None

    def test_rerank_logit_is_squashed(self) -> None:
        passed, score = evaluate_gate("x", [_candidate(rerank=4.0)], 0.5, reranked=True)
        assert passed is True
        assert score is not None
        assert 0.9 < score < 1.0

    def test_negative_rerank_logit_is_blocked(self) -> None:
        passed, score = evaluate_gate("x", [_candidate(rerank=-6.0)], 0.5, reranked=True)
        assert passed is False
        assert score is not None
        assert score < 0.01

    def test_fused_rank_score_is_not_used_when_not_reranked(self) -> None:
        # A top-ranked but irrelevant passage has the maximum fused score. Gating
        # on that score would pass; gating on relevance must not.
        passed, score = evaluate_gate(
            "what is the capital of Peru", [_candidate(score=1.0)], 0.15, reranked=False
        )
        assert passed is False
        assert score == 0.0

    def test_relevant_passage_passes_without_rerank(self) -> None:
        passed, score = evaluate_gate(
            "segment revenue growth", [_candidate(score=0.016)], 0.15, reranked=False
        )
        assert passed is True
        assert score is not None
        assert score > 0.15

    def test_falls_back_when_rerank_score_is_missing(self) -> None:
        # reranked=True but the model produced no score: do not read None as 0.
        passed, score = evaluate_gate(
            "segment revenue", [_candidate(rerank=None)], 0.15, reranked=True
        )
        assert passed is True
        assert score == 1.0

    @pytest.mark.parametrize("tau,expected", [(0.0, True), (0.4, True), (0.6, False)])
    def test_threshold_is_respected(self, tau: float, expected: bool) -> None:
        passed, _ = evaluate_gate(
            "segment revenue margin headcount", [_candidate(score=1.0)], tau, reranked=False
        )
        assert passed is expected
