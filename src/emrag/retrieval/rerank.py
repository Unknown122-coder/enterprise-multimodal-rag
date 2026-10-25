"""Cross-encoder reranking.

A 22M-parameter MiniLM cross-encoder scores roughly 40 query-passage pairs in
300-400 ms on two vCPUs, which fits the latency budget and lifts precision
substantially over fused retrieval scores alone. When the model is unavailable
the reranker degrades to a pass-through so retrieval still works.
"""

from __future__ import annotations

import threading
from typing import Any

from emrag.core.types import Candidate
from emrag.observability.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """fastembed-backed cross-encoder, loaded lazily."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Any = None
        self._unavailable = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """False once loading has failed, so callers can report the degradation."""
        return not self._unavailable

    def _ensure_loaded(self) -> Any:
        if self._model is not None or self._unavailable:
            return self._model
        with self._lock:
            if self._model is None and not self._unavailable:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._model = TextCrossEncoder(
                        model_name=self.model_name,
                        cache_dir=self._cache_dir,
                        threads=self._threads,
                    )
                    logger.info("rerank.loaded", model=self.model_name)
                except Exception as exc:  # noqa: BLE001 - optional component
                    self._unavailable = True
                    logger.warning("rerank.unavailable", error=str(exc)[:200])
        return self._model

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        """Score candidates against the query and return them best first.

        Blocking and CPU bound: call through a threadpool from async code.
        """
        if not candidates:
            return []
        model = self._ensure_loaded()
        if model is None:
            return sorted(candidates, key=lambda c: c.score, reverse=True)

        documents = [_passage_for(c) for c in candidates]
        try:
            scores = list(model.rerank(query, documents))
        except Exception as exc:  # noqa: BLE001 - never fail a query on the reranker
            logger.warning("rerank.failed", error=str(exc)[:200])
            return sorted(candidates, key=lambda c: c.score, reverse=True)

        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
        return sorted(candidates, key=lambda c: c.effective_score, reverse=True)

    def warm_up(self) -> None:
        """Load the model and score one pair."""
        model = self._ensure_loaded()
        if model is not None:
            list(model.rerank("warm up", ["warm up passage"]))


class NoopReranker:
    """Pass-through reranker used when ``RERANK_ENABLED=false``."""

    model_name = "none"
    available = False

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        """Return candidates sorted by their fused retrieval score."""
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def warm_up(self) -> None:
        """Nothing to load."""


def _passage_for(candidate: Candidate) -> str:
    """Give the cross-encoder the heading breadcrumb plus the passage text."""
    breadcrumb = " > ".join(candidate.chunk.section_path)
    text = candidate.chunk.text
    return f"{breadcrumb}: {text}" if breadcrumb else text


def build_reranker(settings: Any = None) -> CrossEncoderReranker | NoopReranker:
    """Return the reranker selected by configuration."""
    from emrag.settings import get_settings

    cfg = settings or get_settings()
    if not cfg.rerank_enabled:
        return NoopReranker()
    return CrossEncoderReranker(
        cfg.rerank_model,
        cache_dir=str(cfg.fastembed_cache_path) if cfg.fastembed_cache_path else None,
    )
