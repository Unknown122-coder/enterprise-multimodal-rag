"""Embedding provider protocol."""

from __future__ import annotations

import abc

from emrag.core.types import DenseSparse, QueryVectors


class EmbeddingProvider(abc.ABC):
    """Produces dense and BM25-sparse vectors for documents and queries.

    The same provider must be used for ingestion and for querying; the model name
    is recorded on every chunk so a mismatch is detectable.
    """

    model_name: str
    sparse_model_name: str
    dim: int

    @abc.abstractmethod
    def embed_documents(self, texts: list[str]) -> list[DenseSparse]:
        """Embed passages. Blocking and CPU bound; call via a threadpool in async code."""

    @abc.abstractmethod
    def embed_query(self, text: str) -> QueryVectors:
        """Embed a single query string."""

    def warm_up(self) -> None:
        """Run one tiny embedding so the first real request is not slowed by lazy init."""
        self.embed_query("warm up")
