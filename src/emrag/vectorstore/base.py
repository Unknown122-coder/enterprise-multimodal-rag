"""Vector store protocol.

One interface, two implementations (Qdrant, Pinecone). Nothing above this layer
may import a vendor client, which is what makes the migration path a config flip.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator

from emrag.core.types import (
    Candidate,
    Chunk,
    DenseSparse,
    QueryVectors,
    RetrievalFilter,
    StrictModel,
)


class UpsertReport(StrictModel):
    """Outcome of an upsert batch."""

    upserted: int = 0
    skipped: int = 0
    batches: int = 0


class StoreHealth(StrictModel):
    """Vector store reachability and size."""

    reachable: bool
    backend: str
    collection: str
    points: int | None = None
    detail: str | None = None


class PointRecord(StrictModel):
    """An exported point, used by the migration script."""

    point_id: str
    chunk: Chunk
    vectors: DenseSparse


class VectorStore(abc.ABC):
    """Async vector store interface."""

    backend: str

    @abc.abstractmethod
    async def ensure_schema(self, dim: int, *, sparse: bool = True) -> None:
        """Create the collection/index and payload indexes if absent. Idempotent."""

    @abc.abstractmethod
    async def upsert(
        self, chunks: list[Chunk], vectors: list[DenseSparse], *, batch_size: int = 128
    ) -> UpsertReport:
        """Insert or replace points. Deterministic ids make this idempotent."""

    @abc.abstractmethod
    async def hybrid_search(
        self,
        query: QueryVectors,
        filters: RetrievalFilter,
        *,
        k_dense: int = 40,
        k_sparse: int = 40,
        k_out: int = 40,
    ) -> list[Candidate]:
        """Run dense and sparse retrieval and return fused candidates."""

    @abc.abstractmethod
    async def get_chunks(self, tenant_id: str, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch specific chunks by id, scoped to a tenant."""

    @abc.abstractmethod
    async def list_chunks(
        self, tenant_id: str, doc_id: str, *, limit: int = 100, offset: str | None = None
    ) -> tuple[list[Chunk], str | None]:
        """Page through a document's chunks, returning the next cursor."""

    @abc.abstractmethod
    async def delete_by_doc(self, tenant_id: str, doc_id: str) -> int:
        """Delete every point of a document. Returns the number requested."""

    @abc.abstractmethod
    async def count(self, tenant_id: str | None = None, doc_id: str | None = None) -> int:
        """Count points, optionally scoped."""

    @abc.abstractmethod
    async def health(self) -> StoreHealth:
        """Probe the store."""

    @abc.abstractmethod
    def export(self, tenant_id: str | None = None) -> Iterator[PointRecord]:
        """Stream every point for migration. Blocking iterator."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release connections."""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], *, k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """Fuse ranked id lists with RRF.

    Used by the Pinecone adapter, which has no server-side fusion, and by query
    rewriting where several variants are retrieved independently.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, identifier in enumerate(ranked, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + rank)
    return scores
