"""Pinecone adapter (alternative backend).

Differences from Qdrant that this adapter hides:

* tenancy is a **namespace**, not a payload filter;
* dense and sparse vectors live in **two separate indexes**, so fusion is done
  client-side with RRF;
* metadata must be flat, so ``bbox`` is split into four floats and nested objects
  are dropped. :func:`flatten_metadata` / :func:`unflatten_metadata` are the only
  place that conversion happens.

The Pinecone SDK is synchronous, so every call is dispatched to a worker thread.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from emrag.core.errors import StoreUnavailable
from emrag.core.types import Candidate, Chunk, DenseSparse, QueryVectors, RetrievalFilter
from emrag.observability.logging import get_logger
from emrag.vectorstore.base import (
    PointRecord,
    StoreHealth,
    UpsertReport,
    VectorStore,
    reciprocal_rank_fusion,
)

logger = get_logger(__name__)

_DROPPED_FIELDS = frozenset({"element_ids"})
_RRF_K = 60


class PineconeStore(VectorStore):
    """Vector store backed by Pinecone serverless (Starter plan compatible)."""

    backend = "pinecone"

    def __init__(
        self,
        api_key: str,
        dense_index: str,
        sparse_index: str | None = None,
        *,
        region: str = "us-east-1",
        cloud: str = "aws",
        client: Any = None,
    ) -> None:
        self._dense_name = dense_index
        self._sparse_name = sparse_index
        self._region = region
        self._cloud = cloud
        if client is not None:
            self._pc = client
        else:
            from pinecone import Pinecone

            self._pc = Pinecone(api_key=api_key)
        self._dense: Any = None
        self._sparse: Any = None

    async def ensure_schema(self, dim: int, *, sparse: bool = True) -> None:
        """Create the dense (and optional sparse) index if absent."""

        def _create() -> None:
            from pinecone import ServerlessSpec

            existing = {i["name"] for i in self._pc.list_indexes()}
            spec = ServerlessSpec(cloud=self._cloud, region=self._region)
            if self._dense_name not in existing:
                self._pc.create_index(
                    name=self._dense_name, dimension=dim, metric="cosine", spec=spec
                )
            if sparse and self._sparse_name and self._sparse_name not in existing:
                self._pc.create_index(
                    name=self._sparse_name,
                    metric="dotproduct",
                    spec=spec,
                    vector_type="sparse",
                )

        try:
            await asyncio.to_thread(_create)
            self._dense = await asyncio.to_thread(self._pc.Index, self._dense_name)
            if sparse and self._sparse_name:
                self._sparse = await asyncio.to_thread(self._pc.Index, self._sparse_name)
        except Exception as exc:
            raise StoreUnavailable(f"pinecone schema setup failed: {exc}") from exc

    async def upsert(
        self, chunks: list[Chunk], vectors: list[DenseSparse], *, batch_size: int = 128
    ) -> UpsertReport:
        """Upsert into both indexes, keyed by ``chunk_id``."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        await self._ensure_handles()
        report = UpsertReport()
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            namespace = batch_chunks[0].tenant_id
            dense_items = [
                {
                    "id": chunk.chunk_id,
                    "values": vec.dense,
                    "metadata": flatten_metadata(chunk),
                }
                for chunk, vec in zip(batch_chunks, batch_vectors, strict=True)
            ]
            try:
                await asyncio.to_thread(
                    self._dense.upsert, vectors=dense_items, namespace=namespace
                )
                if self._sparse is not None:
                    sparse_items = [
                        {
                            "id": chunk.chunk_id,
                            "sparse_values": {
                                "indices": vec.sparse_indices,
                                "values": vec.sparse_values,
                            },
                            "metadata": flatten_metadata(chunk),
                        }
                        for chunk, vec in zip(batch_chunks, batch_vectors, strict=True)
                        if vec.sparse_indices
                    ]
                    if sparse_items:
                        await asyncio.to_thread(
                            self._sparse.upsert, vectors=sparse_items, namespace=namespace
                        )
            except Exception as exc:
                raise StoreUnavailable(f"pinecone upsert failed: {exc}") from exc
            report.upserted += len(dense_items)
            report.batches += 1
        return report

    async def hybrid_search(
        self,
        query: QueryVectors,
        filters: RetrievalFilter,
        *,
        k_dense: int = 40,
        k_sparse: int = 40,
        k_out: int = 40,
    ) -> list[Candidate]:
        """Query both indexes and fuse the two rankings with RRF."""
        await self._ensure_handles()
        meta_filter = build_pinecone_filter(filters)
        namespace = filters.tenant_id

        def _dense_query() -> Any:
            return self._dense.query(
                vector=query.dense,
                top_k=k_dense,
                namespace=namespace,
                include_metadata=True,
                filter=meta_filter,
            )

        def _sparse_query() -> Any:
            return self._sparse.query(
                sparse_vector={"indices": query.sparse_indices, "values": query.sparse_values},
                top_k=k_sparse,
                namespace=namespace,
                include_metadata=True,
                filter=meta_filter,
            )

        try:
            tasks = [asyncio.to_thread(_dense_query)]
            if self._sparse is not None and query.sparse_indices:
                tasks.append(asyncio.to_thread(_sparse_query))
            results = await asyncio.gather(*tasks)
        except Exception as exc:
            raise StoreUnavailable(f"pinecone search failed: {exc}") from exc

        by_id: dict[str, Chunk] = {}
        ranked_lists: list[list[str]] = []
        for result in results:
            ranking: list[str] = []
            for match in result.get("matches", []) if isinstance(result, dict) else result.matches:
                match_id = match["id"] if isinstance(match, dict) else match.id
                metadata = match["metadata"] if isinstance(match, dict) else match.metadata
                chunk = unflatten_metadata(dict(metadata or {}))
                if chunk is None:
                    continue
                by_id[match_id] = chunk
                ranking.append(match_id)
            ranked_lists.append(ranking)

        fused = reciprocal_rank_fusion(ranked_lists, k=_RRF_K)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k_out]
        return [
            Candidate(chunk=by_id[cid], score=score, dense_rank=rank)
            for rank, (cid, score) in enumerate(ordered, start=1)
            if cid in by_id
        ]

    async def get_chunks(self, tenant_id: str, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch chunks by id from the tenant's namespace."""
        if not chunk_ids:
            return []
        await self._ensure_handles()
        try:
            result = await asyncio.to_thread(self._dense.fetch, ids=chunk_ids, namespace=tenant_id)
        except Exception as exc:
            raise StoreUnavailable(f"pinecone fetch failed: {exc}") from exc
        vectors = result["vectors"] if isinstance(result, dict) else result.vectors
        out: list[Chunk] = []
        for record in (vectors or {}).values():
            metadata = record["metadata"] if isinstance(record, dict) else record.metadata
            chunk = unflatten_metadata(dict(metadata or {}))
            if chunk is not None and chunk.tenant_id == tenant_id:
                out.append(chunk)
        return out

    async def list_chunks(
        self, tenant_id: str, doc_id: str, *, limit: int = 100, offset: str | None = None
    ) -> tuple[list[Chunk], str | None]:
        """Page through a document's chunk ids using the list API."""
        await self._ensure_handles()

        def _list() -> Any:
            return self._dense.list_paginated(
                prefix=f"{doc_id}:", namespace=tenant_id, limit=limit, pagination_token=offset
            )

        try:
            page = await asyncio.to_thread(_list)
        except Exception as exc:
            raise StoreUnavailable(f"pinecone list failed: {exc}") from exc
        ids = [v.id if hasattr(v, "id") else v["id"] for v in (page.vectors or [])]
        chunks = await self.get_chunks(tenant_id, ids)
        chunks.sort(key=lambda c: c.chunk_id)
        token = getattr(getattr(page, "pagination", None), "next", None)
        return chunks, token

    async def delete_by_doc(self, tenant_id: str, doc_id: str) -> int:
        """Delete a document's vectors from both indexes.

        Starter-plan serverless indexes do not support delete-by-metadata, so ids
        are listed first and deleted in batches.
        """
        await self._ensure_handles()
        deleted = 0
        token: str | None = None
        while True:
            chunks, token = await self.list_chunks(tenant_id, doc_id, limit=200, offset=token)
            ids = [c.chunk_id for c in chunks]
            if ids:
                try:
                    await asyncio.to_thread(self._dense.delete, ids=ids, namespace=tenant_id)
                    if self._sparse is not None:
                        await asyncio.to_thread(self._sparse.delete, ids=ids, namespace=tenant_id)
                except Exception as exc:
                    raise StoreUnavailable(f"pinecone delete failed: {exc}") from exc
                deleted += len(ids)
            if not token:
                break
        return deleted

    async def count(self, tenant_id: str | None = None, doc_id: str | None = None) -> int:
        """Count vectors using index statistics (namespace granularity only)."""
        await self._ensure_handles()
        try:
            stats = await asyncio.to_thread(self._dense.describe_index_stats)
        except Exception as exc:
            raise StoreUnavailable(f"pinecone stats failed: {exc}") from exc
        namespaces = stats["namespaces"] if isinstance(stats, dict) else stats.namespaces
        if tenant_id:
            entry = (namespaces or {}).get(tenant_id)
            if entry is None:
                return 0
            total = int(entry["vector_count"] if isinstance(entry, dict) else entry.vector_count)
            if doc_id is None:
                return total
            chunks, _ = await self.list_chunks(tenant_id, doc_id, limit=10_000)
            return len(chunks)
        return int(
            stats["total_vector_count"] if isinstance(stats, dict) else stats.total_vector_count
        )

    async def health(self) -> StoreHealth:
        """Probe the dense index, never raising."""
        try:
            points = await self.count()
            return StoreHealth(
                reachable=True,
                backend=self.backend,
                collection=self._dense_name,
                points=points,
            )
        except Exception as exc:  # noqa: BLE001
            return StoreHealth(
                reachable=False,
                backend=self.backend,
                collection=self._dense_name,
                detail=str(exc)[:200],
            )

    def export(self, tenant_id: str | None = None) -> Iterator[PointRecord]:
        """Not supported: artifacts are the migration source of truth."""
        raise NotImplementedError(
            "export from Pinecone is not implemented; rebuild from artifacts with "
            "scripts/upsert_artifacts.py instead"
        )

    async def aclose(self) -> None:
        """No persistent connection to close."""
        return None

    async def _ensure_handles(self) -> None:
        if self._dense is None:
            self._dense = await asyncio.to_thread(self._pc.Index, self._dense_name)
        if self._sparse is None and self._sparse_name:
            try:
                self._sparse = await asyncio.to_thread(self._pc.Index, self._sparse_name)
            except Exception as exc:  # noqa: BLE001 - sparse index is optional
                logger.warning("vectorstore.sparse_unavailable", error=str(exc)[:160])
                self._sparse_name = None


def build_pinecone_filter(filters: RetrievalFilter) -> dict[str, Any]:
    """Compile a :class:`RetrievalFilter` into Pinecone metadata filter syntax.

    The tenant is enforced by the namespace, but ``tenant_id`` is also asserted in
    metadata as defence in depth.
    """
    clauses: list[dict[str, Any]] = [
        {"tenant_id": {"$eq": filters.tenant_id}},
        {"acl_groups": {"$in": list(filters.acl_groups)}},
        {"classification": {"$in": filters.allowed_classifications()}},
    ]
    if filters.doc_ids:
        clauses.append({"doc_id": {"$in": list(filters.doc_ids)}})
    if filters.modalities:
        clauses.append({"modality": {"$in": list(filters.modalities)}})
    if filters.tags:
        clauses.append({"tags": {"$in": list(filters.tags)}})
    if filters.page_range:
        low, high = filters.page_range
        clauses.append({"page": {"$gte": low, "$lte": high}})
    return {"$and": clauses}


def flatten_metadata(chunk: Chunk) -> dict[str, Any]:
    """Convert a chunk into flat Pinecone metadata."""
    data = chunk.model_dump(mode="json")
    bbox = data.pop("bbox")
    for name, value in zip(("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"), bbox, strict=True):
        data[name] = float(value)
    for field in _DROPPED_FIELDS:
        data.pop(field, None)
    return {k: v for k, v in data.items() if v is not None}


def unflatten_metadata(metadata: dict[str, Any]) -> Chunk | None:
    """Rebuild a chunk from flat Pinecone metadata."""
    if not metadata:
        return None
    data = dict(metadata)
    try:
        data["bbox"] = [
            float(data.pop("bbox_x0")),
            float(data.pop("bbox_y0")),
            float(data.pop("bbox_x1")),
            float(data.pop("bbox_y1")),
        ]
    except KeyError:
        return None
    data.setdefault("element_ids", [])
    data["page"] = int(data.get("page", 1))
    data["page_count"] = int(data.get("page_count", 1))
    data["token_count"] = int(data.get("token_count", 1))
    try:
        return Chunk.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vectorstore.payload_invalid", backend="pinecone", error=str(exc)[:200])
        return None
