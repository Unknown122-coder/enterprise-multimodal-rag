"""Qdrant adapter (default backend).

Single collection, tenant in the payload with ``is_tenant=True`` so each tenant's
points are co-located. Dense and BM25-sparse vectors live on the same point and
are fused server-side with RRF via the Query API, so one round trip returns hybrid
results.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from emrag.core.errors import StoreUnavailable
from emrag.core.ids import point_id
from emrag.core.types import Candidate, Chunk, DenseSparse, QueryVectors, RetrievalFilter
from emrag.observability.logging import get_logger
from emrag.vectorstore.base import PointRecord, StoreHealth, UpsertReport, VectorStore

logger = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

_KEYWORD_INDEXES = (
    "doc_id",
    "modality",
    "acl_groups",
    "classification",
    "tags",
    "section_path",
    "ingest_version",
)


class QdrantStore(VectorStore):
    """Vector store backed by Qdrant Cloud or a local Qdrant container."""

    backend = "qdrant"

    def __init__(
        self,
        url: str,
        collection: str,
        *,
        api_key: str | None = None,
        timeout: int = 30,
        client: Any = None,
    ) -> None:
        self._collection = collection
        self._url = url
        if client is not None:
            self._client = client
        elif url == ":memory:":
            # In-process Qdrant: used by integration tests so the full
            # ingest-to-query path runs without a container.
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(location=":memory:")
        elif url.startswith("local://"):
            # On-disk local Qdrant, for development and offline evaluation
            # without a container. One process at a time: the directory is
            # locked, so the API and the ingestion CLI cannot share it.
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(path=url[len("local://") :])
        else:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout)

    # ---- schema -------------------------------------------------------------
    async def ensure_schema(self, dim: int, *, sparse: bool = True) -> None:
        """Create the collection and payload indexes when missing."""
        from qdrant_client import models as qm

        try:
            exists = await self._client.collection_exists(self._collection)
        except Exception as exc:
            raise StoreUnavailable(f"qdrant unreachable: {exc}") from exc

        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
                },
                sparse_vectors_config=(
                    {SPARSE_VECTOR: qm.SparseVectorParams(modifier=qm.Modifier.IDF)}
                    if sparse
                    else None
                ),
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128, payload_m=16),
                optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=10_000),
                on_disk_payload=True,
            )
            logger.info("vectorstore.collection_created", collection=self._collection, dim=dim)

        await self._create_payload_indexes()

    async def _create_payload_indexes(self) -> None:
        from qdrant_client import models as qm

        async def _safe(field: str, schema: Any, **kwargs: Any) -> None:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - already-exists is not an error
                logger.debug("vectorstore.index_skipped", field=field, reason=str(exc)[:120])

        await _safe(
            "tenant_id",
            qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD, is_tenant=True),
        )
        await _safe("page", qm.PayloadSchemaType.INTEGER)
        for field in _KEYWORD_INDEXES:
            await _safe(field, qm.PayloadSchemaType.KEYWORD)

    # ---- writes -------------------------------------------------------------
    async def upsert(
        self, chunks: list[Chunk], vectors: list[DenseSparse], *, batch_size: int = 128
    ) -> UpsertReport:
        """Upsert chunks with deterministic point ids."""
        from qdrant_client import models as qm

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        report = UpsertReport()
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = []
            for chunk, vec in zip(batch_chunks, batch_vectors, strict=True):
                vector_payload: dict[str, Any] = {DENSE_VECTOR: vec.dense}
                if vec.sparse_indices:
                    vector_payload[SPARSE_VECTOR] = qm.SparseVector(
                        indices=vec.sparse_indices, values=vec.sparse_values
                    )
                points.append(
                    qm.PointStruct(
                        id=point_id(chunk.chunk_id),
                        vector=vector_payload,
                        payload=_to_payload(chunk),
                    )
                )
            try:
                await self._client.upsert(
                    collection_name=self._collection, points=points, wait=True
                )
            except Exception as exc:
                raise StoreUnavailable(f"qdrant upsert failed: {exc}") from exc
            report.upserted += len(points)
            report.batches += 1
        return report

    async def delete_by_doc(self, tenant_id: str, doc_id: str) -> int:
        """Delete all points of one document within one tenant."""
        from qdrant_client import models as qm

        before = await self.count(tenant_id=tenant_id, doc_id=doc_id)
        conditions: list[Any] = [
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        ]
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=conditions)),
                wait=True,
            )
        except Exception as exc:
            raise StoreUnavailable(f"qdrant delete failed: {exc}") from exc
        return before

    # ---- reads --------------------------------------------------------------
    async def hybrid_search(
        self,
        query: QueryVectors,
        filters: RetrievalFilter,
        *,
        k_dense: int = 40,
        k_sparse: int = 40,
        k_out: int = 40,
    ) -> list[Candidate]:
        """Prefetch dense and sparse, fuse with RRF server-side, return candidates."""
        from qdrant_client import models as qm

        qfilter = build_qdrant_filter(filters)
        prefetch = [
            qm.Prefetch(query=query.dense, using=DENSE_VECTOR, limit=k_dense, filter=qfilter)
        ]
        if query.sparse_indices:
            prefetch.append(
                qm.Prefetch(
                    query=qm.SparseVector(indices=query.sparse_indices, values=query.sparse_values),
                    using=SPARSE_VECTOR,
                    limit=k_sparse,
                    filter=qfilter,
                )
            )
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                prefetch=prefetch,
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=k_out,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise StoreUnavailable(f"qdrant search failed: {exc}") from exc

        candidates: list[Candidate] = []
        for rank, scored in enumerate(response.points, start=1):
            chunk = _from_payload(scored.payload or {})
            if chunk is None:
                continue
            candidates.append(Candidate(chunk=chunk, score=float(scored.score), dense_rank=rank))
        return candidates

    async def get_chunks(self, tenant_id: str, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch chunks by id, discarding anything outside the tenant."""
        if not chunk_ids:
            return []
        try:
            records = await self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id(cid) for cid in chunk_ids],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise StoreUnavailable(f"qdrant retrieve failed: {exc}") from exc
        out: list[Chunk] = []
        for record in records:
            chunk = _from_payload(record.payload or {})
            if chunk is not None and chunk.tenant_id == tenant_id:
                out.append(chunk)
        return out

    async def list_chunks(
        self, tenant_id: str, doc_id: str, *, limit: int = 100, offset: str | None = None
    ) -> tuple[list[Chunk], str | None]:
        """Scroll a document's chunks in id order."""
        from qdrant_client import models as qm

        try:
            records, next_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
                        qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                    ]
                ),
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise StoreUnavailable(f"qdrant scroll failed: {exc}") from exc
        chunks = [c for c in (_from_payload(r.payload or {}) for r in records) if c is not None]
        chunks.sort(key=lambda c: c.chunk_id)
        return chunks, str(next_offset) if next_offset else None

    async def count(self, tenant_id: str | None = None, doc_id: str | None = None) -> int:
        """Count points, optionally scoped to a tenant and document."""
        from qdrant_client import models as qm

        must: list[Any] = []
        if tenant_id:
            must.append(qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)))
        if doc_id:
            must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
        try:
            result = await self._client.count(
                collection_name=self._collection,
                count_filter=qm.Filter(must=must) if must else None,
                exact=True,
            )
        except Exception as exc:
            raise StoreUnavailable(f"qdrant count failed: {exc}") from exc
        return int(result.count)

    async def health(self) -> StoreHealth:
        """Probe the collection, never raising."""
        try:
            exists = await self._client.collection_exists(self._collection)
            points = await self.count() if exists else None
            return StoreHealth(
                reachable=True,
                backend=self.backend,
                collection=self._collection,
                points=points,
                detail=None if exists else "collection missing",
            )
        except Exception as exc:  # noqa: BLE001
            return StoreHealth(
                reachable=False,
                backend=self.backend,
                collection=self._collection,
                detail=str(exc)[:200],
            )

    def export(self, tenant_id: str | None = None) -> Iterator[PointRecord]:
        """Stream points with vectors, for migration between backends."""
        from qdrant_client import QdrantClient
        from qdrant_client import models as qm

        sync_client = getattr(self._client, "_sync_twin", None)
        if sync_client is None:
            init = getattr(self._client, "init_options", {})
            sync_client = QdrantClient(url=init.get("url"), api_key=init.get("api_key"), timeout=60)
        flt = (
            qm.Filter(
                must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
            )
            if tenant_id
            else None
        )
        offset: Any = None
        while True:
            records, offset = sync_client.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                chunk = _from_payload(record.payload or {})
                if chunk is None:
                    continue
                vectors = record.vector or {}
                dense = list(vectors.get(DENSE_VECTOR) or []) if isinstance(vectors, dict) else []
                sparse = vectors.get(SPARSE_VECTOR) if isinstance(vectors, dict) else None
                yield PointRecord(
                    point_id=str(record.id),
                    chunk=chunk,
                    vectors=DenseSparse(
                        dense=[float(x) for x in dense],
                        sparse_indices=[int(i) for i in getattr(sparse, "indices", [])],
                        sparse_values=[float(v) for v in getattr(sparse, "values", [])],
                    ),
                )
            if offset is None:
                break

    async def aclose(self) -> None:
        """Close the client connection."""
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if asyncio.iscoroutine(result):
            await result


def build_qdrant_filter(filters: RetrievalFilter) -> Any:
    """Compile a :class:`RetrievalFilter` into a Qdrant filter.

    The tenant clause is unconditional. ``tests/unit/test_filters.py`` asserts this,
    because dropping it would silently break tenant isolation.
    """
    from qdrant_client import models as qm

    must: list[Any] = [
        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=filters.tenant_id)),
        qm.FieldCondition(key="acl_groups", match=qm.MatchAny(any=list(filters.acl_groups))),
        qm.FieldCondition(
            key="classification", match=qm.MatchAny(any=filters.allowed_classifications())
        ),
    ]
    if filters.doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(filters.doc_ids))))
    if filters.modalities:
        must.append(
            qm.FieldCondition(
                key="modality", match=qm.MatchAny(any=[str(m) for m in filters.modalities])
            )
        )
    if filters.tags:
        must.append(qm.FieldCondition(key="tags", match=qm.MatchAny(any=list(filters.tags))))
    if filters.page_range:
        low, high = filters.page_range
        must.append(qm.FieldCondition(key="page", range=qm.Range(gte=low, lte=high)))
    if filters.section_contains:
        must.append(
            qm.FieldCondition(key="section_path", match=qm.MatchText(text=filters.section_contains))
        )
    return qm.Filter(must=must)


def _to_payload(chunk: Chunk) -> dict[str, Any]:
    """Serialise a chunk for the Qdrant payload (datetimes become ISO strings)."""
    return chunk.model_dump(mode="json")


def _from_payload(payload: dict[str, Any]) -> Chunk | None:
    """Rebuild a chunk from a payload, skipping anything that no longer validates."""
    if not payload:
        return None
    try:
        return Chunk.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - a stale point must not break a query
        logger.warning("vectorstore.payload_invalid", error=str(exc)[:200])
        return None
