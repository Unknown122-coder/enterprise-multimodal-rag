"""Vector store migration.

Migration never re-parses or re-embeds: the chunk artifacts already hold the
vectors, so moving between backends is a replay plus a verification pass. See
``docs/adr/0004-artifacts-as-truth.md``.

The reusable logic lives here; ``scripts/migrate_store.py`` is a thin CLI over
it, and the same functions back ``scripts/upsert_artifacts.py``.

The part that matters is :func:`compare_recall`. Replaying points is mechanical
and its failure is loud. A filter-syntax difference between backends is silent,
and shows up only as worse retrieval, so a migration is not finished until the
two backends have been measured on the same questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emrag.core.types import Principal
from emrag.ingestion.artifacts import ArtifactReader
from emrag.observability.logging import get_logger
from emrag.storage.artifact_store import ArtifactStore
from emrag.vectorstore.base import VectorStore

logger = get_logger(__name__)


@dataclass
class DocumentReplay:
    """Outcome of replaying one document."""

    tenant_id: str
    doc_id: str
    chunks: int = 0
    upserted: int = 0
    counted: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when every chunk was upserted and, if checked, counted."""
        if self.error is not None:
            return False
        if self.upserted != self.chunks:
            return False
        return self.counted is None or self.counted == self.chunks


@dataclass
class ReplayReport:
    """Aggregate outcome of a replay."""

    documents: list[DocumentReplay] = field(default_factory=list)

    @property
    def chunks(self) -> int:
        """Total chunks replayed."""
        return sum(d.chunks for d in self.documents)

    @property
    def upserted(self) -> int:
        """Total points written."""
        return sum(d.upserted for d in self.documents)

    @property
    def failures(self) -> list[DocumentReplay]:
        """Documents whose replay did not verify."""
        return [d for d in self.documents if not d.ok]

    @property
    def ok(self) -> bool:
        """True when every document verified."""
        return bool(self.documents) and not self.failures


async def replay_artifacts(
    artifacts: ArtifactStore,
    target: VectorStore,
    *,
    tenant_id: str | None = None,
    doc_id: str | None = None,
    batch_size: int = 128,
    verify: bool = True,
) -> ReplayReport:
    """Replay stored chunks and vectors into ``target``.

    Idempotent: point identifiers are derived from chunk identifiers, so running
    this twice leaves the store in the same state.
    """
    reader = ArtifactReader(artifacts)
    pairs = await reader.list_documents(tenant_id)
    if doc_id:
        pairs = [(t, d) for t, d in pairs if d == doc_id]

    report = ReplayReport()
    for tenant, document in pairs:
        entry = DocumentReplay(tenant_id=tenant, doc_id=document)
        try:
            chunks, vectors = await reader.read_chunks(tenant, document)
        except Exception as exc:  # noqa: BLE001 - continue through the corpus
            entry.error = f"read failed: {str(exc)[:160]}"
            report.documents.append(entry)
            continue

        if not chunks:
            entry.error = "artifact contained no chunks"
            report.documents.append(entry)
            continue

        entry.chunks = len(chunks)
        try:
            result = await target.upsert(chunks, vectors, batch_size=batch_size)
            entry.upserted = result.upserted
            if verify:
                entry.counted = await target.count(tenant_id=tenant, doc_id=document)
        except Exception as exc:  # noqa: BLE001
            entry.error = f"upsert failed: {str(exc)[:160]}"
        report.documents.append(entry)

    logger.info(
        "vectorstore.replayed",
        documents=len(report.documents),
        chunks=report.chunks,
        upserted=report.upserted,
        failures=len(report.failures),
    )
    return report


async def measure_recall(
    store: VectorStore, settings: object, golden_path: str, *, top_k: int = 8
) -> float:
    """Measure ``recall@top_k`` for one backend using a golden set.

    Reranking is deliberately off: the question is whether the backend's filters
    and fusion surface the right passage, not how well a cross encoder reorders
    it afterwards.
    """
    from emrag.embeddings.factory import build_embedder
    from emrag.eval.metrics import RetrievalHit, load_golden, recall_at_k
    from emrag.retrieval import filters as filter_builder
    from emrag.retrieval.rerank import NoopReranker
    from emrag.retrieval.retriever import RetrievalOptions, Retriever

    retriever = Retriever(
        store=store,
        embedder=build_embedder(settings),  # type: ignore[arg-type]
        reranker=NoopReranker(),
    )
    items = [i for i in load_golden(golden_path) if i.answerable]
    if not items:
        return 0.0

    total = 0.0
    for item in items:
        principal = Principal(
            subject="migrate",
            tenant_id=item.tenant_id,
            groups=["all"],
            clearance="confidential",
            role="admin",
        )
        outcome = await retriever.retrieve(
            item.question,
            filter_builder.from_request(principal),
            RetrievalOptions(
                top_k=top_k, rewrite="off", rerank=False, parent_expansion=False, gate_tau=0.0
            ),
        )
        hits = [
            RetrievalHit(chunk_id=c.chunk.chunk_id, doc_id=c.chunk.doc_id, page=c.chunk.page)
            for c in outcome.candidates
        ]
        total += recall_at_k(hits, item, top_k)
    return total / len(items)


@dataclass
class RecallComparison:
    """Recall on two backends and the difference between them."""

    source: float | None
    target: float
    max_delta: float

    @property
    def delta(self) -> float | None:
        """How much recall the target loses relative to the source."""
        return None if self.source is None else self.source - self.target

    @property
    def ok(self) -> bool:
        """True when the target is no more than ``max_delta`` worse."""
        return self.delta is None or self.delta <= self.max_delta


async def compare_recall(
    target: VectorStore,
    settings: object,
    golden_path: str,
    *,
    source: VectorStore | None = None,
    top_k: int = 8,
    max_delta: float = 0.02,
) -> RecallComparison:
    """Compare recall between two backends on the same questions."""
    target_recall = await measure_recall(target, settings, golden_path, top_k=top_k)
    source_recall = (
        await measure_recall(source, settings, golden_path, top_k=top_k)
        if source is not None
        else None
    )
    comparison = RecallComparison(source=source_recall, target=target_recall, max_delta=max_delta)
    logger.info(
        "vectorstore.recall_compared",
        target=round(target_recall, 4),
        source=round(source_recall, 4) if source_recall is not None else None,
        ok=comparison.ok,
    )
    return comparison
