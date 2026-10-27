"""Full ingestion pipeline.

Runs identically in Colab (where Docling, OCR and a GPU are available) and as a
backend module. One call ingests one document: triage, parse, caption figures,
chunk, embed, write artifacts, upsert vectors, publish the manifest.

Resume semantics: the unit of checkpointing is the document. If the pipeline is
interrupted, re-running it re-does the current document and skips documents whose
manifest already says ``ready``. Because identifiers are content-derived, upserts
are idempotent, so a repeat never duplicates vectors.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from emrag import INGEST_VERSION
from emrag.core.errors import IngestFailed
from emrag.core.ids import doc_id_for_bytes
from emrag.core.types import (
    Chunk,
    Classification,
    DenseSparse,
    DocumentRecord,
    Element,
    PageInfo,
    PipelineProfile,
    utcnow,
)
from emrag.embeddings.base import EmbeddingProvider
from emrag.ingestion import quality
from emrag.ingestion.artifacts import ArtifactWriter
from emrag.ingestion.chunking import (
    ChunkingConfig,
    DocumentContext,
    chunk_document,
    embedding_text,
)
from emrag.ingestion.figures import FigureDescription, describe_figures, filter_figures
from emrag.ingestion.metadata import validate_chunks
from emrag.ingestion.parsers.pymupdf_parser import ParseResult
from emrag.ingestion.tables import to_structured
from emrag.ingestion.triage import triage_pdf
from emrag.llm.router import LLMRouter
from emrag.observability.logging import get_logger
from emrag.security.injection import scan_text
from emrag.security.pii import PiiMode, apply_pii_policy
from emrag.storage.registry import Registry
from emrag.vectorstore.base import VectorStore

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 32


@dataclass
class IngestRequest:
    """Everything needed to ingest one document."""

    tenant_id: str
    filename: str
    data: bytes
    doc_id: str = ""
    title: str | None = None
    acl_groups: list[str] = field(default_factory=lambda: ["all"])
    classification: Classification = "internal"
    tags: list[str] = field(default_factory=list)
    pii_mode: PiiMode = "allow"
    job_id: str | None = None
    profile: PipelineProfile = "full"

    def resolved_doc_id(self) -> str:
        """Content-addressed document id."""
        return self.doc_id or doc_id_for_bytes(self.data)


@dataclass
class IngestResult:
    """Outcome of one ingestion run."""

    record: DocumentRecord
    chunks: int = 0
    parents: int = 0
    upserted: int = 0
    duration_s: float = 0.0
    degraded: bool = False


class IngestionPipeline:
    """Orchestrates the full ingestion of a single document."""

    def __init__(
        self,
        *,
        registry: Registry,
        store: VectorStore,
        embedder: EmbeddingProvider,
        router: LLMRouter | None = None,
        chunking: ChunkingConfig | None = None,
        tuning: dict[str, Any] | None = None,
        use_docling: bool = True,
    ) -> None:
        self._registry = registry
        self._store = store
        self._embedder = embedder
        self._router = router
        self._chunking = chunking or ChunkingConfig()
        self._tuning = tuning or {}
        self._use_docling = use_docling

    async def run(self, request: IngestRequest) -> IngestResult:
        """Ingest one document end to end."""
        started = time.perf_counter()
        doc_id = request.resolved_doc_id()
        tenant_id = request.tenant_id

        record = DocumentRecord(
            doc_id=doc_id,
            tenant_id=tenant_id,
            filename=request.filename,
            title=request.title,
            size_bytes=len(request.data),
            page_count=0,
            status="processing",
            pipeline_profile=request.profile,
            ingest_version=INGEST_VERSION,
            acl_groups=list(request.acl_groups),
            classification=request.classification,
            tags=list(request.tags),
        )
        await self._registry.put_document(record)
        if request.job_id:
            await self._registry.update_job(request.job_id, status="processing")

        writer = ArtifactWriter(self._registry.store, tenant_id, doc_id)
        try:
            pages, doc_class = await asyncio.to_thread(
                triage_pdf,
                request.data,
                min_chars_native=int(self._ocr_tuning("min_chars_native", 50)),
                image_coverage_scanned=float(self._ocr_tuning("image_coverage_scanned", 0.6)),
            )
            record.page_count = len(pages)
            record.doc_class = doc_class
            await self._registry.put_document(record)

            parsed = await self._parse(request.data, pages, doc_class)
            descriptions = await self._describe_figures(parsed, len(pages))

            context = DocumentContext(
                doc_id=doc_id,
                tenant_id=tenant_id,
                filename=request.filename,
                page_count=max(1, len(pages)),
                title=request.title or _infer_title(parsed.elements),
                source_uri=self._registry.store.uri(f"{writer.prefix}/source.pdf"),
                acl_groups=list(request.acl_groups),
                classification=request.classification,
                tags=list(request.tags),
                pipeline_profile=request.profile,
                embed_model=self._embedder.model_name,
                sparse_model=self._embedder.sparse_model_name,
                ingest_version=INGEST_VERSION,
                ingested_at=utcnow(),
            )
            chunked = chunk_document(
                parsed.elements,
                parsed.pages or pages,
                context,
                config=self._chunking,
                figure_descriptions=descriptions,
            )
            chunks = validate_chunks(chunked.chunks)
            chunks = _apply_content_policies(chunks, request.pii_mode)
            if not chunks:
                raise IngestFailed(f"document {doc_id} produced no chunks")

            vectors = await self._embed(chunks)

            report = quality.evaluate(
                chunks, parsed.pages or pages, parser_degraded_reasons=parsed.degraded_reasons
            )
            # A figure the vision model never described is indexed on its
            # caption alone; the document still works but should say so, so a
            # re-ingest can be scheduled once the vision tier has quota again.
            uncaptioned = sum(1 for d in descriptions if d.fallback and not d.decorative)
            if uncaptioned:
                report.degraded_reasons.append(
                    f"{uncaptioned} figure(s) indexed without a vision description"
                )
                report.passed = False

            # Replace, do not merge. Upserts are idempotent for identical output,
            # but a re-ingest that produces fewer chunks (a changed chunker, or
            # figures now dropped) would otherwise leave stale points behind,
            # still retrievable and still citable.
            stale = await self._store.delete_by_doc(tenant_id, doc_id)
            if stale:
                logger.info("ingest.replaced_points", doc_id=doc_id, removed=stale)
            upsert_report = await self._store.upsert(chunks, vectors)
            report.degraded_reasons.extend(
                quality.verify_upsert(len(chunks), upsert_report.upserted)
            )
            report.passed = report.passed and not quality.verify_upsert(
                len(chunks), upsert_report.upserted
            )

            record.chunk_count = len(chunks)
            record.parent_count = len(chunked.parents)
            record.table_count = chunked.table_count
            record.figure_count = chunked.figure_count
            record.quality_report = report
            record.status = "ready" if report.passed else "ready_degraded"
            record.artifacts_prefix = writer.prefix
            record.title = context.title

            await self._write_artifacts(writer, request, record, parsed, chunks, vectors, chunked)
            await self._registry.put_document(record)
            if request.job_id:
                await self._registry.update_job(
                    request.job_id,
                    status="ready" if report.passed else "ready_degraded",
                    pages_done=len(pages),
                )

            duration = time.perf_counter() - started
            logger.info(
                "ingest.completed",
                doc_id=doc_id,
                pages=len(pages),
                chunks=len(chunks),
                upserted=upsert_report.upserted,
                degraded=not report.passed,
                duration_s=round(duration, 2),
            )
            return IngestResult(
                record=record,
                chunks=len(chunks),
                parents=len(chunked.parents),
                upserted=upsert_report.upserted,
                duration_s=duration,
                degraded=not report.passed,
            )

        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"[:2000]
            await self._registry.put_document(record)
            if request.job_id:
                await self._registry.update_job(request.job_id, status="failed", error=record.error)
            logger.error("ingest.failed", doc_id=doc_id, error=record.error)
            raise

    # ---- stages -------------------------------------------------------------
    async def _parse(self, data: bytes, pages: list[PageInfo], doc_class: str) -> ParseResult:
        """Run the parser chain: Docling first, PyMuPDF as the fallback."""
        needs_ocr = doc_class in ("scanned", "mixed")

        if self._use_docling:
            from emrag.ingestion.parsers import docling_parser

            if docling_parser.docling_available():
                try:
                    return await asyncio.to_thread(
                        docling_parser.parse_pdf,
                        data,
                        pages=pages,
                        needs_ocr=needs_ocr,
                        render_pages=True,
                    )
                except Exception as exc:  # noqa: BLE001 - fall through to PyMuPDF
                    logger.warning("ingest.docling_failed", error=str(exc)[:240])

        from emrag.ingestion.parsers import pymupdf_parser

        result = await asyncio.to_thread(pymupdf_parser.parse_pdf, data, pages=pages)
        result.degraded_reasons.append("parsed without layout model (no table structure)")

        if needs_ocr:
            from emrag.ingestion.parsers import ocr

            scanned = [p.page for p in pages if p.scanned]
            order = max((e.order for e in result.elements), default=0)
            for page_number in scanned:
                elements = await asyncio.to_thread(
                    ocr.ocr_page, data, page_number, order_start=order
                )
                order += len(elements)
                result.elements.extend(elements)
            if not ocr.ocr_available():
                result.degraded_reasons.append("scanned pages not OCR'd (engine unavailable)")
            result.elements.sort(key=lambda e: (e.page, e.order))
        return result

    async def _describe_figures(
        self, parsed: ParseResult, page_count: int
    ) -> list[FigureDescription]:
        """Filter and caption figures, skipping entirely when no vision tier exists."""
        figure_cfg = self._tuning.get("figures", {}) if self._tuning else {}
        survivors = filter_figures(
            parsed.elements,
            parsed.images,
            min_pixels=int(figure_cfg.get("min_pixels", 14_400)),
            max_aspect_ratio=float(figure_cfg.get("max_aspect_ratio", 8.0)),
            page_count=page_count,
            max_per_document=int(figure_cfg.get("max_per_document", 400)),
        )
        dropped = {
            e.image_bytes_ref
            for e in parsed.elements
            if e.kind == "figure" and e not in survivors and e.image_bytes_ref
        }
        for key in dropped:
            parsed.images.pop(key, None)
        parsed.elements = [e for e in parsed.elements if e.kind != "figure" or e in survivors]
        if not survivors:
            return []
        if self._router is None or not self._router.has_tier("vision"):
            descriptions = placeholder_descriptions(survivors)
            skipped = sum(1 for d in descriptions if d.decorative)
            logger.warning(
                "ingest.vision_unavailable",
                figures=len(survivors),
                indexed=len(survivors) - skipped,
                skipped_uncaptioned=skipped,
            )
            return descriptions
        return await describe_figures(
            self._router,
            survivors,
            parsed.images,
            context_by_key=_nearby_text(parsed.elements, survivors),
            batch_size=int(figure_cfg.get("batch_size", 3)),
        )

    async def _embed(self, chunks: list[Chunk]) -> list[DenseSparse]:
        """Embed chunk text in batches on a worker thread."""
        texts = [embedding_text(c) for c in chunks]
        vectors: list[DenseSparse] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors.extend(await asyncio.to_thread(self._embedder.embed_documents, batch))
        if len(vectors) != len(chunks):
            raise IngestFailed(
                f"embedding returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        return vectors

    async def _write_artifacts(
        self,
        writer: ArtifactWriter,
        request: IngestRequest,
        record: DocumentRecord,
        parsed: ParseResult,
        chunks: list[Chunk],
        vectors: list[DenseSparse],
        chunked: Any,
    ) -> None:
        """Persist every artifact for this document."""
        await writer.write_source(request.data)
        await writer.write_chunks(chunks, vectors)
        await writer.write_parents(chunked.parents)
        await writer.write_elements(_elements_payload(parsed.elements))
        await writer.write_quality(record.quality_report)
        await writer.write_docstore(
            chunked.parents,
            parsed.pages,
            {
                "doc_id": record.doc_id,
                "tenant_id": record.tenant_id,
                "ingest_version": INGEST_VERSION,
                "parser": parsed.engine,
            },
        )
        await writer.write_page_renders(parsed.page_renders)
        await writer.write_figures(parsed.images)
        await writer.write_document(record)

    def _ocr_tuning(self, key: str, default: Any) -> Any:
        return (self._tuning.get("ocr", {}) if self._tuning else {}).get(key, default)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _apply_content_policies(chunks: list[Chunk], pii_mode: PiiMode) -> list[Chunk]:
    """Run the injection scan and PII policy over every chunk before indexing."""
    kept: list[Chunk] = []
    for chunk in chunks:
        flags = scan_text(chunk.text)
        retained, pii_flags = apply_pii_policy(chunk, pii_mode)
        if retained is None:
            continue
        retained.risk_flags = sorted({*retained.risk_flags, *flags, *pii_flags})
        kept.append(retained)
    return kept


def _elements_payload(elements: list[Element]) -> list[dict[str, Any]]:
    """Serialise elements, expanding tables into their structured form."""
    out: list[dict[str, Any]] = []
    for element in elements:
        payload = element.model_dump(mode="json")
        if element.kind == "table" and element.table_cells:
            payload["table_json"] = to_structured(element.table_cells, element.table_caption)
        out.append(payload)
    return out


def _nearby_text(elements: list[Element], figures: list[Element]) -> dict[str, str]:
    """Collect prose immediately before and after each figure, as caption context."""
    out: dict[str, str] = {}
    by_order = sorted(elements, key=lambda e: (e.page, e.order))
    index = {id(e): i for i, e in enumerate(by_order)}
    for figure in figures:
        position = index.get(id(figure))
        if position is None or not figure.image_bytes_ref:
            continue
        window = by_order[max(0, position - 2) : position + 3]
        text = " ".join(
            e.text.strip()
            for e in window
            if e.kind in ("paragraph", "caption", "heading") and e.text.strip()
        )
        out[figure.image_bytes_ref] = text[:600]
    return out


def placeholder_descriptions(figures: list[Element]) -> list[FigureDescription]:
    """Describe figures when no vision model is available.

    A figure with a caption is worth indexing on the caption alone. One without
    is marked decorative so the chunker drops it: a chunk reading "Figure on
    page 1" carries no information, competes with real passages in retrieval,
    and could be cited. The crop stays in the artifacts, so a later full ingest
    with a vision model can describe it properly.
    """
    out: list[FigureDescription] = []
    for element in figures:
        caption = (element.text or element.table_caption or "").strip()
        out.append(
            FigureDescription(
                image_key=element.image_bytes_ref or "",
                page=element.page,
                bbox=list(element.bbox),
                description=caption or f"Figure on page {element.page}.",
                decorative=not caption,
            )
        )
    return out


def _infer_title(elements: list[Element]) -> str | None:
    """Use the first heading on the first page as the document title."""
    for element in elements:
        if element.kind == "heading" and element.text.strip():
            return element.text.strip()[:512]
    return None


def inbox_extension(filename: str) -> str:
    """Return a lowercase extension for inbox storage, defaulting to pdf."""
    suffix = PurePosixPath(filename).suffix.lower().lstrip(".")
    return suffix or "pdf"
