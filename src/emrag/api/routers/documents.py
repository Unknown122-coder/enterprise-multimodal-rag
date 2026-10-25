"""Document lifecycle endpoints: upload, list, detail, delete, pages, chunks."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, Query, UploadFile
from fastapi.responses import Response

from emrag.api.deps import PrincipalDep, StateDep, WriterDep
from emrag.api.schemas.common import Page
from emrag.api.schemas.documents import (
    ChunkPublic,
    DeleteAccepted,
    DocumentDetail,
    DocumentSummary,
    IngestAccepted,
    UploadMetadata,
)
from emrag.core.errors import Forbidden, NotFound, UnsupportedMedia, ValidationFailed
from emrag.core.ids import DOC_ID_RE, doc_id_for_bytes
from emrag.core.types import DocumentRecord, PipelineProfile, Principal
from emrag.ingestion.artifacts import ArtifactReader
from emrag.ingestion.lite_pipeline import is_lite_eligible
from emrag.ingestion.pipeline import IngestRequest, inbox_extension
from emrag.observability.logging import get_logger
from emrag.security.acl import can_read_document, require_read
from emrag.security.upload_validation import validate_upload, virustotal_lookup
from emrag.storage.registry import inbox_key

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("", response_model=IngestAccepted, status_code=202, summary="Upload a document")
async def upload_document(
    state: StateDep,
    principal: WriterDep,
    background: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF or page image")],
    metadata: Annotated[str | None, Form(description="JSON metadata")] = None,
) -> IngestAccepted:
    """Validate and accept a document for ingestion.

    Short native PDFs are ingested inline as a background task; anything larger,
    scanned or mixed is queued in the inbox for the full Colab pipeline. Re-
    uploading identical bytes is idempotent: the existing document is returned.
    """
    cfg = state.settings
    data = await file.read()
    try:
        meta = UploadMetadata.parse(metadata)
    except ValueError as exc:
        raise ValidationFailed(str(exc)) from exc

    check = validate_upload(
        data,
        file.filename or "upload.pdf",
        declared_mime=file.content_type,
        max_bytes=cfg.max_upload_bytes,
        max_pages=cfg.max_pages_full,
    )

    if cfg.vt_api_key:
        verdict = await virustotal_lookup(check.sha256, cfg.vt_api_key)
        if verdict == "malicious":
            raise UnsupportedMedia("file was flagged as malicious by hash reputation lookup")

    doc_id = doc_id_for_bytes(data)
    existing = await state.registry.find_document(principal.tenant_id, doc_id)
    if existing is not None and existing.status in ("ready", "ready_degraded", "processing"):
        logger.info("documents.duplicate_upload", doc_id=doc_id, status=existing.status)
        return IngestAccepted(
            document_id=doc_id,
            job_id="-",
            pipeline=existing.pipeline_profile,
            status="processing" if existing.status == "processing" else "queued",
            detail="document already ingested; nothing to do",
        )

    eligible, reason = (
        await is_lite_eligible(data, max_pages=cfg.max_pages_lite)
        if check.mime == "application/pdf"
        else (False, "page images are ingested by the full pipeline")
    )
    profile: PipelineProfile = "lite" if eligible else "full"

    await state.artifacts.put_bytes(
        inbox_key(principal.tenant_id, doc_id, inbox_extension(file.filename or "upload.pdf")),
        data,
    )
    job = await state.registry.create_job(
        kind="ingest", doc_id=doc_id, tenant_id=principal.tenant_id, pages=check.page_count
    )
    record = DocumentRecord(
        doc_id=doc_id,
        tenant_id=principal.tenant_id,
        filename=file.filename or "upload.pdf",
        title=meta.title,
        mime=check.mime,
        size_bytes=check.size_bytes,
        page_count=check.page_count,
        status="queued",
        pipeline_profile=profile,
        acl_groups=meta.acl_groups or ["all"],
        classification=meta.classification,
        tags=meta.tags,
    )
    await state.registry.put_document(record)

    ingest_request = IngestRequest(
        tenant_id=principal.tenant_id,
        filename=record.filename,
        data=data,
        doc_id=doc_id,
        title=meta.title,
        acl_groups=record.acl_groups,
        classification=record.classification,
        tags=record.tags,
        pii_mode=state.authenticator.pii_mode_for(principal.tenant_id),  # type: ignore[arg-type]
        job_id=job.job_id,
        profile=profile,
    )

    if profile == "lite":
        background.add_task(_run_lite, state, ingest_request)
        detail = reason
    else:
        detail = f"{reason}; queued for the offline pipeline"

    logger.info(
        "documents.accepted",
        doc_id=doc_id,
        pages=check.page_count,
        profile=profile,
        size_bytes=check.size_bytes,
    )
    return IngestAccepted(
        document_id=doc_id, job_id=job.job_id, pipeline=profile, status="queued", detail=detail
    )


async def _run_lite(state: object, request: IngestRequest) -> None:
    """Background task wrapper that never lets an exception escape.

    Waits for start-up to finish first. Parsing while three ONNX models load on
    two shared vCPUs turns a sub-second ingest into roughly a minute, which is
    what a user's first upload after a free-tier cold start would otherwise hit.
    """
    try:
        await state.wait_until_warm()  # type: ignore[attr-defined]
        await state.lite.run(request)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - job status already records the failure
        logger.error("documents.lite_failed", doc_id=request.doc_id, error=str(exc)[:300])


@router.get("", response_model=Page[DocumentSummary], summary="List documents")
async def list_documents(
    state: StateDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    q: Annotated[str | None, Query(description="filename contains")] = None,
) -> Page[DocumentSummary]:
    """List the tenant's documents, newest first."""
    records, next_cursor = await state.registry.list_documents(
        principal.tenant_id,
        limit=limit,
        cursor=cursor,
        status=status,
        tag=tag,
        name_contains=q,
    )
    visible = [r for r in records if can_read_document(principal, r)]
    return Page[DocumentSummary](items=[_summary(r) for r in visible], next_cursor=next_cursor)


@router.get("/{document_id}", response_model=DocumentDetail, summary="Get a document")
async def get_document(
    document_id: str, state: StateDep, principal: PrincipalDep
) -> DocumentDetail:
    """Return one document's full record, including its quality report."""
    record = require_read(principal, await _load(state, principal, document_id))
    return _detail(record)


@router.delete(
    "/{document_id}", response_model=DeleteAccepted, status_code=202, summary="Delete a document"
)
async def delete_document(
    document_id: str, state: StateDep, principal: WriterDep, background: BackgroundTasks
) -> DeleteAccepted:
    """Delete a document's vectors, artifacts and registry entry."""
    record = require_read(principal, await _load(state, principal, document_id))
    job = await state.registry.create_job(
        kind="delete", doc_id=record.doc_id, tenant_id=principal.tenant_id
    )
    record.status = "deleting"
    await state.registry.put_document(record)
    background.add_task(_run_delete, state, principal.tenant_id, record.doc_id, job.job_id)
    return DeleteAccepted(job_id=job.job_id)


async def _run_delete(state: object, tenant_id: str, doc_id: str, job_id: str) -> None:
    """Background deletion: vectors first, then artifacts and the manifest."""
    try:
        deleted = await state.store.delete_by_doc(tenant_id, doc_id)  # type: ignore[attr-defined]
        await state.registry.delete_document(tenant_id, doc_id)  # type: ignore[attr-defined]
        state.docstores.evict(tenant_id, doc_id)  # type: ignore[attr-defined]
        await state.registry.update_job(job_id, status="ready")  # type: ignore[attr-defined]
        logger.info("documents.deleted", doc_id=doc_id, vectors=deleted)
    except Exception as exc:  # noqa: BLE001
        await state.registry.update_job(  # type: ignore[attr-defined]
            job_id, status="failed", error=str(exc)[:500]
        )
        logger.error("documents.delete_failed", doc_id=doc_id, error=str(exc)[:300])


@router.get("/{document_id}/pages/{page}", summary="Render a page, optionally highlighted")
async def get_page(
    document_id: str,
    page: int,
    state: StateDep,
    principal: PrincipalDep,
    bbox: Annotated[str | None, Query(description="x0,y0,x1,y1 as page fractions")] = None,
) -> Response:
    """Return a page preview as WebP, with an optional highlight rectangle."""
    record = require_read(principal, await _load(state, principal, document_id))
    if page < 1 or (record.page_count and page > record.page_count):
        raise NotFound(f"page {page} does not exist in {document_id}")

    reader = ArtifactReader(state.artifacts)
    try:
        image = await reader.read_page_render(principal.tenant_id, document_id, page)
    except Exception:  # noqa: BLE001 - fall back to rendering from the source PDF
        source = await reader.read_source(principal.tenant_id, document_id)
        from emrag.ingestion.parsers.pymupdf_parser import render_page_at

        image = await asyncio.to_thread(render_page_at, source, page)

    if bbox:
        image = await asyncio.to_thread(_highlight, image, bbox)
    return Response(
        content=image,
        media_type="image/webp",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/{document_id}/figures/{image_ref}", summary="Fetch a figure crop")
async def get_figure(
    document_id: str, image_ref: str, state: StateDep, principal: PrincipalDep
) -> Response:
    """Return the raw image behind a figure citation."""
    require_read(principal, await _load(state, principal, document_id))
    if "/" in image_ref or ".." in image_ref:
        raise ValidationFailed("invalid figure reference")
    reader = ArtifactReader(state.artifacts)
    data = await reader.read_figure(principal.tenant_id, document_id, image_ref)
    return Response(
        content=data,
        media_type=_image_media_type(image_ref),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get(
    "/{document_id}/chunks", response_model=Page[ChunkPublic], summary="List a document's chunks"
)
async def list_chunks(
    document_id: str,
    state: StateDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    cursor: str | None = None,
    page: int | None = None,
    modality: str | None = None,
) -> Page[ChunkPublic]:
    """Page through a document's chunks, for debugging and source browsing."""
    require_read(principal, await _load(state, principal, document_id))
    chunks, next_cursor = await state.store.list_chunks(
        principal.tenant_id, document_id, limit=limit, offset=cursor
    )
    if page is not None:
        chunks = [c for c in chunks if c.page == page]
    if modality:
        chunks = [c for c in chunks if c.modality == modality]
    return Page[ChunkPublic](
        items=[ChunkPublic.from_chunk(c) for c in chunks], next_cursor=next_cursor
    )


async def _load(state: StateDep, principal: Principal, document_id: str) -> DocumentRecord:
    """Load a document record, validating the identifier shape first."""
    if not DOC_ID_RE.match(document_id):
        raise ValidationFailed(f"malformed document id: {document_id!r}")
    record = await state.registry.find_document(principal.tenant_id, document_id)
    if record is None:
        raise NotFound(f"document {document_id} not found")
    if record.tenant_id != principal.tenant_id:
        raise Forbidden("document belongs to another tenant")
    return record


def _summary(record: DocumentRecord) -> DocumentSummary:
    return DocumentSummary(
        document_id=record.doc_id,
        filename=record.filename,
        title=record.title,
        status=record.status,
        doc_class=record.doc_class,
        page_count=record.page_count,
        chunk_count=record.chunk_count,
        classification=record.classification,
        tags=list(record.tags),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _detail(record: DocumentRecord) -> DocumentDetail:
    return DocumentDetail(
        **_summary(record).model_dump(),
        mime=record.mime,
        size_bytes=record.size_bytes,
        pipeline_profile=record.pipeline_profile,
        ingest_version=record.ingest_version,
        parent_count=record.parent_count,
        table_count=record.table_count,
        figure_count=record.figure_count,
        acl_groups=list(record.acl_groups),
        quality_report=record.quality_report,
        error=record.error,
        artifacts_prefix=record.artifacts_prefix,
    )


def _highlight(image: bytes, bbox: str) -> bytes:
    """Draw a translucent rectangle over the cited region of a page render."""
    try:
        coordinates = [float(v) for v in bbox.split(",")]
        if len(coordinates) != 4 or any(not 0.0 <= c <= 1.0 for c in coordinates):
            return image
    except ValueError:
        return image

    try:
        import io

        from PIL import Image, ImageDraw

        with Image.open(io.BytesIO(image)) as source:
            canvas = source.convert("RGBA")
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            x0, y0, x1, y1 = coordinates
            box = (
                x0 * canvas.width,
                y0 * canvas.height,
                x1 * canvas.width,
                y1 * canvas.height,
            )
            draw.rectangle(box, fill=(255, 214, 10, 64), outline=(255, 149, 0, 220), width=3)
            merged = Image.alpha_composite(canvas, overlay).convert("RGB")
            buffer = io.BytesIO()
            merged.save(buffer, format="WEBP", quality=80)
            return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - highlighting is cosmetic
        logger.debug("documents.highlight_failed", error=str(exc)[:160])
        return image


def _image_media_type(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lowered.endswith(".webp"):
        return "image/webp"
    return "image/png"
