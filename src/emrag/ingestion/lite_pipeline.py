"""Lite ingestion path.

For small, native-text PDFs the full pipeline is overkill: there is nothing to
OCR and usually nothing worth captioning, and a user who uploaded a ten-page memo
expects to query it immediately rather than wait for an operator to run a
notebook. This path runs on the API host as a background task with a hard page
limit, skipping Docling, OCR and vision entirely.

Artifacts are written in exactly the same layout as the full path, with
``pipeline_profile="lite"`` recorded on every chunk, so a document can later be
re-ingested at full quality without any special case.
"""

from __future__ import annotations

import asyncio

from emrag.core.types import PipelineProfile
from emrag.ingestion.chunking import ChunkingConfig
from emrag.ingestion.pipeline import IngestionPipeline, IngestRequest, IngestResult
from emrag.ingestion.triage import triage_pdf
from emrag.observability.logging import get_logger

logger = get_logger(__name__)


async def is_lite_eligible(data: bytes, *, max_pages: int) -> tuple[bool, str]:
    """Decide whether a document can take the lite path.

    Eligible when it is short and its text layer is extractable. Returns the
    decision and a human-readable reason for the upload response.
    """
    try:
        pages, doc_class = await asyncio.to_thread(triage_pdf, data)
    except Exception as exc:  # noqa: BLE001 - treat unreadable files as full-path
        return False, f"triage failed ({type(exc).__name__}); queued for full ingestion"
    if len(pages) > max_pages:
        return False, f"{len(pages)} pages exceeds the lite limit of {max_pages}"
    if doc_class != "native":
        return False, f"document is {doc_class}; OCR runs in the full pipeline"
    return True, f"{len(pages)} native pages ingested inline"


class LitePipeline:
    """Thin wrapper that forces the host-safe pipeline configuration."""

    def __init__(
        self, pipeline: IngestionPipeline, *, chunking: ChunkingConfig | None = None
    ) -> None:
        self._pipeline = pipeline
        self._chunking = chunking

    async def run(self, request: IngestRequest) -> IngestResult:
        """Ingest one document without Docling, OCR or vision captioning."""
        profile: PipelineProfile = "lite"
        request.profile = profile
        logger.info("ingest.lite_started", doc_id=request.resolved_doc_id())
        return await self._pipeline.run(request)
