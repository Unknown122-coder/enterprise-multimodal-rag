"""Document resource schemas."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from emrag.core.types import (
    Chunk,
    Classification,
    DocClass,
    DocStatus,
    JobProgress,
    JobStatus,
    PipelineProfile,
    QualityReport,
)


class UploadMetadata(BaseModel):
    """Optional ``metadata`` part of a multipart upload, supplied as JSON."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=512)
    tags: list[str] = Field(default_factory=list, max_length=20)
    acl_groups: list[str] = Field(default_factory=lambda: ["all"], max_length=20)
    classification: Classification = "internal"

    @field_validator("tags", "acl_groups")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value if v and v.strip()]
        return cleaned

    @classmethod
    def parse(cls, raw: str | None) -> UploadMetadata:
        """Parse the JSON metadata part, raising ``ValueError`` when malformed."""
        if not raw or not raw.strip():
            return cls()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("metadata must be a JSON object")
        return cls.model_validate(payload)


class IngestAccepted(BaseModel):
    """Response to an accepted upload."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    job_id: str
    pipeline: PipelineProfile
    status: Literal["queued", "processing"] = "queued"
    detail: str | None = None


class DocumentSummary(BaseModel):
    """Document list item."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    title: str | None = None
    status: DocStatus
    doc_class: DocClass
    page_count: int
    chunk_count: int
    classification: Classification
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentSummary):
    """Full document record."""

    model_config = ConfigDict(extra="forbid")

    mime: str
    size_bytes: int
    pipeline_profile: PipelineProfile
    ingest_version: str
    parent_count: int
    table_count: int
    figure_count: int
    acl_groups: list[str]
    quality_report: QualityReport
    error: str | None = None
    artifacts_prefix: str


class ChunkPublic(BaseModel):
    """A chunk as exposed over the API: the contract minus internal debug refs."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    tenant_id: str
    filename: str
    title: str | None
    page: int
    page_count: int
    bbox: list[float]
    bbox_confidence: float
    section_path: list[str]
    modality: str
    text: str
    text_hash: str
    token_count: int
    lang: str
    table_id: str | None
    table_part: str | None
    image_ref: str | None
    caption: str | None
    parent_id: str
    prev_id: str | None
    next_id: str | None
    classification: Classification
    risk_flags: list[str]
    pii: bool
    ingest_version: str
    pipeline_profile: PipelineProfile
    embed_model: str
    ingested_at: datetime
    tags: list[str]

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkPublic:
        """Project a stored chunk into its public form."""
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.doc_id,
            tenant_id=chunk.tenant_id,
            filename=chunk.filename,
            title=chunk.title,
            page=chunk.page,
            page_count=chunk.page_count,
            bbox=list(chunk.bbox),
            bbox_confidence=chunk.bbox_confidence,
            section_path=list(chunk.section_path),
            modality=chunk.modality,
            text=chunk.text,
            text_hash=chunk.text_hash,
            token_count=chunk.token_count,
            lang=chunk.lang,
            table_id=chunk.table_id,
            table_part=chunk.table_part,
            image_ref=chunk.image_ref,
            caption=chunk.caption,
            parent_id=chunk.parent_id,
            prev_id=chunk.prev_id,
            next_id=chunk.next_id,
            classification=chunk.classification,
            risk_flags=list(chunk.risk_flags),
            pii=chunk.pii,
            ingest_version=chunk.ingest_version,
            pipeline_profile=chunk.pipeline_profile,
            embed_model=chunk.embed_model,
            ingested_at=chunk.ingested_at,
            tags=list(chunk.tags),
        )


class JobResponse(BaseModel):
    """Job status response."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: str
    document_id: str
    status: JobStatus
    progress: JobProgress
    worker: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class DeleteAccepted(BaseModel):
    """Response to an accepted delete request."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["queued"] = "queued"
