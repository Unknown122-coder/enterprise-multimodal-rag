"""Shared typed models.

These mirror the Build Contract schemas exactly (chunk contract, API objects).
Everything that crosses a module boundary is one of these types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from emrag.core.ids import CHUNK_ID_RE, DOC_ID_RE, PARENT_ID_RE, TENANT_ID_RE

Modality = Literal["text", "table", "figure", "ocr_text"]
Classification = Literal["public", "internal", "confidential"]
PipelineProfile = Literal["full", "lite"]
DocClass = Literal["native", "scanned", "mixed"]
JobKind = Literal["ingest", "delete"]
JobStatus = Literal["queued", "processing", "ready", "ready_degraded", "failed"]
DocStatus = Literal["queued", "processing", "ready", "ready_degraded", "failed", "deleting"]
QueryStatus = Literal["answered", "partial", "not_found", "refused"]
Role = Literal["reader", "writer", "admin"]
ModelTier = Literal["fast", "quality", "rewrite", "repair", "vision", "judge"]

CLASSIFICATION_ORDER: dict[str, int] = {"public": 0, "internal": 1, "confidential": 2}

BBox = Annotated[list[float], Field(min_length=4, max_length=4)]


def utcnow() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base model: rejects unexpected fields and validates on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Element(StrictModel):
    """One layout element produced by a parser, before chunking."""

    kind: Literal["heading", "paragraph", "list_item", "table", "figure", "caption", "ocr_block"]
    page: int = Field(ge=1)
    bbox: BBox
    bbox_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    text: str = ""
    level: int | None = Field(default=None, ge=1, le=8)
    element_id: str | None = None
    table_cells: list[list[str]] | None = None
    table_caption: str | None = None
    image_bytes_ref: str | None = None
    order: int = 0


class PageInfo(StrictModel):
    """Per-page triage result and geometry."""

    page: int = Field(ge=1)
    width_pt: float
    height_pt: float
    char_count: int = 0
    image_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    scanned: bool = False
    blank: bool = False


class Chunk(StrictModel):
    """An embedded, citable unit of content."""

    chunk_id: str
    doc_id: str
    tenant_id: str
    filename: str = Field(max_length=255)
    title: str | None = Field(default=None, max_length=512)
    source_uri: str | None = None

    page: int = Field(ge=1)
    page_count: int = Field(ge=1)
    bbox: BBox
    bbox_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    page_width_pt: float | None = None
    page_height_pt: float | None = None
    section_path: list[str] = Field(default_factory=list, max_length=8)
    element_ids: list[str] = Field(default_factory=list)

    modality: Modality
    text: str = Field(min_length=1, max_length=32768)
    text_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    token_count: int = Field(ge=1, le=1024)
    lang: str = Field(default="en", pattern=r"^[a-z]{2}$")
    table_id: str | None = None
    table_part: str | None = Field(default=None, pattern=r"^[0-9]+/[0-9]+$")
    image_ref: str | None = None
    caption: str | None = None

    parent_id: str
    prev_id: str | None = None
    next_id: str | None = None

    acl_groups: list[str] = Field(min_length=1)
    classification: Classification = "internal"
    risk_flags: list[str] = Field(default_factory=list)
    pii: bool = False

    ingest_version: str
    pipeline_profile: PipelineProfile
    embed_model: str
    sparse_model: str
    ingested_at: datetime
    tags: list[str] = Field(default_factory=list)

    @field_validator("chunk_id")
    @classmethod
    def _v_chunk_id(cls, v: str) -> str:
        if not CHUNK_ID_RE.match(v):
            raise ValueError("chunk_id does not match the contract pattern: " + repr(v))
        return v

    @field_validator("doc_id")
    @classmethod
    def _v_doc_id(cls, v: str) -> str:
        if not DOC_ID_RE.match(v):
            raise ValueError("doc_id does not match the contract pattern: " + repr(v))
        return v

    @field_validator("tenant_id")
    @classmethod
    def _v_tenant(cls, v: str) -> str:
        if not TENANT_ID_RE.match(v):
            raise ValueError("tenant_id does not match the contract pattern: " + repr(v))
        return v

    @field_validator("parent_id")
    @classmethod
    def _v_parent(cls, v: str) -> str:
        if not PARENT_ID_RE.match(v):
            raise ValueError("parent_id does not match the contract pattern: " + repr(v))
        return v

    @field_validator("bbox")
    @classmethod
    def _v_bbox(cls, v: list[float]) -> list[float]:
        if any(not 0.0 <= c <= 1.0 for c in v):
            raise ValueError("bbox coordinates must be normalised to [0, 1]")
        if v[0] > v[2] or v[1] > v[3]:
            raise ValueError("bbox must be ordered as x0 <= x1 and y0 <= y1")
        return v

    @model_validator(mode="after")
    def _v_consistency(self) -> Chunk:
        if not self.chunk_id.startswith(self.doc_id):
            raise ValueError("chunk_id must be prefixed with its doc_id")
        if not self.parent_id.startswith(self.doc_id):
            raise ValueError("parent_id must be prefixed with its doc_id")
        if self.page > self.page_count:
            raise ValueError("page must not exceed page_count")
        if self.modality == "figure" and not self.image_ref:
            raise ValueError("figure chunks require image_ref")
        if self.modality == "table" and not self.table_id:
            raise ValueError("table chunks require table_id")
        return self


class Parent(StrictModel):
    """A section-level container used for context expansion. Never embedded."""

    parent_id: str
    doc_id: str
    tenant_id: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str
    token_count: int = Field(ge=0)
    child_ids: list[str] = Field(default_factory=list)


class DenseSparse(StrictModel):
    """A dense vector plus its BM25 sparse companion."""

    dense: list[float]
    sparse_indices: list[int] = Field(default_factory=list)
    sparse_values: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _v_sparse(self) -> DenseSparse:
        if len(self.sparse_indices) != len(self.sparse_values):
            raise ValueError("sparse_indices and sparse_values must have equal length")
        return self


class QualityReport(StrictModel):
    """Outcome of the ingestion quality gates."""

    pages_total: int = 0
    pages_with_chunks: int = 0
    pages_blank: int = 0
    pages_failed: list[int] = Field(default_factory=list)
    ocr_pages: int = 0
    degraded_reasons: list[str] = Field(default_factory=list)
    passed: bool = True


class DocumentRecord(StrictModel):
    """Registry record for one ingested document."""

    doc_id: str
    tenant_id: str
    filename: str
    title: str | None = None
    mime: str = "application/pdf"
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=0)
    doc_class: DocClass = "native"
    status: DocStatus = "queued"
    pipeline_profile: PipelineProfile = "full"
    ingest_version: str = "0.0.0"
    chunk_count: int = 0
    parent_count: int = 0
    figure_count: int = 0
    table_count: int = 0
    acl_groups: list[str] = Field(default_factory=lambda: ["all"])
    classification: Classification = "internal"
    tags: list[str] = Field(default_factory=list)
    quality_report: QualityReport = Field(default_factory=QualityReport)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    artifacts_prefix: str = ""


class JobProgress(StrictModel):
    """Page-level progress of an ingestion job."""

    pages_done: int = 0
    pages_total: int = 0


class Job(StrictModel):
    """An asynchronous ingestion or deletion job."""

    job_id: str
    kind: JobKind
    doc_id: str
    tenant_id: str
    status: JobStatus = "queued"
    progress: JobProgress = Field(default_factory=JobProgress)
    worker: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RetrievalFilter(StrictModel):
    """Provider-neutral filter. The tenant is mandatory and never user supplied."""

    tenant_id: str
    acl_groups: list[str] = Field(min_length=1)
    max_classification: Classification = "confidential"
    doc_ids: list[str] | None = None
    page_range: tuple[int, int] | None = None
    modalities: list[Modality] | None = None
    tags: list[str] | None = None
    section_contains: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def _v_tenant(cls, v: str) -> str:
        if not TENANT_ID_RE.match(v):
            raise ValueError("invalid tenant_id: " + repr(v))
        return v

    def allowed_classifications(self) -> list[str]:
        """Return classifications at or below ``max_classification``."""
        ceiling = CLASSIFICATION_ORDER[self.max_classification]
        return [k for k, v in CLASSIFICATION_ORDER.items() if v <= ceiling]


class Candidate(StrictModel):
    """A retrieved chunk with its scores."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    @property
    def effective_score(self) -> float:
        """Return the rerank score when available, else the fused retrieval score."""
        return self.rerank_score if self.rerank_score is not None else self.score


class QueryVectors(StrictModel):
    """Embedded form of one query string."""

    dense: list[float]
    sparse_indices: list[int] = Field(default_factory=list)
    sparse_values: list[float] = Field(default_factory=list)


class Citation(StrictModel):
    """A verifiable pointer from an answer claim into a source chunk."""

    ref: int = Field(ge=1)
    chunk_id: str
    document_id: str
    filename: str
    title: str | None = None
    page: int = Field(ge=1)
    bbox: BBox
    modality: Modality
    section_path: list[str] = Field(default_factory=list)
    quote: str | None = None
    verified: bool = False
    match_score: float = 0.0
    retrieval_score: float = 0.0
    page_image_url: str | None = None
    image_url: str | None = None


class Claim(StrictModel):
    """One assertion of the answer plus the citations that support it."""

    index: int = Field(ge=0)
    text: str
    citation_refs: list[int] = Field(default_factory=list)


class Answer(StrictModel):
    """Rendered answer: markdown plus the structured claims behind it."""

    markdown: str
    claims: list[Claim] = Field(default_factory=list)


class Usage(StrictModel):
    """Token accounting for one generation call."""

    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    repair_rounds: int = 0


class RetrievalInfo(StrictModel):
    """Retrieval telemetry returned to the client."""

    strategy: str
    candidates: int
    used: int
    rewrites: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    gate_score: float | None = None


class QueryResult(StrictModel):
    """Complete outcome of a query, independent of transport."""

    request_id: str
    status: QueryStatus
    answer: Answer | None = None
    citations: list[Citation] = Field(default_factory=list)
    nearest_sources: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reason: str | None = None
    retrieval: RetrievalInfo
    generation: Usage | None = None
    latency_ms: int = 0
    debug: dict[str, Any] | None = None


class Principal(StrictModel):
    """The authenticated caller. The tenant comes from the credential only."""

    subject: str
    tenant_id: str
    groups: list[str] = Field(min_length=1)
    clearance: Classification = "internal"
    role: Role = "reader"
    auth_method: Literal["api_key", "jwt", "dev_anon"] = "api_key"

    def can_write(self) -> bool:
        """Return True when the principal may create or delete documents."""
        return self.role in ("writer", "admin")

    def is_admin(self) -> bool:
        """Return True for tenant administrators."""
        return self.role == "admin"
