"""Query request and response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from emrag import SCHEMA_VERSION
from emrag.core.types import (
    Answer,
    Citation,
    QueryResult,
    QueryStatus,
    RetrievalInfo,
    Usage,
)

RewriteMode = Literal["auto", "off", "on"]
ModelTier = Literal["fast", "quality"]


class QueryFilters(BaseModel):
    """Narrowing filters supplied by the caller. Cannot widen tenant access."""

    model_config = ConfigDict(extra="forbid")

    doc_ids: list[str] | None = Field(default=None, max_length=50)
    page_range: tuple[int, int] | None = None
    modalities: list[Literal["text", "table", "figure", "ocr_text"]] | None = None
    tags: list[str] | None = Field(default=None, max_length=20)
    section_contains: str | None = Field(default=None, max_length=200)


class QueryOptions(BaseModel):
    """Per-request retrieval and generation knobs."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=8, ge=1, le=12)
    rewrite: RewriteMode = "auto"
    parent_expansion: bool = True
    repair: bool = True
    deadline_ms: int = Field(default=4500, ge=1000, le=10000)
    model_tier: ModelTier = "fast"
    include_page_images: bool = True


class Conversation(BaseModel):
    """Compact conversation context. Only a summary is sent to the model."""

    model_config = ConfigDict(extra="forbid")

    summary: str | None = Field(default=None, max_length=600)


class QueryRequest(BaseModel):
    """Body of ``POST /v1/query`` and ``POST /v1/query/stream``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    filters: QueryFilters | None = None
    options: QueryOptions = Field(default_factory=QueryOptions)
    conversation: Conversation | None = None

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class QueryResponse(BaseModel):
    """Body of a successful ``POST /v1/query``."""

    model_config = ConfigDict(extra="forbid")

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
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_result(cls, result: QueryResult) -> QueryResponse:
        """Adapt the transport-independent result into the HTTP response."""
        return cls(
            request_id=result.request_id,
            status=result.status,
            answer=result.answer,
            citations=result.citations,
            nearest_sources=result.nearest_sources,
            warnings=result.warnings,
            reason=result.reason,
            retrieval=result.retrieval,
            generation=result.generation,
            latency_ms=result.latency_ms,
            debug=result.debug,
        )


class RetrieveRequest(BaseModel):
    """Body of ``POST /v1/retrieve`` (retrieval only, no generation)."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    filters: QueryFilters | None = None
    top_k: int = Field(default=8, ge=1, le=40)
    rerank: bool = True


class RetrievedChunk(BaseModel):
    """One retrieval hit, including its text so a UI can display it."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    filename: str
    page: int
    bbox: list[float]
    modality: str
    section_path: list[str]
    text: str
    score: float
    rerank_score: float | None = None
    page_image_url: str | None = None


class RetrieveResponse(BaseModel):
    """Body of ``POST /v1/retrieve``."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[RetrievedChunk]
    gate_passed: bool
    gate_score: float | None = None
    strategy: str
    latency_ms: int
    schema_version: str = SCHEMA_VERSION


class UsageResponse(BaseModel):
    """Body of ``GET /v1/usage``."""

    model_config = ConfigDict(extra="forbid")

    window_start: str
    models: dict[str, dict[str, Any]]


class MetaResponse(BaseModel):
    """Body of ``GET /v1/meta``."""

    model_config = ConfigDict(extra="forbid")

    version: str
    ingest_version: str
    schema_version: str
    vector_backend: str
    collection: str
    embed_model: str
    sparse_model: str
    rerank_model: str
    tokenizer: str
    models: dict[str, list[str]]
