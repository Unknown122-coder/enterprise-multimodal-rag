"""Health, readiness and metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from emrag import INGEST_VERSION, SCHEMA_VERSION, __version__
from emrag.api.deps import StateDep
from emrag.api.schemas.common import Health, Readiness
from emrag.api.schemas.query import MetaResponse
from emrag.core import tokens
from emrag.observability.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["ops"])


def _artifact_location(state: StateDep) -> str:
    """Where artifacts are stored, surfaced so a mis-set path is visible at boot."""
    settings = state.settings
    if settings.artifact_backend == "hf":
        return f"hf://{settings.hf_dataset_repo}"
    return str(settings.artifact_local_root)


@router.get("/healthz", response_model=Health, summary="Liveness probe")
async def healthz() -> Health:
    """Return immediately; used by the container health check."""
    return Health(ok=True, version=__version__)


@router.get("/readyz", response_model=Readiness, summary="Readiness probe")
async def readyz(state: StateDep) -> Readiness:
    """Report whether models are warm, the vector store is reachable and the registry synced."""
    store_health = await state.store.health()
    registry_stats = await state.registry.stats()
    checks = {
        "models": state.warm,
        # Chunk boundaries depend on the tokeniser, so a fallback is a
        # correctness issue and not merely slower.
        "tokenizer": tokens.encoder_available(),
        "store": store_health.reachable,
        "registry": bool(registry_stats.get("loaded")),
        "llm": bool(state.router.models_by_tier()),
    }
    detail: dict[str, str] = {
        "backend": store_health.backend,
        "collection": store_health.collection,
        "points": str(store_health.points if store_health.points is not None else "unknown"),
        "documents": str(registry_stats.get("documents", 0)),
        "tokenizer": tokens.ENCODING_NAME if tokens.encoder_available() else "fallback-heuristic",
        "artifact_root": _artifact_location(state),
    }
    if store_health.detail:
        detail["store_detail"] = store_health.detail
    return Readiness(ready=all(checks.values()), checks=checks, detail=detail)


@router.get("/v1/meta", response_model=MetaResponse, summary="Deployment metadata")
async def meta(state: StateDep) -> MetaResponse:
    """Report versions, the active backend and the routable models."""
    cfg = state.settings
    return MetaResponse(
        version=__version__,
        ingest_version=INGEST_VERSION,
        schema_version=SCHEMA_VERSION,
        vector_backend=cfg.vector_backend,
        collection=cfg.collection_name,
        embed_model=state.embedder.model_name,
        sparse_model=state.embedder.sparse_model_name,
        rerank_model=cfg.rerank_model if cfg.rerank_enabled else "none",
        tokenizer=tokens.ENCODING_NAME if tokens.encoder_available() else "fallback-heuristic",
        models=state.router.models_by_tier(),
    )
