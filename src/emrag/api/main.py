"""FastAPI application factory.

Startup does the expensive work once: load the embedding and reranking models,
ensure the vector collection exists, sync the document registry, and run one
warm-up inference. ``/readyz`` stays red until that finishes, which is what makes
a cold start on a free-tier host visible rather than mysterious.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from emrag import __version__
from emrag.api.deps import AppState, build_state
from emrag.api.middleware.errors import register_exception_handlers
from emrag.api.middleware.ratelimit import build_limiter, register_rate_limiting
from emrag.api.middleware.request_id import RequestIdMiddleware
from emrag.api.middleware.security_headers import SecurityHeadersMiddleware
from emrag.api.routers import documents, health, internal, jobs, query, retrieve, usage
from emrag.observability.logging import configure_logging, get_logger
from emrag.observability.tracing import configure_tracing
from emrag.settings import Settings, get_settings

logger = get_logger(__name__)

DESCRIPTION = """
Answers questions over corporate documents with citations that are verified
against the source text before they are returned.

Every claim carries at least one citation naming the document, page, chunk id,
bounding region and a quote that provably appears in that chunk. When the corpus
does not support an answer the response status is `not_found` rather than prose.
""".strip()

QUOTA_FLUSH_SECONDS = 60


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Importable for tests without starting a server."""
    cfg = settings or get_settings()
    configure_logging(cfg.log_level, json_output=cfg.app_env != "dev")
    configure_tracing(cfg.otel_exporter_otlp_endpoint)

    app = FastAPI(
        title="Enterprise Multimodal RAG",
        version=__version__,
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )
    app.state.settings = cfg

    app.add_middleware(SecurityHeadersMiddleware)
    if cfg.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origin_list,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Tenant-ID"],
            expose_headers=["X-Request-ID", "X-Response-Time-ms"],
            max_age=600,
        )
    register_rate_limiting(app, build_limiter(cfg))
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(retrieve.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(usage.router)
    app.include_router(internal.router)
    return app


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build state, warm the models, then clean up on shutdown."""
    cfg: Settings = app.state.settings
    state = build_state(cfg)
    app.state.emrag = state

    flusher = asyncio.create_task(_flush_quota(state))
    warmer = asyncio.create_task(_warm_up(state))
    try:
        yield
    finally:
        for task in (flusher, warmer):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await state.aclose()
        logger.info("app.stopped")


async def _warm_up(state: AppState) -> None:
    """Ensure the schema exists, sync the registry and warm the models."""
    try:
        await state.store.ensure_schema(state.settings.embed_dim, sparse=True)
    except Exception as exc:  # noqa: BLE001 - the store may come up after the API
        logger.warning("app.schema_unavailable", error=str(exc)[:200])

    try:
        await state.registry.refresh()
    except Exception as exc:  # noqa: BLE001
        logger.warning("app.registry_sync_failed", error=str(exc)[:200])

    try:
        await asyncio.to_thread(state.embedder.warm_up)
        reranker = getattr(state.retriever, "_reranker", None)
        if reranker is not None:
            await asyncio.to_thread(reranker.warm_up)
        state.warm = True
        logger.info(
            "app.ready",
            version=__version__,
            backend=state.settings.vector_backend,
            embed_model=state.embedder.model_name,
            tiers=sorted(state.router.models_by_tier()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("app.warm_up_failed", error=str(exc)[:300])


async def _flush_quota(state: AppState) -> None:
    """Persist the quota ledger periodically so a restart keeps daily counters."""
    while True:
        try:
            await asyncio.sleep(QUOTA_FLUSH_SECONDS)
            await state.router.ledger.flush()
        except asyncio.CancelledError:
            await state.router.ledger.flush()
            raise
        except Exception as exc:  # noqa: BLE001 - never kill the background task
            logger.warning("app.quota_flush_failed", error=str(exc)[:200])


app = create_app()
