"""Dependency wiring.

A single :class:`AppState` object is built once during the application lifespan
and shared by every request. Models and clients are expensive, so nothing is
constructed per request; the only per-request work is authentication and building
the retrieval filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Header, Request

from emrag.citation.verifier import CitationVerifier
from emrag.core.errors import Forbidden, Unauthenticated
from emrag.core.types import Principal
from emrag.embeddings.base import EmbeddingProvider
from emrag.embeddings.factory import build_embedder
from emrag.generation.answerer import Answerer
from emrag.ingestion.chunking import ChunkingConfig
from emrag.ingestion.lite_pipeline import LitePipeline
from emrag.ingestion.pipeline import IngestionPipeline
from emrag.llm.router import LLMRouter
from emrag.observability.logging import get_logger
from emrag.retrieval.rerank import build_reranker
from emrag.retrieval.retriever import Retriever
from emrag.security.auth import Authenticator
from emrag.settings import Settings, get_settings
from emrag.storage.artifact_store import ArtifactStore, build_artifact_store
from emrag.storage.docstore import DocStoreCache
from emrag.storage.registry import Registry
from emrag.vectorstore.base import VectorStore
from emrag.vectorstore.factory import build_vector_store

logger = get_logger(__name__)


@dataclass
class AppState:
    """Process-wide singletons."""

    settings: Settings
    artifacts: ArtifactStore
    registry: Registry
    docstores: DocStoreCache
    store: VectorStore
    embedder: EmbeddingProvider
    router: LLMRouter
    retriever: Retriever
    answerer: Answerer
    authenticator: Authenticator
    pipeline: IngestionPipeline
    lite: LitePipeline
    warm: bool = False

    async def aclose(self) -> None:
        """Release every resource held by the application."""
        await self.router.aclose()
        await self.store.aclose()


def build_state(settings: Settings | None = None) -> AppState:
    """Construct the application state. Called once per process."""
    cfg = settings or get_settings()

    artifacts = build_artifact_store(cfg)
    registry = Registry(artifacts)
    docstores = DocStoreCache(artifacts, Path(cfg.artifact_local_root) / "cache" / "docstores")
    store = build_vector_store(cfg)
    embedder = build_embedder(cfg)
    router = LLMRouter.from_settings(cfg)
    reranker = build_reranker(cfg)

    retriever = Retriever(
        store=store,
        embedder=embedder,
        reranker=reranker,
        router=router,
        docstores=docstores,
    )
    verifier = CitationVerifier(threshold=cfg.quote_match_threshold)
    answerer = Answerer(retriever=retriever, router=router, verifier=verifier)

    chunking = ChunkingConfig.from_mapping(cfg.tuning_section("chunking"))
    pipeline = IngestionPipeline(
        registry=registry,
        store=store,
        embedder=embedder,
        router=router,
        chunking=chunking,
        tuning=cfg.tuning,
        use_docling=cfg.use_docling,
    )
    lite = LitePipeline(
        IngestionPipeline(
            registry=registry,
            store=store,
            embedder=embedder,
            router=None,
            chunking=chunking,
            tuning=cfg.tuning,
            use_docling=False,
        ),
        chunking=chunking,
    )

    return AppState(
        settings=cfg,
        artifacts=artifacts,
        registry=registry,
        docstores=docstores,
        store=store,
        embedder=embedder,
        router=router,
        retriever=retriever,
        answerer=answerer,
        authenticator=Authenticator(cfg),
        pipeline=pipeline,
        lite=lite,
    )


def get_state(request: Request) -> AppState:
    """Return the application state attached to the running app."""
    state: AppState | None = getattr(request.app.state, "emrag", None)
    if state is None:  # pragma: no cover - lifespan always sets this
        raise RuntimeError("application state is not initialised")
    return state


StateDep = Annotated[AppState, Depends(get_state)]


async def get_principal(
    state: StateDep,
    authorization: Annotated[str | None, Header()] = None,
    x_tenant_id: Annotated[str | None, Header()] = None,
) -> Principal:
    """Authenticate the caller and resolve its tenant."""
    principal = await state.authenticator.authenticate(authorization, tenant_override=x_tenant_id)
    from emrag.observability.logging import request_id_var, set_request_context

    set_request_context(request_id_var.get(), principal.tenant_id)
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]


async def get_writer(principal: PrincipalDep) -> Principal:
    """Require a credential that may create or delete documents."""
    if not principal.can_write():
        raise Forbidden("this credential is read-only")
    return principal


WriterDep = Annotated[Principal, Depends(get_writer)]


async def get_admin(principal: PrincipalDep) -> Principal:
    """Require a tenant administrator."""
    if not principal.is_admin():
        raise Forbidden("this endpoint requires an admin credential")
    return principal


AdminDep = Annotated[Principal, Depends(get_admin)]


__all__ = [
    "AdminDep",
    "AppState",
    "PrincipalDep",
    "StateDep",
    "Unauthenticated",
    "WriterDep",
    "build_state",
    "get_state",
]
