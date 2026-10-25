"""Embedding provider factory."""

from __future__ import annotations

import functools

from emrag.embeddings.base import EmbeddingProvider
from emrag.settings import Settings, get_settings


def build_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the embedder selected by ``EMBED_BACKEND``."""
    cfg = settings or get_settings()
    if cfg.embed_backend == "deterministic":
        from emrag.embeddings.deterministic import DeterministicEmbedder

        return DeterministicEmbedder(dim=cfg.embed_dim)  # type: ignore[return-value]

    if cfg.embed_backend == "gemini":
        from emrag.embeddings.gemini_embedder import GeminiEmbedder

        return GeminiEmbedder(
            cfg.gemini_api_key or "", model_name=cfg.embed_model, dim=cfg.embed_dim
        )

    from emrag.embeddings.fastembed_embedder import FastEmbedEmbedder

    return FastEmbedEmbedder(
        model_name=cfg.embed_model,
        sparse_model_name=cfg.sparse_model,
        dim=cfg.embed_dim,
        cache_dir=str(cfg.fastembed_cache_path) if cfg.fastembed_cache_path else None,
    )


@functools.lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    """Process-wide embedder singleton (models are expensive to load)."""
    return build_embedder()
