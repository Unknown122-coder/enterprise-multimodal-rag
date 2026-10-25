"""Vector store factory. The only place a backend is chosen."""

from __future__ import annotations

from emrag.core.errors import ConfigurationMissing
from emrag.settings import Settings, get_settings
from emrag.vectorstore.base import VectorStore


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the vector store selected by ``VECTOR_BACKEND``."""
    cfg = settings or get_settings()
    if cfg.vector_backend == "qdrant":
        from emrag.vectorstore.qdrant_store import QdrantStore

        return QdrantStore(
            url=cfg.qdrant_url, collection=cfg.collection_name, api_key=cfg.qdrant_api_key
        )
    if cfg.vector_backend == "pinecone":
        from emrag.vectorstore.pinecone_store import PineconeStore

        if not cfg.pinecone_api_key:
            raise ConfigurationMissing("PINECONE_API_KEY is required for VECTOR_BACKEND=pinecone")
        return PineconeStore(
            api_key=cfg.pinecone_api_key,
            dense_index=cfg.pinecone_index_dense,
            sparse_index=cfg.pinecone_index_sparse,
        )
    raise ConfigurationMissing(f"unknown VECTOR_BACKEND: {cfg.vector_backend!r}")
