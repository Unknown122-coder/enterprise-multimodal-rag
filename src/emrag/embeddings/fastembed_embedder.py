"""fastembed-backed embeddings.

Dense: ``BAAI/bge-base-en-v1.5`` (768d, cosine). Sparse: ``Qdrant/bm25``, whose
IDF is computed server-side by Qdrant. Both run as ONNX on CPU, which keeps the
query path free of any API quota and lets Colab and the API host produce
byte-identical vectors.
"""

from __future__ import annotations

import threading
from typing import Any

from emrag.core.types import DenseSparse, QueryVectors
from emrag.embeddings.base import EmbeddingProvider
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

# bge models are trained with an asymmetric query instruction.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FastEmbedEmbedder(EmbeddingProvider):
    """Dense + sparse embeddings via fastembed, loaded lazily and thread-safely."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25",
        *,
        dim: int = 768,
        cache_dir: str | None = None,
        threads: int | None = None,
        use_query_prefix: bool = True,
    ) -> None:
        self.model_name = model_name
        self.sparse_model_name = sparse_model_name
        self.dim = dim
        self._cache_dir = cache_dir
        self._threads = threads
        self._use_query_prefix = use_query_prefix and "bge" in model_name.lower()
        self._dense: Any = None
        self._sparse: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._dense is not None and self._sparse is not None:
            return
        with self._lock:
            if self._dense is None:
                from fastembed import TextEmbedding

                self._dense = TextEmbedding(
                    model_name=self.model_name, cache_dir=self._cache_dir, threads=self._threads
                )
                logger.info("embeddings.loaded", model=self.model_name, dim=self.dim)
            if self._sparse is None:
                from fastembed import SparseTextEmbedding

                self._sparse = SparseTextEmbedding(
                    model_name=self.sparse_model_name,
                    cache_dir=self._cache_dir,
                    threads=self._threads,
                )
                logger.info("embeddings.loaded", model=self.sparse_model_name)

    def embed_documents(self, texts: list[str]) -> list[DenseSparse]:
        """Embed passages in one batch."""
        if not texts:
            return []
        self._ensure_loaded()
        dense_vectors = list(self._dense.embed(texts))
        sparse_vectors = list(self._sparse.embed(texts))
        out: list[DenseSparse] = []
        for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True):
            out.append(
                DenseSparse(
                    dense=[float(x) for x in dense],
                    sparse_indices=[int(i) for i in sparse.indices],
                    sparse_values=[float(v) for v in sparse.values],
                )
            )
        return out

    def embed_query(self, text: str) -> QueryVectors:
        """Embed one query, applying the bge query instruction when appropriate."""
        self._ensure_loaded()
        dense_input = f"{_QUERY_PREFIX}{text}" if self._use_query_prefix else text
        dense = next(iter(self._dense.query_embed(dense_input)))
        sparse = next(iter(self._sparse.query_embed(text)))
        return QueryVectors(
            dense=[float(x) for x in dense],
            sparse_indices=[int(i) for i in sparse.indices],
            sparse_values=[float(v) for v in sparse.values],
        )

    def warm_up(self) -> None:
        """Load both models and run one embedding of each kind."""
        self._ensure_loaded()
        self.embed_documents(["warm up"])
        self.embed_query("warm up")
