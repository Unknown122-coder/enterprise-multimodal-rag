"""Deterministic hash embedder.

Not a semantic model. It exists so unit and integration tests can exercise the
full ingest-to-query path offline, with stable vectors and no model download.
Selected with ``EMBED_BACKEND=deterministic``.
"""

from __future__ import annotations

import hashlib
import math
import re

from emrag.core.types import DenseSparse, QueryVectors

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicEmbedder:
    """Hashing embedder with a bag-of-words sparse companion."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.model_name = f"deterministic-hash-{dim}"
        self.sparse_model_name = "deterministic-bow"

    def embed_documents(self, texts: list[str]) -> list[DenseSparse]:
        """Embed passages."""
        out: list[DenseSparse] = []
        for text in texts:
            dense = self._dense(text)
            indices, values = self._sparse(text)
            out.append(DenseSparse(dense=dense, sparse_indices=indices, sparse_values=values))
        return out

    def embed_query(self, text: str) -> QueryVectors:
        """Embed one query."""
        indices, values = self._sparse(text)
        return QueryVectors(dense=self._dense(text), sparse_indices=indices, sparse_values=values)

    def warm_up(self) -> None:
        """No-op: nothing to load."""

    def _dense(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _sparse(self, text: str) -> tuple[list[int], list[float]]:
        counts: dict[int, float] = {}
        for token in _TOKEN_RE.findall(text.lower()):
            idx = int.from_bytes(
                hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big"
            )
            counts[idx] = counts.get(idx, 0.0) + 1.0
        items = sorted(counts.items())
        return [k for k, _ in items], [v for _, v in items]
