"""Gemini embedding provider (alternative profile).

Exists for hosts too small to run a local ONNX model, notably Render's 512 MB
free web service. It is **not** the default, and the trade-off is deliberate and
significant:

* the query path now consumes an API quota, so a day's queries are capped and an
  exhausted quota breaks search rather than degrading it;
* there is no sparse companion vector, so retrieval is dense-only and loses the
  lexical recall that exact identifiers, part numbers and rare terms depend on;
* ingestion and query must use the same model and dimension, so switching profile
  on an existing corpus requires a re-embed from artifacts.

Matryoshka truncation keeps the output at 768 dimensions, matching the default
profile so the collection schema is unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from emrag.core.errors import ProviderError, RateLimited
from emrag.core.types import DenseSparse, QueryVectors
from emrag.embeddings.base import EmbeddingProvider
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_MAX_BATCH = 100
_MAX_ATTEMPTS = 3


class GeminiEmbedder(EmbeddingProvider):
    """Dense-only embeddings from the Gemini embedding API."""

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = "gemini-embedding-001",
        dim: int = 768,
        base_url: str = _BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is required for EMBED_BACKEND=gemini")
        self.model_name = model_name
        self.sparse_model_name = "none"
        self.dim = dim
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))

    def embed_documents(self, texts: list[str]) -> list[DenseSparse]:
        """Embed passages with the retrieval-document task type."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            vectors.extend(
                self._embed_batch(texts[start : start + _MAX_BATCH], "RETRIEVAL_DOCUMENT")
            )
        return [DenseSparse(dense=v) for v in vectors]

    def embed_query(self, text: str) -> QueryVectors:
        """Embed one query with the retrieval-query task type."""
        dense = self._embed_batch([text], "RETRIEVAL_QUERY")[0]
        return QueryVectors(dense=dense)

    def warm_up(self) -> None:
        """Issue one tiny request so a cold start surfaces a bad key immediately."""
        self.embed_query("warm up")

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        url = f"{self._base_url}/models/{self.model_name}:batchEmbedContents"
        body: dict[str, Any] = {
            "requests": [
                {
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dim,
                }
                for text in texts
            ]
        }
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.post(url, params={"key": self._api_key}, json=body)
            except httpx.HTTPError as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise ProviderError(f"gemini embedding transport error: {exc}") from exc
                time.sleep(attempt * 1.5)
                continue

            if response.status_code == 429:
                if attempt == _MAX_ATTEMPTS:
                    raise RateLimited("gemini embedding quota exhausted", retry_after_s=30)
                time.sleep(attempt * 3.0)
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    f"gemini embedding returned HTTP {response.status_code}: {response.text[:200]}"
                )

            payload = response.json()
            out: list[list[float]] = []
            for item in payload.get("embeddings", []):
                values = [float(v) for v in item.get("values", [])]
                if len(values) != self.dim:
                    raise ProviderError(
                        f"gemini returned {len(values)} dimensions, expected {self.dim}"
                    )
                out.append(_normalise(values))
            if len(out) != len(texts):
                raise ProviderError(
                    f"gemini returned {len(out)} embeddings for {len(texts)} inputs"
                )
            return out
        raise ProviderError("gemini embedding failed after retries")


def _normalise(vector: list[float]) -> list[float]:
    """L2-normalise, which Matryoshka-truncated outputs require for cosine."""
    magnitude = sum(v * v for v in vector) ** 0.5
    if magnitude == 0:
        return vector
    return [v / magnitude for v in vector]
