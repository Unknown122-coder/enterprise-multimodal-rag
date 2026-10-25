"""Expose the LLM router to LlamaIndex.

The router is the only component that knows about free-tier quotas: it picks a
model with budget remaining, falls through on a 429, and keeps the daily ledger.
Anything that calls a provider directly bypasses that and will start collecting
rate-limit errors.

This adapter wraps the router as a LlamaIndex ``CustomLLM`` so LlamaIndex
components (its own query engines, node post-processors or evaluators) route
through the same ledger. It is optional: importing this module without
``llama-index-core`` installed raises, and nothing in the serving path imports it.

See ``docs/adr/0007-framework-deviation.md`` for why the rest of the system does
not depend on LlamaIndex.

Usage::

    from emrag.llm.llamaindex_adapter import RoutedLLM
    llm = RoutedLLM(router=state.router, tier="fast")
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from emrag.core.types import ModelTier
from emrag.llm.base import LLMMessage, LLMRequest
from emrag.llm.router import LLMRouter
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

try:
    from llama_index.core.llms import (
        CompletionResponse,
        CompletionResponseGen,
        CustomLLM,
        LLMMetadata,
    )
    from llama_index.core.llms.callbacks import llm_completion_callback
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "emrag.llm.llamaindex_adapter needs llama-index-core. "
        "Install it with: pip install llama-index-core"
    ) from exc


DEFAULT_CONTEXT_WINDOW = 131_072


class RoutedLLM(CustomLLM):
    """A LlamaIndex LLM that delegates to the quota-aware router.

    Attributes are declared as pydantic fields because ``CustomLLM`` is a
    pydantic model.
    """

    router: Any
    tier: str = "fast"
    max_tokens: int = 700
    temperature: float = 0.0
    system_prompt_text: str | None = None

    def __init__(
        self,
        *,
        router: LLMRouter,
        tier: ModelTier = "fast",
        max_tokens: int = 700,
        temperature: float = 0.0,
        system_prompt_text: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            router=router,
            tier=tier,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt_text=system_prompt_text,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Report the tier rather than a single model, since routing is dynamic."""
        candidates = self.router.candidates(self.tier)
        context = (
            max(c.capabilities.context for c in candidates)
            if candidates
            else DEFAULT_CONTEXT_WINDOW
        )
        return LLMMetadata(
            context_window=context,
            num_output=self.max_tokens,
            model_name=f"emrag-router:{self.tier}",
            is_chat_model=True,
        )

    def _build_request(self, prompt: str) -> LLMRequest:
        messages: list[LLMMessage] = []
        if self.system_prompt_text:
            messages.append(LLMMessage(role="system", content=self.system_prompt_text))
        messages.append(LLMMessage(role="user", content=prompt))
        return LLMRequest(
            messages=messages, max_tokens=self.max_tokens, temperature=self.temperature
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Synchronous completion.

        LlamaIndex calls this from sync code; the router is async, so the call is
        bridged. Inside a running loop this would deadlock, so :meth:`acomplete`
        must be used there instead, which LlamaIndex does for async pipelines.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            response = asyncio.run(self.router.complete(self.tier, self._build_request(prompt)))
            return CompletionResponse(text=response.text, raw=response.model_dump())
        raise RuntimeError(
            "RoutedLLM.complete() was called from inside an event loop; use acomplete() instead"
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Asynchronous completion, the preferred entry point."""
        response = await self.router.complete(self.tier, self._build_request(prompt))
        return CompletionResponse(text=response.text, raw=response.model_dump())

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming is not bridged to sync callers.

        The router's stream is async. Rather than pump an async generator from a
        sync one and risk a deadlock, this yields the full completion once.
        """
        response = self.complete(prompt)
        yield response

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        """Asynchronous streaming through the router."""
        _spec, deltas = await self.router.stream(self.tier, self._build_request(prompt))
        collected: list[str] = []

        async def _generator() -> Any:
            async for delta in deltas:
                collected.append(delta)
                yield CompletionResponse(text="".join(collected), delta=delta)

        return _generator()


def embedding_for_llamaindex(provider: Any) -> Any:
    """Wrap an :class:`~emrag.embeddings.base.EmbeddingProvider` for LlamaIndex.

    Returns a ``BaseEmbedding`` that reuses the configured embedder, so a
    LlamaIndex component cannot accidentally index with a different model than
    the one recorded on the chunks.
    """
    from llama_index.core.embeddings import BaseEmbedding

    class _RoutedEmbedding(BaseEmbedding):
        """Delegates to the project's embedding provider."""

        def _get_query_embedding(self, query: str) -> list[float]:
            return [float(v) for v in provider.embed_query(query).dense]

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return await asyncio.to_thread(self._get_query_embedding, query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return [float(v) for v in provider.embed_documents([text])[0].dense]

        def _get_text_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
            return [v.dense for v in provider.embed_documents(list(texts))]

    return _RoutedEmbedding(model_name=provider.model_name, embed_batch_size=32)
