"""LLM provider protocol and wire types.

Providers are implemented directly over the documented REST endpoints with
``httpx`` rather than vendor SDKs. Assumption: this keeps the dependency surface
small, makes retry/429 behaviour explicit, and lets tests mock HTTP with respx
instead of patching SDK internals.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import Field

from emrag.core.types import ModelTier, StrictModel

Role = Literal["system", "user", "assistant"]


class ImagePart(StrictModel):
    """An inline image for vision requests."""

    mime: str = "image/png"
    data_b64: str


class LLMMessage(StrictModel):
    """One chat message."""

    role: Role
    content: str


class LLMRequest(StrictModel):
    """A provider-neutral generation request."""

    messages: list[LLMMessage] = Field(min_length=1)
    max_tokens: int = Field(default=700, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    json_schema: dict[str, Any] | None = None
    schema_name: str = "response"
    images: list[ImagePart] = Field(default_factory=list)
    stop: list[str] = Field(default_factory=list)

    def estimated_prompt_tokens(self) -> int:
        """Rough prompt size, used for quota reservation before the call."""
        from emrag.core.tokens import count_tokens

        text = "\n".join(m.content for m in self.messages)
        # ~258 tokens per inline image is a documented Gemini figure; close enough.
        return count_tokens(text) + 258 * len(self.images)


class LLMResponse(StrictModel):
    """A completed generation."""

    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str = "stop"


class Limits(StrictModel):
    """Published free-tier limits for one model."""

    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    tpd: int | None = None


class Capabilities(StrictModel):
    """Model capability flags."""

    json_schema: bool = False
    vision: bool = False
    context: int = 8192


class ModelSpec(StrictModel):
    """One routable model entry from ``configs/providers.yaml``."""

    key: str
    provider: str
    model: str
    tiers: list[ModelTier]
    priority: int = 100
    limits: Limits = Field(default_factory=Limits)
    capabilities: Capabilities = Field(default_factory=Capabilities)


class LLMProvider(abc.ABC):
    """Async provider interface."""

    name: str

    @abc.abstractmethod
    async def complete(
        self, spec: ModelSpec, request: LLMRequest, *, timeout_s: float
    ) -> LLMResponse:
        """Return a full completion for ``request``."""

    @abc.abstractmethod
    def stream(
        self, spec: ModelSpec, request: LLMRequest, *, timeout_s: float
    ) -> AsyncIterator[str]:
        """Yield text deltas for ``request``."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release network resources."""
