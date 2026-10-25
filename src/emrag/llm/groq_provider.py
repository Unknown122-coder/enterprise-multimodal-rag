"""Groq provider (OpenAI-compatible chat completions).

Implements retry with exponential backoff + jitter, ``Retry-After`` handling for
429s, strict JSON schema output and SSE streaming.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from emrag.core.errors import ProviderError, RateLimited
from emrag.llm.base import ImagePart, LLMProvider, LLMRequest, LLMResponse, ModelSpec
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


class GroqProvider(LLMProvider):
    """Async client for the Groq chat-completions API."""

    name = "groq"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.groq.com/openai/v1",
        client: httpx.AsyncClient | None = None,
        connect_timeout_s: float = 5.0,
    ) -> None:
        if not api_key:
            raise ProviderError("GROQ_API_KEY is not configured")
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=connect_timeout_s),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    async def complete(
        self, spec: ModelSpec, request: LLMRequest, *, timeout_s: float
    ) -> LLMResponse:
        """Return a full completion, retrying transient failures."""
        body = self._build_body(spec, request, stream=False)
        started = time.perf_counter()
        data = await self._post_with_retry("/chat/completions", body, timeout_s)
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            finish = choice.get("finish_reason") or "stop"
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"groq returned an unexpected payload: {exc}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=spec.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            finish_reason=finish,
        )

    async def stream(
        self, spec: ModelSpec, request: LLMRequest, *, timeout_s: float
    ) -> AsyncIterator[str]:
        """Yield text deltas from the streaming endpoint."""
        body = self._build_body(spec, request, stream=True)
        url = f"{self._base_url}/chat/completions"
        async with self._client.stream("POST", url, json=body, timeout=timeout_s) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                self._raise_for_status(resp.status_code, raw.decode("utf-8", "replace"), resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()

    # ---- internals ---------------------------------------------------------
    def _build_body(self, spec: ModelSpec, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for i, msg in enumerate(request.messages):
            is_last_user = msg.role == "user" and i == len(request.messages) - 1
            if request.images and is_last_user:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg.content},
                            *[_image_block(img) for img in request.images],
                        ],
                    }
                )
            else:
                messages.append({"role": msg.role, "content": msg.content})

        body: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.stop:
            body["stop"] = request.stop
        if request.json_schema and not stream:
            if spec.capabilities.json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.schema_name,
                        "schema": request.json_schema,
                        "strict": True,
                    },
                }
            else:
                body["response_format"] = {"type": "json_object"}
        return body

    async def _post_with_retry(
        self, path: str, body: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = await self._client.post(url, json=body, timeout=timeout_s)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt == _MAX_ATTEMPTS:
                    raise ProviderError(f"groq request timed out after {timeout_s}s") from exc
                await asyncio.sleep(_backoff(attempt))
                continue
            except httpx.HTTPError as exc:
                raise ProviderError(f"groq transport error: {exc}") from exc

            if resp.status_code < 400:
                result: dict[str, Any] = resp.json()
                return result
            if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS:
                delay = _retry_after(resp) or _backoff(attempt)
                logger.warning(
                    "llm.retry", provider=self.name, status=resp.status_code, delay_s=delay
                )
                await asyncio.sleep(delay)
                continue
            self._raise_for_status(resp.status_code, resp.text, resp)
        raise ProviderError(f"groq request failed: {last_exc}")

    def _raise_for_status(self, status: int, text: str, resp: httpx.Response) -> None:
        snippet = text[:300]
        if status == 429:
            raise RateLimited(
                f"groq rate limited: {snippet}", retry_after_s=int(_retry_after(resp) or 30)
            )
        raise ProviderError(f"groq returned HTTP {status}: {snippet}")


def _image_block(image: ImagePart) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image.mime};base64,{image.data_b64}"},
    }


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        seconds: float = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds < 60.0 else 60.0


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter, capped at 8 seconds."""
    # int ** int is Any to the type checker, so the base is bound as a float.
    base: float = 2.0 ** (attempt - 1)
    return min(8.0, base * 0.75) * (0.5 + random.random() / 2)
