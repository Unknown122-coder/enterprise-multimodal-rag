"""Gemini provider (Google AI Studio generateContent REST API).

Used for vision captioning during ingestion, as the volume tier for answers and as
the evaluation judge. Shares retry and 429 semantics with the Groq provider.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from emrag.core.errors import ProviderError, ProviderTimeout, RateLimited
from emrag.llm.base import LLMProvider, LLMRequest, LLMResponse, ModelSpec
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
# Keys that Gemini's responseSchema dialect does not accept.
_SCHEMA_STRIP = ("additionalProperties", "$schema", "$id", "title", "examples", "default")


class GeminiProvider(LLMProvider):
    """Async client for the Gemini generateContent API."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        client: httpx.AsyncClient | None = None,
        connect_timeout_s: float = 5.0,
    ) -> None:
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not configured")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=connect_timeout_s),
            headers={"Content-Type": "application/json"},
        )

    async def complete(
        self, spec: ModelSpec, request: LLMRequest, *, timeout_s: float
    ) -> LLMResponse:
        """Return a full completion, retrying transient failures."""
        body = self._build_body(spec, request)
        path = f"/models/{spec.model}:generateContent"
        started = time.perf_counter()
        data = await self._post_with_retry(path, body, timeout_s)
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = _extract_text(data)
        usage = data.get("usageMetadata") or {}
        candidates = data.get("candidates") or [{}]
        return LLMResponse(
            text=text,
            provider=self.name,
            model=spec.model,
            prompt_tokens=int(usage.get("promptTokenCount", 0)),
            # Thinking tokens count against the output budget and the quota,
            # so they belong in completion_tokens even though no text came of them.
            completion_tokens=int(usage.get("candidatesTokenCount", 0))
            + int(usage.get("thoughtsTokenCount", 0)),
            latency_ms=latency_ms,
            finish_reason=str(candidates[0].get("finishReason", "STOP")).lower(),
        )

    async def stream(
        self, spec: ModelSpec, request: LLMRequest, *, timeout_s: float
    ) -> AsyncIterator[str]:
        """Yield text deltas from the streamGenerateContent endpoint."""
        body = self._build_body(spec, request)
        url = f"{self._base_url}/models/{spec.model}:streamGenerateContent"
        params = {"key": self._api_key, "alt": "sse"}
        async with self._client.stream(
            "POST", url, json=body, params=params, timeout=timeout_s
        ) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                self._raise_for_status(resp.status_code, raw.decode("utf-8", "replace"), resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = _extract_text(chunk)
                if delta:
                    yield delta

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()

    # ---- internals ---------------------------------------------------------
    def _build_body(self, spec: ModelSpec, request: LLMRequest) -> dict[str, Any]:
        system_text = "\n\n".join(m.content for m in request.messages if m.role == "system")
        contents: list[dict[str, Any]] = []
        turns = [m for m in request.messages if m.role != "system"]
        for i, msg in enumerate(turns):
            parts: list[dict[str, Any]] = [{"text": msg.content}]
            if request.images and msg.role == "user" and i == len(turns) - 1:
                parts.extend(
                    {"inlineData": {"mimeType": img.mime, "data": img.data_b64}}
                    for img in request.images
                )
            contents.append({"role": "user" if msg.role == "user" else "model", "parts": parts})

        gen_config: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_tokens,
        }
        if request.stop:
            gen_config["stopSequences"] = request.stop
        gen_config.update(spec.params)
        if request.json_schema:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = _to_gemini_schema(request.json_schema)

        body: dict[str, Any] = {"contents": contents, "generationConfig": gen_config}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        return body

    async def _post_with_retry(
        self, path: str, body: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        params = {"key": self._api_key}
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = await self._client.post(url, json=body, params=params, timeout=timeout_s)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt == _MAX_ATTEMPTS:
                    raise ProviderTimeout(f"gemini request timed out after {timeout_s}s") from exc
                await asyncio.sleep(_backoff(attempt))
                continue
            except httpx.HTTPError as exc:
                raise ProviderError(f"gemini transport error: {exc}") from exc

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
        raise ProviderError(f"gemini request failed: {last_exc}")

    def _raise_for_status(self, status: int, text: str, resp: httpx.Response) -> None:
        snippet = text[:300]
        if status == 429:
            raise RateLimited(
                f"gemini rate limited: {snippet}", retry_after_s=int(_retry_after(resp) or 30)
            )
        raise ProviderError(f"gemini returned HTTP {status}: {snippet}")


def _extract_text(payload: dict[str, Any]) -> str:
    """Concatenate all text parts of the first candidate."""
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema subset into Gemini's responseSchema dialect."""
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _SCHEMA_STRIP:
            continue
        if key == "type" and isinstance(value, str):
            out["type"] = value.upper()
        elif key == "type" and isinstance(value, list):
            # JSON Schema's ["string", "null"] is Gemini's STRING + nullable.
            concrete = [v for v in value if v != "null"]
            out["type"] = str(concrete[0]).upper() if concrete else "STRING"
            if "null" in value:
                out["nullable"] = True
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        elif key == "enum":
            out["enum"] = value
        elif key == "required":
            out["required"] = value
        elif key in ("description", "nullable", "minItems", "maxItems"):
            out[key] = value
    return out


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
