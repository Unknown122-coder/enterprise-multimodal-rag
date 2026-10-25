"""LLM router.

Chooses a model for a requested tier, honouring the quota ledger, then falls
through to the next candidate on quota exhaustion, rate limiting or provider
failure. Candidates are ordered by the ``priority`` field in
``configs/providers.yaml`` (lower first).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from emrag.core.errors import ModelUnavailable, ProviderError, QuotaExhausted, RateLimited
from emrag.core.types import ModelTier
from emrag.llm.base import Capabilities, Limits, LLMProvider, LLMRequest, LLMResponse, ModelSpec
from emrag.llm.quota import QuotaLedger
from emrag.observability.logging import get_logger
from emrag.settings import Settings

logger = get_logger(__name__)


class RouterConfig:
    """Routing knobs read from the provider catalogue."""

    __slots__ = (
        "connect_timeout_s",
        "cooldown_seconds",
        "daily_reserve_ratio",
        "max_attempts_per_tier",
        "request_timeout_s",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.daily_reserve_ratio = float(raw.get("daily_reserve_ratio", 0.10))
        self.max_attempts_per_tier = int(raw.get("max_attempts_per_tier", 3))
        self.cooldown_seconds = int(raw.get("cooldown_seconds", 60))
        self.request_timeout_s = float(raw.get("request_timeout_s", 30.0))
        self.connect_timeout_s = float(raw.get("connect_timeout_s", 5.0))


class LLMRouter:
    """Tier-aware model selection with quota accounting and fallback."""

    def __init__(
        self,
        specs: list[ModelSpec],
        providers: dict[str, LLMProvider],
        ledger: QuotaLedger,
        config: RouterConfig,
    ) -> None:
        self._specs = specs
        self._providers = providers
        self._ledger = ledger
        self._config = config

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_settings(
        cls, settings: Settings, *, providers: dict[str, LLMProvider] | None = None
    ) -> LLMRouter:
        """Build a router from ``configs/providers.yaml`` and the environment.

        ``providers`` may be supplied by tests to inject fakes.
        """
        catalogue = settings.provider_catalogue
        config = RouterConfig(catalogue.get("routing", {}))
        specs = [_parse_spec(entry) for entry in catalogue.get("models", [])]

        if providers is None:
            providers = _build_providers(settings, catalogue, config)

        available = set(providers)
        usable = [s for s in specs if s.provider in available]
        if not usable:
            logger.warning("llm.router.no_providers", configured=sorted(available))

        state_path = Path(settings.artifact_local_root) / "state" / "usage.json"
        ledger = QuotaLedger(
            {s.key: s.limits for s in usable},
            daily_reserve_ratio=config.daily_reserve_ratio,
            cooldown_seconds=config.cooldown_seconds,
            state_path=state_path if settings.app_env != "test" else None,
        )
        return cls(usable, providers, ledger, config)

    # ---- introspection ------------------------------------------------------
    @property
    def ledger(self) -> QuotaLedger:
        """The quota ledger backing this router."""
        return self._ledger

    def models_by_tier(self) -> dict[str, list[str]]:
        """Return the configured model keys grouped by tier."""
        out: dict[str, list[str]] = {}
        for spec in sorted(self._specs, key=lambda s: s.priority):
            for tier in spec.tiers:
                out.setdefault(tier, []).append(spec.key)
        return out

    def candidates(self, tier: ModelTier, *, require_vision: bool = False) -> list[ModelSpec]:
        """Return models serving ``tier``, best priority first."""
        pool = [s for s in self._specs if tier in s.tiers]
        if require_vision:
            pool = [s for s in pool if s.capabilities.vision]
        return sorted(pool, key=lambda s: (s.priority, s.key))

    def has_tier(self, tier: ModelTier) -> bool:
        """True when at least one model serves ``tier``."""
        return bool(self.candidates(tier))

    # ---- generation ---------------------------------------------------------
    async def complete(
        self,
        tier: ModelTier,
        request: LLMRequest,
        *,
        require_vision: bool = False,
        deadline_ms: int | None = None,
    ) -> LLMResponse:
        """Generate a completion on the first candidate with available quota."""
        pool = self.candidates(tier, require_vision=require_vision)
        if not pool:
            raise ModelUnavailable(
                f"no model is configured for tier {tier!r}"
                + (" with vision support" if require_vision else "")
                + ". Set GROQ_API_KEY or GEMINI_API_KEY, and check that "
                "configs/providers.yaml lists a model for this tier."
            )

        est = request.estimated_prompt_tokens() + request.max_tokens
        started = time.monotonic()
        blocked: list[str] = []
        retry_hint = 0
        attempts = 0

        for spec in pool:
            if attempts >= self._config.max_attempts_per_tier:
                break
            if deadline_ms is not None:
                elapsed_ms = (time.monotonic() - started) * 1000
                if elapsed_ms > deadline_ms:
                    break

            decision = await self._ledger.reserve(spec.key, est)
            if not decision:
                blocked.append(f"{spec.key}:{decision.reason}")
                retry_hint = max(retry_hint, decision.retry_after_s)
                continue

            attempts += 1
            provider = self._providers[spec.provider]
            timeout_s = self._timeout_for(deadline_ms, started)
            try:
                response = await provider.complete(spec, request, timeout_s=timeout_s)
            except RateLimited as exc:
                await self._ledger.release(spec.key, est)
                await self._ledger.cool(spec.key, exc.retry_after_s)
                blocked.append(f"{spec.key}:429")
                retry_hint = max(retry_hint, exc.retry_after_s or 30)
                continue
            except ProviderError as exc:
                await self._ledger.release(spec.key, est)
                await self._ledger.cool(spec.key)
                logger.warning("llm.provider_failed", model=spec.key, error=str(exc))
                blocked.append(f"{spec.key}:error")
                continue

            await self._ledger.commit(
                spec.key,
                response.prompt_tokens,
                response.completion_tokens,
                reserved=est,
            )
            logger.info(
                "llm.call",
                tier=tier,
                model=spec.key,
                provider=spec.provider,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
                fallbacks=len(blocked),
            )
            return response

        raise QuotaExhausted(
            f"all models for tier {tier!r} are unavailable ({', '.join(blocked) or 'none tried'})",
            retry_after_s=retry_hint or 60,
        )

    async def stream(
        self,
        tier: ModelTier,
        request: LLMRequest,
        *,
        deadline_ms: int | None = None,
    ) -> tuple[ModelSpec, AsyncIterator[str]]:
        """Pick a model and return it together with its text-delta iterator.

        Quota is committed with an estimate once the stream is chosen, because
        streaming responses do not always carry a usage block.
        """
        pool = self.candidates(tier)
        if not pool:
            raise ModelUnavailable(
                f"no model is configured for tier {tier!r}. Set GROQ_API_KEY or "
                "GEMINI_API_KEY, and check configs/providers.yaml."
            )

        est = request.estimated_prompt_tokens() + request.max_tokens
        started = time.monotonic()
        blocked: list[str] = []
        retry_hint = 0

        for spec in pool:
            decision = await self._ledger.reserve(spec.key, est)
            if not decision:
                blocked.append(f"{spec.key}:{decision.reason}")
                retry_hint = max(retry_hint, decision.retry_after_s)
                continue
            provider = self._providers[spec.provider]
            timeout_s = self._timeout_for(deadline_ms, started)
            await self._ledger.commit(
                spec.key, request.estimated_prompt_tokens(), request.max_tokens, reserved=est
            )
            logger.info("llm.stream", tier=tier, model=spec.key, fallbacks=len(blocked))
            return spec, provider.stream(spec, request, timeout_s=timeout_s)

        raise QuotaExhausted(
            f"all models for tier {tier!r} are unavailable ({', '.join(blocked) or 'none tried'})",
            retry_after_s=retry_hint or 60,
        )

    async def aclose(self) -> None:
        """Flush the ledger and close every provider."""
        await self._ledger.flush()
        for provider in self._providers.values():
            await provider.aclose()

    def _timeout_for(self, deadline_ms: int | None, started: float) -> float:
        if deadline_ms is None:
            return self._config.request_timeout_s
        remaining = (deadline_ms / 1000.0) - (time.monotonic() - started)
        return max(1.0, min(self._config.request_timeout_s, remaining))


def _parse_spec(entry: dict[str, Any]) -> ModelSpec:
    """Build a :class:`ModelSpec` from one catalogue entry."""
    return ModelSpec(
        key=entry["key"],
        provider=entry["provider"],
        model=entry["model"],
        tiers=list(entry.get("tiers", [])),
        priority=int(entry.get("priority", 100)),
        limits=Limits(**(entry.get("limits") or {})),
        capabilities=Capabilities(**(entry.get("capabilities") or {})),
    )


def _build_providers(
    settings: Settings, catalogue: dict[str, Any], config: RouterConfig
) -> dict[str, LLMProvider]:
    """Instantiate every provider whose API key is present."""
    from emrag.llm.gemini_provider import GeminiProvider
    from emrag.llm.groq_provider import GroqProvider

    out: dict[str, LLMProvider] = {}
    defs = catalogue.get("providers", {})
    key_lookup = {"GROQ_API_KEY": settings.groq_api_key, "GEMINI_API_KEY": settings.gemini_api_key}

    for name, meta in defs.items():
        api_key = key_lookup.get(meta.get("api_key_env", ""))
        if not api_key:
            continue
        base_url = meta.get("base_url")
        if name == "groq":
            out[name] = GroqProvider(
                api_key, base_url=base_url, connect_timeout_s=config.connect_timeout_s
            )
        elif name == "gemini":
            out[name] = GeminiProvider(
                api_key, base_url=base_url, connect_timeout_s=config.connect_timeout_s
            )
        else:
            logger.warning("llm.router.unknown_provider", provider=name)
    return out
