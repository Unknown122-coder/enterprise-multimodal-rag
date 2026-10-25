"""Quota ledger.

Tracks requests-per-minute, tokens-per-minute, requests-per-day and
tokens-per-day for each ``(provider, model)`` pair against the limits declared in
``configs/providers.yaml``. The router consults it before every call so the system
degrades to another model instead of collecting HTTP 429s.

Daily counters are optionally persisted so a container restart does not reset a
day's usage to zero (worst case the ledger under-counts by one flush interval).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from emrag.llm.base import Limits

_MINUTE = 60.0


@dataclass
class _ModelState:
    """Sliding-window counters for one model."""

    limits: Limits
    req_times: deque[float] = field(default_factory=deque)
    tok_events: deque[tuple[float, int]] = field(default_factory=deque)
    day: str = ""
    day_requests: int = 0
    day_tokens: int = 0
    cooling_until: float = 0.0
    inflight_tokens: int = 0

    def prune(self, now: float) -> None:
        """Drop events older than the one-minute window."""
        cutoff = now - _MINUTE
        while self.req_times and self.req_times[0] < cutoff:
            self.req_times.popleft()
        while self.tok_events and self.tok_events[0][0] < cutoff:
            self.tok_events.popleft()

    def roll_day(self, today: str) -> None:
        """Reset daily counters when the UTC date changes."""
        if self.day != today:
            self.day = today
            self.day_requests = 0
            self.day_tokens = 0

    def minute_tokens(self) -> int:
        """Tokens consumed within the trailing minute, including in-flight."""
        return sum(t for _, t in self.tok_events) + self.inflight_tokens


class QuotaDecision:
    """Result of a reservation attempt."""

    __slots__ = ("allowed", "reason", "retry_after_s")

    def __init__(self, allowed: bool, reason: str = "", retry_after_s: int = 0) -> None:
        self.allowed = allowed
        self.reason = reason
        self.retry_after_s = retry_after_s

    def __bool__(self) -> bool:
        return self.allowed


class QuotaLedger:
    """In-process quota accounting with optional JSON persistence."""

    def __init__(
        self,
        limits_by_key: dict[str, Limits],
        *,
        daily_reserve_ratio: float = 0.10,
        cooldown_seconds: int = 60,
        state_path: Path | None = None,
    ) -> None:
        self._state: dict[str, _ModelState] = {
            key: _ModelState(limits=limits) for key, limits in limits_by_key.items()
        }
        self._reserve = daily_reserve_ratio
        self._cooldown = cooldown_seconds
        self._state_path = state_path
        self._lock = asyncio.Lock()
        self._dirty = False
        if state_path is not None:
            self._load()

    # ---- public API ---------------------------------------------------------
    async def reserve(self, key: str, est_tokens: int) -> QuotaDecision:
        """Reserve budget for a call, returning whether it may proceed."""
        async with self._lock:
            return self._reserve_locked(key, est_tokens)

    async def commit(
        self, key: str, prompt_tokens: int, completion_tokens: int, *, reserved: int = 0
    ) -> None:
        """Record actual usage, releasing the reservation it replaces.

        ``reserved`` is the estimate passed to :meth:`reserve` for this call. The
        full reservation is released and the measured usage recorded in its place,
        so a conservative estimate does not keep blocking later calls.
        """
        total = prompt_tokens + completion_tokens
        async with self._lock:
            st = self._state.get(key)
            if st is None:
                return
            now = time.monotonic()
            st.prune(now)
            st.inflight_tokens = max(0, st.inflight_tokens - (reserved or total))
            st.tok_events.append((now, total))
            st.roll_day(_today())
            st.day_tokens += total
            self._dirty = True

    async def release(self, key: str, est_tokens: int) -> None:
        """Release a reservation whose call failed before producing usage."""
        async with self._lock:
            st = self._state.get(key)
            if st is not None:
                st.inflight_tokens = max(0, st.inflight_tokens - est_tokens)

    async def cool(self, key: str, seconds: int | None = None) -> None:
        """Mark a model unavailable for a short period (after a 429 or 5xx)."""
        async with self._lock:
            st = self._state.get(key)
            if st is not None:
                st.cooling_until = time.monotonic() + (seconds or self._cooldown)

    async def snapshot(self) -> dict[str, Any]:
        """Return a serialisable view of current usage, for ``GET /v1/usage``."""
        async with self._lock:
            now = time.monotonic()
            today = _today()
            out: dict[str, Any] = {"window_start": _iso_now(), "models": {}}
            for key, st in self._state.items():
                st.prune(now)
                st.roll_day(today)
                out["models"][key] = {
                    "rpm_used": len(st.req_times),
                    "rpm_limit": st.limits.rpm,
                    "tpm_used": st.minute_tokens(),
                    "tpm_limit": st.limits.tpm,
                    "rpd_used": st.day_requests,
                    "rpd_limit": st.limits.rpd,
                    "tpd_used": st.day_tokens,
                    "tpd_limit": st.limits.tpd,
                    "cooling_s": max(0, int(st.cooling_until - now)),
                }
            return out

    async def flush(self) -> None:
        """Persist daily counters if a state path was configured."""
        if self._state_path is None or not self._dirty:
            return
        async with self._lock:
            payload = {
                "day": _today(),
                "models": {
                    k: {"day_requests": s.day_requests, "day_tokens": s.day_tokens}
                    for k, s in self._state.items()
                },
            }
            self._dirty = False
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            # Persistence is best effort: an ephemeral filesystem must not break serving.
            self._dirty = True

    # ---- internals ---------------------------------------------------------
    def _reserve_locked(self, key: str, est_tokens: int) -> QuotaDecision:
        st = self._state.get(key)
        if st is None:
            return QuotaDecision(False, f"unknown model key {key}")
        now = time.monotonic()
        if st.cooling_until > now:
            return QuotaDecision(False, "cooling", int(st.cooling_until - now) + 1)
        st.prune(now)
        st.roll_day(_today())
        lim = st.limits

        if lim.rpm is not None and len(st.req_times) >= lim.rpm:
            # An empty window means the limit itself is zero: wait a full window.
            wait = int(_MINUTE - (now - st.req_times[0])) + 1 if st.req_times else int(_MINUTE)
            return QuotaDecision(False, "rpm", wait)
        if lim.tpm is not None and st.minute_tokens() + est_tokens > lim.tpm:
            wait = int(_MINUTE - (now - st.tok_events[0][0])) + 1 if st.tok_events else 5
            return QuotaDecision(False, "tpm", wait)
        if lim.rpd is not None and st.day_requests >= self._daily_cap(lim.rpd):
            return QuotaDecision(False, "rpd", _seconds_to_utc_midnight())
        if lim.tpd is not None and st.day_tokens + est_tokens > self._daily_cap(lim.tpd):
            return QuotaDecision(False, "tpd", _seconds_to_utc_midnight())

        st.req_times.append(now)
        st.day_requests += 1
        st.inflight_tokens += est_tokens
        self._dirty = True
        return QuotaDecision(True)

    def _daily_cap(self, published: int) -> int:
        """Apply the safety reserve so a batch job cannot drain the day."""
        return max(1, int(published * (1.0 - self._reserve)))

    def _load(self) -> None:
        assert self._state_path is not None
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if data.get("day") != _today():
            return
        for key, vals in data.get("models", {}).items():
            st = self._state.get(key)
            if st is not None:
                st.day = _today()
                st.day_requests = int(vals.get("day_requests", 0))
                st.day_tokens = int(vals.get("day_tokens", 0))


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _seconds_to_utc_midnight() -> int:
    now = datetime.now(UTC)
    midnight = now.replace(hour=23, minute=59, second=59, microsecond=0)
    return max(1, int((midnight - now).total_seconds()))
