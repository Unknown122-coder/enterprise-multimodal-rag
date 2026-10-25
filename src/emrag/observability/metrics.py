"""In-process metrics.

Counters and latency histograms kept in memory and exposed to admins. A free-tier
single-instance deployment has nowhere to ship a time series, and this is enough
to answer the operational questions that matter: how many queries returned
not_found, how often claims were dropped, how often a provider fell back.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

_BUCKETS_MS = (100, 250, 500, 1000, 2000, 3000, 5000, 10000)


class Metrics:
    """Thread-safe counters and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[int]] = defaultdict(lambda: [0] * (len(_BUCKETS_MS) + 1))
        self._sums: dict[str, float] = defaultdict(float)
        self._started = time.time()

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add to a counter."""
        with self._lock:
            self._counters[_key(name, labels)] += value

    def observe(self, name: str, milliseconds: float, **labels: str) -> None:
        """Record a latency observation."""
        key = _key(name, labels)
        index = len(_BUCKETS_MS)
        for position, edge in enumerate(_BUCKETS_MS):
            if milliseconds <= edge:
                index = position
                break
        with self._lock:
            self._histograms[key][index] += 1
            self._sums[key] += milliseconds

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable view of every metric."""
        with self._lock:
            histograms = {}
            for key, buckets in self._histograms.items():
                total = sum(buckets)
                histograms[key] = {
                    "count": total,
                    "mean_ms": round(self._sums[key] / total, 1) if total else 0.0,
                    "buckets": dict(
                        zip(
                            [f"<={b}ms" for b in _BUCKETS_MS] + ["+Inf"],
                            buckets,
                            strict=True,
                        )
                    ),
                }
            return {
                "uptime_s": int(time.time() - self._started),
                "counters": dict(self._counters),
                "histograms": histograms,
            }

    def reset(self) -> None:
        """Clear every metric; used by tests."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._sums.clear()


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


METRICS = Metrics()
