"""The evaluation runner must survive a failing item, and a deadline must not cool a model.

The first full-mode run on a real corpus died on item 9 of 42 with nothing
written: one QuotaExhausted propagated out of the loop. The same run showed a
second problem: the interactive 4.5 s deadline forced a 2.6 s timeout on the
healthiest Gemini model, which the router then treated as a provider failure
and cooled for a minute, so the next item found every model unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from emrag.core.errors import ProviderTimeout, QuotaExhausted
from emrag.eval import run_eval
from emrag.eval.metrics import GoldItem
from emrag.llm.base import Capabilities, Limits, LLMMessage, LLMRequest, ModelSpec
from emrag.llm.quota import QuotaLedger
from emrag.llm.router import LLMRouter, RouterConfig
from emrag.retrieval.retriever import RetrievalOutcome
from tests.conftest import ScriptedProvider

# --------------------------------------------------------------------------- #
# router: deadline-forced timeouts
# --------------------------------------------------------------------------- #


@dataclass
class TimingOutProvider(ScriptedProvider):
    """Times out a given number of times, then answers."""

    timeouts: int = 0
    seen_timeouts: list[float] = field(default_factory=list)

    async def complete(self, spec: ModelSpec, request: LLMRequest, *, timeout_s: float) -> Any:
        if self.timeouts > 0:
            self.timeouts -= 1
            self.seen_timeouts.append(timeout_s)
            raise ProviderTimeout(f"timed out after {timeout_s}s")
        return await super().complete(spec, request, timeout_s=timeout_s)


def _spec(key: str = "groq/m", priority: int = 10) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider="groq",
        model=key.split("/")[1],
        tiers=["fast"],
        priority=priority,
        limits=Limits(rpm=60, tpm=100_000, rpd=1000, tpd=1_000_000),
        capabilities=Capabilities(json_schema=True, vision=False, context=100_000),
    )


def _request() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content="hi")], max_tokens=16)


class TestDeadlineTimeouts:
    async def test_deadline_forced_timeout_does_not_cool_the_model(self) -> None:
        provider = TimingOutProvider(name="groq", timeouts=1, responses=["ok"])
        spec = _spec()
        ledger = QuotaLedger({spec.key: spec.limits})
        router = LLMRouter(
            [spec], {"groq": provider}, ledger, RouterConfig({"request_timeout_s": 30})
        )

        with pytest.raises(QuotaExhausted) as excinfo:
            await router.complete("fast", _request(), deadline_ms=2000)
        assert provider.seen_timeouts and provider.seen_timeouts[0] <= 2.0
        assert "deadline" in str(excinfo.value)

        # The model is still in rotation: the very next call succeeds on it.
        response = await router.complete("fast", _request())
        assert response.text == "ok"

    async def test_genuine_timeout_still_cools_the_model(self) -> None:
        provider = TimingOutProvider(name="groq", timeouts=1, responses=["ok"])
        spec = _spec()
        ledger = QuotaLedger({spec.key: spec.limits})
        router = LLMRouter(
            [spec], {"groq": provider}, ledger, RouterConfig({"request_timeout_s": 30})
        )

        with pytest.raises(QuotaExhausted) as excinfo:
            await router.complete("fast", _request())
        assert ":error" in str(excinfo.value)
        decision = await ledger.reserve(spec.key, 10)
        assert not decision and decision.reason == "cooling"


# --------------------------------------------------------------------------- #
# eval runner: failed items
# --------------------------------------------------------------------------- #


class FlakyRetriever:
    """Raises a scripted sequence of exceptions before returning an outcome."""

    def __init__(self, failures: dict[str, list[Exception]]) -> None:
        self.failures = failures
        self.calls: list[str] = []

    async def retrieve(self, question: str, filters: Any, options: Any) -> RetrievalOutcome:
        self.calls.append(question)
        queue = self.failures.get(question)
        if queue:
            raise queue.pop(0)
        return RetrievalOutcome(candidates=[], gate_passed=False)


def _state(retriever: FlakyRetriever) -> Any:
    settings = SimpleNamespace(
        candidates_k=40,
        rerank_enabled=False,
        evidence_token_budget=2500,
        retrieval_gate_tau=0.05,
        vector_backend="memory",
        default_deadline_ms=4500,
    )
    return SimpleNamespace(
        settings=settings,
        retriever=retriever,
        embedder=SimpleNamespace(model_name="deterministic"),
        router=SimpleNamespace(has_tier=lambda tier: False),
    )


class TestRunnerResilience:
    async def test_quota_exhaustion_waits_and_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waits: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        monkeypatch.setattr(run_eval.asyncio, "sleep", fake_sleep)
        retriever = FlakyRetriever({"q1": [QuotaExhausted("tpm", retry_after_s=7)]})
        items = [GoldItem(id="a1", question="q1", answerable=False, tenant_id="demo")]

        report = await run_eval.evaluate(
            _state(retriever), items, mode="retrieval", top_k=8, judge=False
        )

        assert waits == [7.0]
        assert retriever.calls == ["q1", "q1"]
        assert report["errors"] == 0
        assert report["per_item"][0]["gate_passed"] is False

    async def test_other_failures_are_recorded_and_the_run_continues(self) -> None:
        retriever = FlakyRetriever({"q1": [RuntimeError("store exploded")]})
        items = [
            GoldItem(id="a1", question="q1", answerable=False, tenant_id="demo"),
            GoldItem(id="a2", question="q2", answerable=False, tenant_id="demo"),
        ]

        report = await run_eval.evaluate(
            _state(retriever), items, mode="retrieval", top_k=8, judge=False
        )

        assert report["errors"] == 1
        assert report["per_item"][0]["id"] == "a1"
        assert "store exploded" in report["per_item"][0]["error"]
        assert report["per_item"][1]["id"] == "a2"
        assert "error" not in report["per_item"][1]
        assert report["metrics"]["not_found_gate"] == 1.0  # a2 scored; a1 did not poison it

    async def test_persistent_quota_exhaustion_gives_up_after_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_sleep(seconds: float) -> None:
            pass

        monkeypatch.setattr(run_eval.asyncio, "sleep", fake_sleep)
        retriever = FlakyRetriever(
            {"q1": [QuotaExhausted("tpm", retry_after_s=1)] * (run_eval.QUOTA_RETRIES + 1)}
        )
        items = [GoldItem(id="a1", question="q1", answerable=False, tenant_id="demo")]

        report = await run_eval.evaluate(
            _state(retriever), items, mode="retrieval", top_k=8, judge=False
        )

        assert report["errors"] == 1
        assert len(retriever.calls) == run_eval.QUOTA_RETRIES + 1
