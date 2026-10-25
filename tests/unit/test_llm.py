"""Unit tests for the LLM providers, quota ledger, router fallback and budgeting.

Provider tests mock HTTP with respx, so retry, 429 handling and schema plumbing
are asserted against real request bodies without touching a network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from emrag.core.errors import ModelUnavailable, ProviderError, QuotaExhausted, RateLimited
from emrag.core.types import Candidate, Chunk, Parent
from emrag.llm.base import Capabilities, Limits, LLMMessage, LLMRequest, ModelSpec
from emrag.llm.gemini_provider import GeminiProvider, _to_gemini_schema
from emrag.llm.groq_provider import GroqProvider
from emrag.llm.quota import QuotaLedger
from emrag.llm.router import LLMRouter, RouterConfig
from emrag.retrieval.budget import deduplicate, select_evidence

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)


def _spec(
    provider: str = "groq", model: str = "llama-3.3-70b-versatile", **kw: object
) -> ModelSpec:
    return ModelSpec(
        key=f"{provider}/{model}",
        provider=provider,
        model=model,
        tiers=["fast"],
        priority=int(kw.get("priority", 10)),  # type: ignore[arg-type]
        limits=kw.get("limits", Limits(rpm=30, tpm=10_000, rpd=100, tpd=100_000)),  # type: ignore[arg-type]
        capabilities=Capabilities(json_schema=True, vision=False, context=131072),
    )


def _request(text: str = "hello") -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content=text)], max_tokens=64)


def _groq_payload(content: str = "ok") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


class TestGroqProvider:
    """Groq transport, retry and schema behaviour."""

    @respx.mock
    async def test_successful_completion(self) -> None:
        route = respx.post(GROQ_URL).mock(
            return_value=httpx.Response(200, json=_groq_payload("the answer"))
        )
        provider = GroqProvider("key")
        response = await provider.complete(_spec(), _request(), timeout_s=5)
        await provider.aclose()
        assert response.text == "the answer"
        assert response.prompt_tokens == 11
        assert response.completion_tokens == 7
        assert response.provider == "groq"
        assert route.called

    @respx.mock
    async def test_json_schema_is_sent_when_supported(self) -> None:
        route = respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_payload()))
        provider = GroqProvider("key")
        request = _request()
        request.json_schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        request.schema_name = "thing"
        await provider.complete(_spec(), request, timeout_s=5)
        await provider.aclose()
        body = json.loads(route.calls[0].request.content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "thing"
        assert body["response_format"]["json_schema"]["strict"] is True

    @respx.mock
    async def test_retries_then_succeeds(self) -> None:
        respx.post(GROQ_URL).mock(
            side_effect=[
                httpx.Response(503, text="unavailable"),
                httpx.Response(200, json=_groq_payload("second try")),
            ]
        )
        provider = GroqProvider("key")
        response = await provider.complete(_spec(), _request(), timeout_s=5)
        await provider.aclose()
        assert response.text == "second try"

    @respx.mock
    async def test_persistent_429_raises_rate_limited(self) -> None:
        respx.post(GROQ_URL).mock(
            return_value=httpx.Response(429, text="slow down", headers={"retry-after": "7"})
        )
        provider = GroqProvider("key")
        with pytest.raises(RateLimited) as caught:
            await provider.complete(_spec(), _request(), timeout_s=5)
        await provider.aclose()
        assert caught.value.retry_after_s == 7

    @respx.mock
    async def test_client_error_is_not_retried(self) -> None:
        route = respx.post(GROQ_URL).mock(return_value=httpx.Response(400, text="bad model"))
        provider = GroqProvider("key")
        with pytest.raises(ProviderError):
            await provider.complete(_spec(), _request(), timeout_s=5)
        await provider.aclose()
        assert route.call_count == 1

    @respx.mock
    async def test_streaming_yields_deltas(self) -> None:
        body = (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(GROQ_URL).mock(return_value=httpx.Response(200, text=body))
        provider = GroqProvider("key")
        pieces = [p async for p in provider.stream(_spec(), _request(), timeout_s=5)]
        await provider.aclose()
        assert "".join(pieces) == "Hello"

    def test_missing_key_is_rejected(self) -> None:
        with pytest.raises(ProviderError):
            GroqProvider("")


class TestGeminiProvider:
    """Gemini transport and schema dialect conversion."""

    @respx.mock
    async def test_successful_completion(self) -> None:
        payload = {
            "candidates": [
                {"content": {"parts": [{"text": "gemini says"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4},
        }
        route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json=payload))
        provider = GeminiProvider("key")
        response = await provider.complete(
            _spec("gemini", "gemini-2.5-flash"), _request(), timeout_s=5
        )
        await provider.aclose()
        assert response.text == "gemini says"
        assert response.prompt_tokens == 9
        assert route.called

    @respx.mock
    async def test_system_message_becomes_system_instruction(self) -> None:
        payload = {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}
        route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json=payload))
        provider = GeminiProvider("key")
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content="be terse"),
                LLMMessage(role="user", content="hi"),
            ]
        )
        await provider.complete(_spec("gemini", "gemini-2.5-flash"), request, timeout_s=5)
        await provider.aclose()
        body = json.loads(route.calls[0].request.content)
        assert body["systemInstruction"]["parts"][0]["text"] == "be terse"
        assert len(body["contents"]) == 1

    def test_schema_conversion_uppercases_types_and_strips_unsupported(self) -> None:
        converted = _to_gemini_schema(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["a"],
                "properties": {"a": {"type": "array", "items": {"type": "string"}}},
            }
        )
        assert converted["type"] == "OBJECT"
        assert "additionalProperties" not in converted
        assert converted["properties"]["a"]["type"] == "ARRAY"
        assert converted["properties"]["a"]["items"]["type"] == "STRING"
        assert converted["required"] == ["a"]


class TestQuotaLedger:
    """Quota accounting must block before an upstream 429 happens."""

    async def test_allows_within_limits(self) -> None:
        ledger = QuotaLedger({"m": Limits(rpm=2, tpm=1000, rpd=10, tpd=10_000)})
        assert await ledger.reserve("m", 100)
        assert await ledger.reserve("m", 100)

    async def test_blocks_on_requests_per_minute(self) -> None:
        ledger = QuotaLedger({"m": Limits(rpm=1)})
        assert await ledger.reserve("m", 10)
        decision = await ledger.reserve("m", 10)
        assert not decision
        assert decision.reason == "rpm"
        assert decision.retry_after_s > 0

    async def test_blocks_on_tokens_per_minute(self) -> None:
        ledger = QuotaLedger({"m": Limits(tpm=500)})
        assert await ledger.reserve("m", 400)
        assert not await ledger.reserve("m", 200)

    async def test_daily_reserve_is_applied(self) -> None:
        ledger = QuotaLedger({"m": Limits(rpd=10)}, daily_reserve_ratio=0.5)
        for _ in range(5):
            assert await ledger.reserve("m", 1)
        decision = await ledger.reserve("m", 1)
        assert not decision
        assert decision.reason == "rpd"

    async def test_cooling_blocks_a_model(self) -> None:
        ledger = QuotaLedger({"m": Limits(rpm=100)})
        await ledger.cool("m", 30)
        decision = await ledger.reserve("m", 1)
        assert not decision
        assert decision.reason == "cooling"

    async def test_commit_releases_the_reservation_it_replaces(self) -> None:
        ledger = QuotaLedger({"m": Limits(tpm=1000)})
        await ledger.reserve("m", 900)
        await ledger.commit("m", 50, 20, reserved=900)
        # Only the measured 70 tokens remain charged, so a large call now fits.
        assert await ledger.reserve("m", 800)

    async def test_commit_without_reserved_charges_measured_usage(self) -> None:
        ledger = QuotaLedger({"m": Limits(tpm=1000)})
        await ledger.reserve("m", 100)
        await ledger.commit("m", 50, 20)
        snapshot = await ledger.snapshot()
        assert snapshot["models"]["m"]["tpd_used"] == 70

    async def test_release_frees_a_failed_reservation(self) -> None:
        ledger = QuotaLedger({"m": Limits(tpm=1000)})
        await ledger.reserve("m", 900)
        await ledger.release("m", 900)
        assert await ledger.reserve("m", 900)

    async def test_snapshot_reports_usage(self) -> None:
        ledger = QuotaLedger({"m": Limits(rpm=5, tpm=100, rpd=10, tpd=1000)})
        await ledger.reserve("m", 20)
        await ledger.commit("m", 10, 5)
        snapshot = await ledger.snapshot()
        assert snapshot["models"]["m"]["rpm_used"] == 1
        assert snapshot["models"]["m"]["tpd_used"] == 15

    async def test_unknown_model_is_refused(self) -> None:
        ledger = QuotaLedger({})
        assert not await ledger.reserve("nope", 1)

    async def test_persistence_round_trip(self, tmp_path: object) -> None:
        path = tmp_path / "usage.json"  # type: ignore[operator]
        first = QuotaLedger({"m": Limits(rpd=100)}, state_path=path)
        await first.reserve("m", 10)
        await first.commit("m", 5, 5)
        await first.flush()
        second = QuotaLedger({"m": Limits(rpd=100)}, state_path=path)
        snapshot = await second.snapshot()
        assert snapshot["models"]["m"]["rpd_used"] == 1
        assert snapshot["models"]["m"]["tpd_used"] == 10


class TestRouterFallback:
    """The router must degrade to another model rather than fail."""

    def _router(self, providers: dict[str, object], specs: list[ModelSpec]) -> LLMRouter:
        ledger = QuotaLedger({s.key: s.limits for s in specs})
        return LLMRouter(specs, providers, ledger, RouterConfig({}))  # type: ignore[arg-type]

    async def test_uses_highest_priority_model(self, scripted_provider: object) -> None:
        specs = [_spec(priority=20, model="second"), _spec(priority=10, model="first")]
        router = self._router({"groq": scripted_provider}, specs)
        scripted_provider.responses.append("picked")  # type: ignore[attr-defined]
        response = await router.complete("fast", _request())
        assert response.model == "first"

    async def test_falls_back_when_rate_limited(self, scripted_provider: object) -> None:
        from tests.conftest import ScriptedProvider

        failing = ScriptedProvider(name="groq", fail_times=5)
        working = ScriptedProvider(name="gemini", responses=["from gemini"])
        specs = [
            _spec("groq", "primary", priority=10),
            _spec("gemini", "secondary", priority=20),
        ]
        router = self._router({"groq": failing, "gemini": working}, specs)
        response = await router.complete("fast", _request())
        assert response.text == "from gemini"
        assert response.provider == "gemini"

    async def test_quota_exhausted_when_every_model_blocked(
        self, scripted_provider: object
    ) -> None:
        # A configured but spent model is a different condition, and carries a
        # retry hint the caller can act on.
        specs = [_spec(limits=Limits(rpm=0))]
        router = self._router({"groq": scripted_provider}, specs)
        with pytest.raises(QuotaExhausted) as caught:
            await router.complete("fast", _request())
        assert caught.value.code == "QUOTA_EXHAUSTED"
        assert caught.value.retry_after_s

    async def test_unconfigured_tier_is_not_reported_as_quota(
        self, scripted_provider: object
    ) -> None:
        # Waiting does not help when no model is configured, so the error must
        # be distinguishable from an exhausted budget.
        router = self._router({"groq": scripted_provider}, [_spec()])
        with pytest.raises(ModelUnavailable) as caught:
            await router.complete("quality", _request())
        assert "no model is configured" in str(caught.value)
        assert caught.value.code == "MODEL_UNAVAILABLE"

    async def test_vision_requirement_narrows_the_pool(self, scripted_provider: object) -> None:
        router = self._router({"groq": scripted_provider}, [_spec()])
        with pytest.raises(ModelUnavailable) as caught:
            await router.complete("fast", _request(), require_vision=True)
        assert "vision" in str(caught.value)

    async def test_models_by_tier_is_ordered_by_priority(self, scripted_provider: object) -> None:
        specs = [_spec(model="b", priority=20), _spec(model="a", priority=10)]
        router = self._router({"groq": scripted_provider}, specs)
        assert router.models_by_tier()["fast"] == ["groq/a", "groq/b"]


class TestContextBudget:
    """The budgeter deduplicates, respects the token allowance and expands parents."""

    def _candidate(self, text: str, score: float, seq: int = 1, parent: int = 5) -> Candidate:
        from emrag.core.ids import chunk_id as make_chunk_id
        from emrag.core.ids import doc_id_for_bytes
        from emrag.core.ids import parent_id as make_parent_id
        from emrag.core.textnorm import text_hash

        doc_id = doc_id_for_bytes(b"budget")
        chunk = Chunk(
            chunk_id=make_chunk_id(doc_id, 1, seq),
            doc_id=doc_id,
            tenant_id="demo",
            filename="f.pdf",
            page=1,
            page_count=1,
            bbox=[0.0, 0.0, 1.0, 1.0],
            modality="text",
            text=text,
            text_hash=text_hash(text),
            token_count=max(1, len(text) // 4),
            parent_id=make_parent_id(doc_id, parent),
            acl_groups=["all"],
            ingest_version="1.0.0",
            pipeline_profile="full",
            embed_model="t",
            sparse_model="t",
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        return Candidate(chunk=chunk, score=score, rerank_score=score)

    def test_deduplicates_near_identical_text(self) -> None:
        text = "Segment revenue grew fourteen percent year over year in the quarter."
        kept, dropped = deduplicate([self._candidate(text, 0.9, 1), self._candidate(text, 0.8, 2)])
        assert len(kept) == 1
        assert dropped == 1

    def test_keeps_distinct_passages(self) -> None:
        kept, dropped = deduplicate(
            [
                self._candidate("Revenue grew fourteen percent in the third quarter.", 0.9, 1),
                self._candidate("Headcount reached four hundred and twelve employees.", 0.8, 2),
            ]
        )
        assert len(kept) == 2
        assert dropped == 0

    def test_assigns_sequential_aliases(self) -> None:
        candidates = [
            self._candidate(f"Distinct passage number {i} about topic {i}.", 1.0 - i / 10, i + 1)
            for i in range(3)
        ]
        result = select_evidence(candidates, token_budget=2000, top_k=3)
        assert [b.alias for b in result.blocks] == ["c1", "c2", "c3"]

    def test_respects_the_token_budget(self) -> None:
        candidates = [
            self._candidate(f"Passage {i}. " + "filler words here " * 60, 1.0 - i / 10, i + 1)
            for i in range(8)
        ]
        result = select_evidence(candidates, token_budget=400, top_k=8)
        assert result.tokens_used <= 400
        assert len(result.blocks) < 8
        assert result.dropped_budget >= 1

    def test_parent_expansion_adds_section_context(self) -> None:
        from emrag.core.ids import doc_id_for_bytes
        from emrag.core.ids import parent_id as make_parent_id

        doc_id = doc_id_for_bytes(b"budget")
        candidate = self._candidate("Child passage about segment revenue growth.", 0.9, 1)
        parents = {
            make_parent_id(doc_id, 5): Parent(
                parent_id=make_parent_id(doc_id, 5),
                doc_id=doc_id,
                tenant_id="demo",
                section_path=["Section"],
                page_start=1,
                page_end=1,
                text="Full section text providing a much broader context for the child passage.",
                token_count=16,
                child_ids=[candidate.chunk.chunk_id],
            )
        }
        result = select_evidence(
            [candidate], token_budget=2000, top_k=1, parents=parents, parent_expansion=True
        )
        assert any(block.is_parent for block in result.blocks)
        assert any(block.alias.startswith("s") for block in result.blocks)

    def test_parent_expansion_skipped_when_disabled(self) -> None:
        candidate = self._candidate("Child passage about segment revenue growth.", 0.9, 1)
        result = select_evidence(
            [candidate], token_budget=2000, top_k=1, parents={}, parent_expansion=False
        )
        assert all(not block.is_parent for block in result.blocks)
