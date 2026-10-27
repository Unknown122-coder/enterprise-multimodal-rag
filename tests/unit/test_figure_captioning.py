"""Figure captioning must survive a truncated or slow vision reply.

On the first real 36-page document, two of three caption batches came back as
truncated JSON (the budget was 900 tokens for four detailed descriptions) and
six of ten figures silently fell back to their captions. A batch that fails is
now retried one image at a time, each vision request carries its own timeout,
and the quality report says how many figures still lack a description.
"""

from __future__ import annotations

import json
import time

import pytest

from emrag.core.types import Element
from emrag.ingestion.figures import VISION_TIMEOUT_S, FigureDescription, describe_figures
from emrag.llm.base import Capabilities, Limits, LLMMessage, LLMRequest, ModelSpec
from emrag.llm.quota import QuotaLedger
from emrag.llm.router import LLMRouter, RouterConfig
from tests.conftest import ScriptedProvider

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _figure(index: int) -> Element:
    return Element(
        kind="figure",
        page=index + 1,
        bbox=[0.1, 0.1, 0.5, 0.5],
        text=f"Figure {index}: caption {index}",
        image_bytes_ref=f"p{index:03d}.png",
        element_id=f"f{index}",
        order=index,
    )


def _caption(index: int, text: str) -> dict[str, object]:
    return {
        "index": index,
        "type": "chart",
        "title": f"t{index}",
        "description": text,
        "axes": "",
        "data_points": [],
        "ocr_text": "",
        "insights": [],
    }


def _router(provider: ScriptedProvider) -> LLMRouter:
    spec = ModelSpec(
        key="gemini/vision",
        provider="gemini",
        model="vision-model",
        tiers=["vision"],
        priority=10,
        limits=Limits(rpm=30, tpm=100_000, rpd=100, tpd=1_000_000),
        capabilities=Capabilities(json_schema=True, vision=True, context=100_000),
    )
    ledger = QuotaLedger({spec.key: spec.limits})
    return LLMRouter([spec], {"gemini": provider}, ledger, RouterConfig({}))


@pytest.fixture
def figures() -> tuple[list[Element], dict[str, bytes]]:
    elements = [_figure(i) for i in range(3)]
    images = {e.image_bytes_ref or "": PNG + bytes([i]) for i, e in enumerate(elements)}
    return elements, images


async def test_batch_success_needs_one_call(
    figures: tuple[list[Element], dict[str, bytes]],
) -> None:
    elements, images = figures
    provider = ScriptedProvider(
        name="gemini",
        responses=[json.dumps({"figures": [_caption(i, f"desc {i}") for i in range(3)]})],
    )
    out = await describe_figures(_router(provider), elements, images, batch_size=3)
    assert [d.description for d in out] == ["desc 0", "desc 1", "desc 2"]
    assert not any(d.fallback for d in out)
    assert len(provider.requests) == 1


async def test_truncated_batch_is_retried_per_image(
    figures: tuple[list[Element], dict[str, bytes]],
) -> None:
    elements, images = figures
    truncated = '{"figures": [{"index": 0, "type": "chart", "description": "cut off'
    provider = ScriptedProvider(
        name="gemini",
        responses=[
            truncated,
            json.dumps({"figures": [_caption(0, "single 0")]}),
            truncated,  # the second image fails even on its own
            json.dumps({"figures": [_caption(0, "single 2")]}),
        ],
    )
    out = await describe_figures(_router(provider), elements, images, batch_size=3)
    assert len(provider.requests) == 4
    assert [len(r.images) for r in provider.requests] == [3, 1, 1, 1]
    assert out[0].description == "single 0" and not out[0].fallback
    assert out[2].description == "single 2" and not out[2].fallback
    # The one that failed twice keeps its caption and is flagged, not decorative.
    assert out[1].description == "Figure 1: caption 1"
    assert out[1].fallback is True
    assert out[1].decorative is False


async def test_single_image_batch_is_not_retried(
    figures: tuple[list[Element], dict[str, bytes]],
) -> None:
    elements, images = figures
    provider = ScriptedProvider(name="gemini", responses=["not json"])
    out = await describe_figures(_router(provider), elements[:1], images, batch_size=1)
    assert len(provider.requests) == 1
    assert out[0].fallback is True


async def test_vision_requests_carry_a_longer_timeout(
    figures: tuple[list[Element], dict[str, bytes]],
) -> None:
    elements, images = figures
    provider = ScriptedProvider(name="gemini", responses=[json.dumps({"figures": []})])
    await describe_figures(_router(provider), elements, images, batch_size=3)
    request = provider.requests[0]
    assert request.timeout_s == VISION_TIMEOUT_S
    assert request.max_tokens >= 2000


class TestRouterTimeoutOverride:
    """The router must honour a per-request timeout but never exceed the deadline."""

    def _router(self) -> LLMRouter:
        return LLMRouter([], {}, QuotaLedger({}), RouterConfig({"request_timeout_s": 30}))

    def test_default_without_override(self) -> None:
        assert self._router()._timeout_for(None, 0.0) == 30.0

    def test_override_wins_without_deadline(self) -> None:
        assert self._router()._timeout_for(None, 0.0, 90.0) == 90.0

    def test_deadline_still_caps_override(self) -> None:
        started = time.monotonic()
        assert self._router()._timeout_for(10_000, started, 90.0) <= 10.0

    def test_request_rejects_absurd_timeout(self) -> None:
        messages = [LLMMessage(role="user", content="x")]
        assert LLMRequest(messages=messages, timeout_s=90).timeout_s == 90
        with pytest.raises(ValueError):
            LLMRequest(messages=messages, timeout_s=0)
        with pytest.raises(ValueError):
            LLMRequest(messages=messages, timeout_s=1000)


def test_fallback_flag_defaults_false() -> None:
    assert FigureDescription(image_key="k", page=1, bbox=[0, 0, 1, 1]).fallback is False
