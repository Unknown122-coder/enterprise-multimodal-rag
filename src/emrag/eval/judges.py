"""LLM judges for faithfulness and answer relevance.

The judge is always a *different* model from the one that produced the answer,
routed through the ``judge`` tier, so a model is never asked to grade its own
work. Judging is the only evaluation step that spends quota, which is why it runs
nightly rather than per commit.
"""

from __future__ import annotations

import json
from typing import Any

from emrag.llm.base import LLMMessage, LLMRequest
from emrag.llm.router import LLMRouter
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

FAITHFULNESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entailed", "verdicts"],
    "properties": {
        "entailed": {"type": "integer"},
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "entailed"],
                "properties": {
                    "claim": {"type": "integer"},
                    "entailed": {"type": "boolean"},
                    "why": {"type": "string"},
                },
            },
        },
    },
}

RELEVANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score"],
    "properties": {
        "score": {"type": "integer"},
        "why": {"type": "string"},
    },
}

_FAITHFULNESS_SYSTEM = (
    "You check whether each claim of an answer is entailed by the quoted evidence given "
    "with it. A claim is entailed only if the quote states or directly implies it. Extra "
    "detail that the quote does not support makes a claim not entailed. Judge only "
    "entailment, not writing quality. Reply with JSON only."
)

_RELEVANCE_SYSTEM = (
    "You rate how well an answer addresses a question, on a scale of 1 to 5, where 5 "
    "fully answers it and 1 does not address it at all. A correct but incomplete answer "
    "scores 3. Ignore style. Reply with JSON only."
)


async def judge_faithfulness(
    router: LLMRouter, claims: list[dict[str, Any]], citations: list[dict[str, Any]]
) -> float:
    """Return the share of claims entailed by their cited quotes."""
    if not claims:
        return 1.0
    quotes_by_ref = {c.get("ref"): c.get("quote", "") for c in citations}
    lines: list[str] = []
    for index, claim in enumerate(claims):
        quotes = [
            quotes_by_ref.get(ref, "") for ref in claim.get("citation_refs", []) if ref is not None
        ]
        joined = " | ".join(q for q in quotes if q)
        lines.append(f"Claim {index}: {claim.get('text', '')}\nEvidence {index}: {joined}")

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=_FAITHFULNESS_SYSTEM),
            LLMMessage(role="user", content="\n\n".join(lines)),
        ],
        max_tokens=600,
        json_schema=FAITHFULNESS_SCHEMA,
        schema_name="faithfulness",
    )
    try:
        response = await router.complete("judge", request)
        payload = json.loads(response.text or "{}")
    except Exception as exc:  # noqa: BLE001 - a failed judge must not fail the run
        logger.warning("eval.judge_failed", metric="faithfulness", error=str(exc)[:200])
        return float("nan")

    verdicts = payload.get("verdicts") or []
    if verdicts:
        return sum(1 for v in verdicts if v.get("entailed")) / len(verdicts)
    return min(1.0, float(payload.get("entailed", 0)) / len(claims))


async def judge_relevance(router: LLMRouter, question: str, answer: str) -> float:
    """Return a 1 to 5 relevance score for an answer."""
    if not answer.strip():
        return 1.0
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=_RELEVANCE_SYSTEM),
            LLMMessage(role="user", content=f"Question: {question}\n\nAnswer: {answer}"),
        ],
        max_tokens=200,
        json_schema=RELEVANCE_SCHEMA,
        schema_name="relevance",
    )
    try:
        response = await router.complete("judge", request)
        payload = json.loads(response.text or "{}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("eval.judge_failed", metric="relevance", error=str(exc)[:200])
        return float("nan")
    return max(1.0, min(5.0, float(payload.get("score", 1))))
