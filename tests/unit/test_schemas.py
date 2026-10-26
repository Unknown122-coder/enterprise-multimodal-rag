"""Every JSON schema sent to a provider must satisfy strict output modes.

Groq rejects a schema outright unless each object lists every property in
``required`` and forbids extras. That rejection surfaced in production as
"all models unavailable", because the router treated it like any other provider
failure and fell through every candidate. This test makes the rule a build
failure instead of a runtime one.
"""

from __future__ import annotations

from typing import Any

import pytest

from emrag.citation.contract import ANSWER_SCHEMA, REPAIR_SCHEMA, STREAM_TRAILER_SCHEMA
from emrag.eval.judges import FAITHFULNESS_SCHEMA, RELEVANCE_SCHEMA
from emrag.ingestion.figures import CAPTION_SCHEMA
from emrag.llm.gemini_provider import _to_gemini_schema
from emrag.retrieval.rewrite import REWRITE_SCHEMA

ALL_SCHEMAS = {
    "answer": ANSWER_SCHEMA,
    "repair": REPAIR_SCHEMA,
    "stream_trailer": STREAM_TRAILER_SCHEMA,
    "rewrite": REWRITE_SCHEMA,
    "caption": CAPTION_SCHEMA,
    "faithfulness": FAITHFULNESS_SCHEMA,
    "relevance": RELEVANCE_SCHEMA,
}


def _violations(schema: dict[str, Any], path: str = "$") -> list[str]:
    problems: list[str] = []
    if schema.get("type") == "object":
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        if set(props) != required:
            problems.append(f"{path}: required={sorted(required)} properties={sorted(props)}")
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties must be false")
        for name, sub in props.items():
            problems.extend(_violations(sub, f"{path}.{name}"))
    if isinstance(schema.get("items"), dict):
        problems.extend(_violations(schema["items"], f"{path}[]"))
    return problems


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
def test_schema_is_strict_mode_compliant(name: str) -> None:
    assert _violations(ALL_SCHEMAS[name]) == []


def test_optional_answer_fields_are_nullable_not_absent() -> None:
    # The model may have nothing to say in `notes`; strict mode still requires
    # the key, so the type must admit null rather than letting the key vanish.
    assert ANSWER_SCHEMA["properties"]["notes"]["type"] == ["string", "null"]
    assert "notes" in ANSWER_SCHEMA["required"]


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
def test_gemini_dialect_conversion_keeps_every_property(name: str) -> None:
    def props(schema: dict[str, Any]) -> set[str]:
        out = set(schema.get("properties") or {})
        for sub in (schema.get("properties") or {}).values():
            out |= {f"{k}" for k in props(sub)}
        if isinstance(schema.get("items"), dict):
            out |= props(schema["items"])
        return out

    converted = _to_gemini_schema(ALL_SCHEMAS[name])
    assert props(converted) == props(ALL_SCHEMAS[name])
    assert "additionalProperties" not in converted


def test_gemini_dialect_maps_nullable_union_types() -> None:
    converted = _to_gemini_schema(
        {"type": "object", "properties": {"n": {"type": ["string", "null"]}}}
    )
    assert converted["properties"]["n"] == {"type": "STRING", "nullable": True}
