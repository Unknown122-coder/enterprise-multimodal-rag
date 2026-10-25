"""LLM output contracts and prompt assembly.

The grammar the model must produce, the schemas that enforce it, and the evidence
rendering that makes aliases unambiguous. Two grammars exist because streaming
and strict JSON cannot both be had at once:

* non-streaming: a JSON object validated by :data:`ANSWER_SCHEMA`;
* streaming: markdown with ``[c3]`` markers, a terminator line, then a JSON
  trailer validated by :data:`STREAM_TRAILER_SCHEMA`.

Aliases (``c1``, ``c2``, ...) are used instead of chunk ids so the model cannot
invent a plausible-looking identifier, and so the prompt stays small.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emrag.core.types import Candidate
from emrag.retrieval.budget import EvidenceBlock
from emrag.security.injection import sanitise

PROMPT_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"

STREAM_TERMINATOR = "---CITATIONS---"
ALIAS_RE = re.compile(r"^[cs]\d{1,2}$")
MARKER_RE = re.compile(r"\[([cs]\d{1,2})\]")

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sufficient_evidence", "answer"],
    "properties": {
        "sufficient_evidence": {"type": "boolean"},
        "answer": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "quote"],
                            "properties": {
                                "id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "unanswered_aspects": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
}

STREAM_TRAILER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sufficient_evidence", "citations"],
    "properties": {
        "sufficient_evidence": {"type": "boolean"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "id", "quote"],
                "properties": {
                    "claim": {"type": "integer"},
                    "id": {"type": "string"},
                    "quote": {"type": "string"},
                },
            },
        },
    },
}

REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {
        "answer": ANSWER_SCHEMA["properties"]["answer"],
    },
}


@dataclass
class EvidenceBundle:
    """Rendered evidence plus the alias to chunk mapping used for verification."""

    prompt_text: str
    alias_to_candidate: dict[str, Candidate]
    parent_aliases: set[str]

    def chunk_ids(self) -> dict[str, str]:
        """Alias to chunk id mapping."""
        return {alias: c.chunk.chunk_id for alias, c in self.alias_to_candidate.items()}


@functools.lru_cache(maxsize=8)
def load_prompt(name: str) -> str:
    """Load a prompt template by filename stem."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_evidence(blocks: list[EvidenceBlock]) -> EvidenceBundle:
    """Render evidence blocks into prompt text with stable aliases.

    Block text is sanitised first: invisible characters and fake ``</evidence>``
    delimiters are removed so document content cannot break out of its own block.
    """
    parts: list[str] = []
    mapping: dict[str, Candidate] = {}
    parents: set[str] = set()

    for block in blocks:
        chunk = block.candidate.chunk
        mapping[block.alias] = block.candidate
        if block.is_parent:
            parents.add(block.alias)
        attributes = [
            f'id="{block.alias}"',
            f'doc="{_attr(chunk.filename)}"',
            f'page="{chunk.page}"',
            f'modality="{"section" if block.is_parent else chunk.modality}"',
        ]
        if chunk.section_path:
            attributes.append(f'section="{_attr(" > ".join(chunk.section_path))}"')
        if chunk.risk_flags:
            attributes.append('untrusted="true"')
        parts.append(
            f"<evidence {' '.join(attributes)}>\n{sanitise(block.text).strip()}\n</evidence>"
        )

    return EvidenceBundle(
        prompt_text="\n\n".join(parts), alias_to_candidate=mapping, parent_aliases=parents
    )


def build_answer_messages(
    question: str, bundle: EvidenceBundle, *, conversation_summary: str | None = None
) -> tuple[str, str]:
    """Return the ``(system, user)`` prompt pair for a non-streaming answer."""
    system = load_prompt("answer_json")
    history = (
        f"Earlier in this conversation: {conversation_summary.strip()[:600]}\n\n"
        if conversation_summary
        else ""
    )
    user = (
        f"{history}EVIDENCE\n{bundle.prompt_text}\n\n"
        f"QUESTION\n{question.strip()}\n\n"
        f"Valid citation ids: {', '.join(sorted(bundle.alias_to_candidate))}."
    )
    return system, user


def build_stream_messages(
    question: str, bundle: EvidenceBundle, *, conversation_summary: str | None = None
) -> tuple[str, str]:
    """Return the ``(system, user)`` prompt pair for a streaming answer."""
    system = load_prompt("answer_stream")
    history = (
        f"Earlier in this conversation: {conversation_summary.strip()[:600]}\n\n"
        if conversation_summary
        else ""
    )
    user = (
        f"{history}EVIDENCE\n{bundle.prompt_text}\n\n"
        f"QUESTION\n{question.strip()}\n\n"
        f"Valid citation ids: {', '.join(sorted(bundle.alias_to_candidate))}."
    )
    return system, user


def build_repair_messages(
    question: str, bundle: EvidenceBundle, failed_claims: list[str]
) -> tuple[str, str]:
    """Return the ``(system, user)`` prompt pair for the repair round."""
    system = load_prompt("repair")
    listed = "\n".join(f"- {claim}" for claim in failed_claims)
    user = (
        f"EVIDENCE\n{bundle.prompt_text}\n\n"
        f"QUESTION\n{question.strip()}\n\n"
        f"CLAIMS THAT FAILED VERIFICATION\n{listed}\n\n"
        f"Valid citation ids: {', '.join(sorted(bundle.alias_to_candidate))}."
    )
    return system, user


def parse_answer_json(raw: str) -> dict[str, Any]:
    """Parse a non-streaming answer payload, tolerating code fences."""
    return _loads(raw)


def split_stream_output(raw: str) -> tuple[str, dict[str, Any]]:
    """Split streamed output into the markdown body and the parsed JSON trailer."""
    if STREAM_TERMINATOR in raw:
        body, _, trailer = raw.partition(STREAM_TERMINATOR)
    else:
        body, trailer = raw, ""
    try:
        parsed = _loads(trailer) if trailer.strip() else {}
    except ValueError:
        parsed = {}
    return body.strip(), parsed


def markers_in(text: str) -> list[str]:
    """Return the citation aliases referenced inline in streamed markdown."""
    return [m for m in MARKER_RE.findall(text) if ALIAS_RE.match(m)]


def strip_markers(text: str) -> str:
    """Remove inline ``[c3]`` markers, used when renumbering a streamed answer."""
    return re.sub(r"\s*\[[cs]\d{1,2}\]", "", text).strip()


def _loads(raw: str) -> dict[str, Any]:
    """Parse JSON from a model response, stripping fences and trailing prose."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"model response is not JSON: {text[:160]!r}") from None
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def _attr(value: str) -> str:
    """Escape a value for use inside an XML-style attribute."""
    return value.replace('"', "'").replace("<", "").replace(">", "")[:200]
