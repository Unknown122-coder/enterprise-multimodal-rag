"""Context budgeting.

Chooses which retrieved chunks actually reach the model. Three jobs:

* **deduplicate** - identical or near-identical passages waste budget, and
  overlapping chunks from the same section often repeat text;
* **budget** - fill a token allowance greedily by rerank score, truncating the
  last passage at a sentence boundary rather than mid-word;
* **expand** - optionally append the parent section of the strongest hits, which
  helps questions whose answer straddles a chunk boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from emrag.core.textnorm import normalize
from emrag.core.tokens import count_tokens, truncate_to_tokens
from emrag.core.types import Candidate, Parent
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_DEDUPE_OVERLAP = 0.85
_PER_CHUNK_OVERHEAD = 40  # evidence block header tokens


@dataclass
class EvidenceBlock:
    """One evidence item as presented to the model."""

    alias: str
    candidate: Candidate
    text: str
    tokens: int
    is_parent: bool = False


@dataclass
class BudgetResult:
    """Selected evidence plus what was dropped."""

    blocks: list[EvidenceBlock]
    tokens_used: int = 0
    dropped_duplicates: int = 0
    dropped_budget: int = 0


def deduplicate(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """Drop candidates whose text substantially repeats a higher-scoring one."""
    kept: list[Candidate] = []
    kept_tokens: list[set[str]] = []
    dropped = 0

    for candidate in candidates:
        tokens = set(normalize(candidate.chunk.text).split())
        if not tokens:
            dropped += 1
            continue
        duplicate = False
        for existing in kept_tokens:
            overlap = len(tokens & existing) / max(1, min(len(tokens), len(existing)))
            if overlap >= _DEDUPE_OVERLAP:
                duplicate = True
                break
        if duplicate:
            dropped += 1
            continue
        kept.append(candidate)
        kept_tokens.append(tokens)
    return kept, dropped


def select_evidence(
    candidates: list[Candidate],
    *,
    token_budget: int,
    top_k: int,
    parents: dict[str, Parent] | None = None,
    parent_expansion: bool = True,
    max_parents: int = 2,
) -> BudgetResult:
    """Pick the evidence blocks for one generation call.

    Candidates must already be sorted best first. Aliases are short (``c1``,
    ``c2``, ...) to keep the prompt small and to avoid exposing chunk ids to the
    model, which would invite it to invent plausible-looking ones.
    """
    unique, duplicates = deduplicate(candidates)
    selected = unique[:top_k]

    blocks: list[EvidenceBlock] = []
    used = 0
    dropped_budget = 0

    for index, candidate in enumerate(selected, start=1):
        alias = f"c{index}"
        text = candidate.chunk.text
        cost = count_tokens(text) + _PER_CHUNK_OVERHEAD
        remaining = token_budget - used
        if remaining <= _PER_CHUNK_OVERHEAD:
            dropped_budget += len(selected) - index + 1
            break
        if cost > remaining:
            text = truncate_to_tokens(text, remaining - _PER_CHUNK_OVERHEAD)
            if not text.strip():
                dropped_budget += 1
                continue
            cost = count_tokens(text) + _PER_CHUNK_OVERHEAD
        blocks.append(EvidenceBlock(alias=alias, candidate=candidate, text=text, tokens=cost))
        used += cost

    if parent_expansion and parents:
        used += _append_parents(blocks, parents, token_budget - used, max_parents)

    logger.info(
        "retrieval.budget",
        blocks=len(blocks),
        tokens=used,
        budget=token_budget,
        duplicates=duplicates,
        dropped=dropped_budget,
    )
    return BudgetResult(
        blocks=blocks,
        tokens_used=used,
        dropped_duplicates=duplicates,
        dropped_budget=dropped_budget,
    )


def _append_parents(
    blocks: list[EvidenceBlock],
    parents: dict[str, Parent],
    remaining: int,
    max_parents: int,
) -> int:
    """Append parent sections of the strongest hits while budget allows."""
    if remaining <= _PER_CHUNK_OVERHEAD * 2:
        return 0
    added = 0
    spent = 0
    seen: set[str] = set()
    for block in list(blocks):
        if added >= max_parents:
            break
        parent_id = block.candidate.chunk.parent_id
        if parent_id in seen:
            continue
        seen.add(parent_id)
        parent = parents.get(parent_id)
        if parent is None:
            continue
        if normalize(parent.text) == normalize(block.candidate.chunk.text):
            continue
        budget = remaining - spent - _PER_CHUNK_OVERHEAD
        if budget <= 80:
            break
        text = truncate_to_tokens(parent.text, budget)
        if not text.strip():
            continue
        cost = count_tokens(text) + _PER_CHUNK_OVERHEAD
        blocks.append(
            EvidenceBlock(
                alias=f"s{added + 1}",
                candidate=block.candidate,
                text=text,
                tokens=cost,
                is_parent=True,
            )
        )
        spent += cost
        added += 1
    return spent
