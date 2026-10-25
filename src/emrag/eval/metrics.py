"""Retrieval and answer metrics.

Two families:

* **retrieval**, computable without any LLM call, so they run in CI on every pull
  request and block a merge that regresses recall;
* **answer**, which need a judge model and therefore run nightly.

Gold items may name expected chunk ids, expected pages, or both. Pages are the
more durable target, because a chunking change alters chunk ids while the page an
answer lives on does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldItem:
    """One golden question."""

    id: str
    question: str
    answerable: bool = True
    tenant_id: str = "demo"
    category: str = "factoid"
    modality: str = "text"
    difficulty: str = "medium"
    source_file: str = ""
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_pages: list[int] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    gold_answer: str = ""
    gold_quotes: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GoldItem:
        """Build an item from a JSONL record, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def load_golden(path: str | Path) -> list[GoldItem]:
    """Load a golden set from JSONL, skipping blank lines and comments."""
    items: list[GoldItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        items.append(GoldItem.from_dict(json.loads(stripped)))
    return items


@dataclass
class RetrievalHit:
    """One retrieved chunk, reduced to what metrics need."""

    chunk_id: str
    doc_id: str
    page: int
    text: str = ""


def is_relevant(hit: RetrievalHit, item: GoldItem) -> bool:
    """Decide whether a hit satisfies a gold item.

    Chunk ids are checked first when the gold set names them; otherwise the
    document plus page pair is used.
    """
    if item.expected_chunk_ids:
        return hit.chunk_id in item.expected_chunk_ids
    if item.expected_doc_ids and hit.doc_id not in item.expected_doc_ids:
        return False
    if item.expected_pages:
        return hit.page in item.expected_pages
    return bool(item.expected_doc_ids)


def recall_at_k(hits: list[RetrievalHit], item: GoldItem, k: int) -> float:
    """1.0 when any of the first ``k`` hits is relevant, else 0.0."""
    return 1.0 if any(is_relevant(h, item) for h in hits[:k]) else 0.0


def mrr_at_k(hits: list[RetrievalHit], item: GoldItem, k: int) -> float:
    """Reciprocal rank of the first relevant hit within ``k``."""
    for rank, hit in enumerate(hits[:k], start=1):
        if is_relevant(hit, item):
            return 1.0 / rank
    return 0.0


def precision_at_k(hits: list[RetrievalHit], item: GoldItem, k: int) -> float:
    """Share of the first ``k`` hits that are relevant."""
    window = hits[:k]
    if not window:
        return 0.0
    return sum(1 for h in window if is_relevant(h, item)) / len(window)


def quote_coverage(hits: list[RetrievalHit], item: GoldItem) -> float:
    """Share of gold quotes that appear in the retrieved text.

    A direct check that the evidence needed to cite an answer was actually
    retrieved, independent of how the model then phrased things.
    """
    if not item.gold_quotes:
        return 1.0
    from emrag.core.textnorm import normalize

    corpus = normalize(" ".join(h.text for h in hits))
    found = sum(1 for quote in item.gold_quotes if normalize(quote) in corpus)
    return found / len(item.gold_quotes)


def citation_accuracy(citations: list[dict[str, Any]]) -> float:
    """Share of returned citations marked verified.

    This is 1.0 by construction after filtering; the test suite asserts it, which
    turns the traceability guarantee into a regression test.
    """
    if not citations:
        return 1.0
    return sum(1 for c in citations if c.get("verified")) / len(citations)


def citation_location_accuracy(citations: list[dict[str, Any]], item: GoldItem) -> float:
    """Share of citations pointing at a page the gold set accepts."""
    if not citations or not (item.expected_pages or item.expected_doc_ids):
        return 1.0
    correct = 0
    for citation in citations:
        page_ok = not item.expected_pages or citation.get("page") in item.expected_pages
        doc_ok = not item.expected_doc_ids or citation.get("document_id") in item.expected_doc_ids
        correct += int(page_ok and doc_ok)
    return correct / len(citations)


def not_found_correct(status: str, item: GoldItem) -> bool:
    """Whether the refusal behaviour matched the gold expectation."""
    refused = status in ("not_found", "refused")
    return refused if not item.answerable else not refused


@dataclass
class Aggregate:
    """Mean of a metric across a run."""

    name: str
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        """Record one observation."""
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        """Arithmetic mean, or 0.0 for an empty aggregate."""
        return self.total / self.count if self.count else 0.0


class MetricSet:
    """Collects named aggregates for a run."""

    def __init__(self) -> None:
        self._metrics: dict[str, Aggregate] = {}

    def add(self, name: str, value: float) -> None:
        """Record one observation under ``name``."""
        self._metrics.setdefault(name, Aggregate(name)).add(value)

    def means(self) -> dict[str, float]:
        """Return every mean, rounded for reporting."""
        return {name: round(agg.mean, 4) for name, agg in sorted(self._metrics.items())}

    def counts(self) -> dict[str, int]:
        """Return the observation count per metric."""
        return {name: agg.count for name, agg in sorted(self._metrics.items())}

    def percentile(self, name: str, values: list[float], percentile: float) -> float:
        """Record and return a percentile over a value list."""
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(len(ordered) * percentile))
        result = ordered[index]
        self.add(name, result)
        return result
