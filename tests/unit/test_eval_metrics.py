"""Unit tests for the evaluation metrics and golden-set loader."""

from __future__ import annotations

from pathlib import Path

from emrag.eval.metrics import (
    GoldItem,
    MetricSet,
    RetrievalHit,
    citation_accuracy,
    citation_location_accuracy,
    is_relevant,
    load_golden,
    mrr_at_k,
    not_found_correct,
    precision_at_k,
    quote_coverage,
    recall_at_k,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "v1.jsonl"


def _item(**kw: object) -> GoldItem:
    base: dict[str, object] = {"id": "x", "question": "q", "expected_pages": [3]}
    base.update(kw)
    return GoldItem(**base)  # type: ignore[arg-type]


def _hit(page: int, chunk: str = "d_0000000000000000:p003:c0001", text: str = "") -> RetrievalHit:
    return RetrievalHit(chunk_id=chunk, doc_id="d_0000000000000000", page=page, text=text)


class TestGoldenSet:
    """The shipped golden set must load and meet its composition contract."""

    def test_loads(self) -> None:
        items = load_golden(GOLDEN)
        assert len(items) == 20
        assert all(item.id for item in items)
        assert all(item.question for item in items)

    def test_has_an_adversarial_subset(self) -> None:
        items = load_golden(GOLDEN)
        unanswerable = [i for i in items if not i.answerable]
        assert len(unanswerable) >= 5
        assert all(i.notes for i in unanswerable), "adversarial items must explain themselves"

    def test_answerable_items_carry_quotes_and_pages(self) -> None:
        for item in load_golden(GOLDEN):
            if item.answerable:
                assert item.gold_quotes, f"{item.id} has no gold quote"
                assert item.expected_pages, f"{item.id} has no expected page"

    def test_comments_and_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text('// a comment\n\n{"id": "a", "question": "q"}\n', encoding="utf-8")
        assert [i.id for i in load_golden(path)] == ["a"]


class TestRetrievalMetrics:
    """Relevance falls back from chunk ids to document and page."""

    def test_page_match_is_relevant(self) -> None:
        assert is_relevant(_hit(3), _item())
        assert not is_relevant(_hit(4), _item())

    def test_chunk_id_takes_precedence(self) -> None:
        item = _item(expected_chunk_ids=["d_0000000000000000:p003:c0001"])
        assert is_relevant(_hit(3), item)
        assert not is_relevant(_hit(3, chunk="d_0000000000000000:p003:c0002"), item)

    def test_doc_id_mismatch_is_irrelevant(self) -> None:
        item = _item(expected_doc_ids=["d_1111111111111111"])
        assert not is_relevant(_hit(3), item)

    def test_recall_and_mrr(self) -> None:
        hits = [_hit(9), _hit(3)]
        assert recall_at_k(hits, _item(), 2) == 1.0
        assert recall_at_k(hits, _item(), 1) == 0.0
        assert mrr_at_k(hits, _item(), 2) == 0.5

    def test_precision(self) -> None:
        assert precision_at_k([_hit(3), _hit(4)], _item(), 2) == 0.5

    def test_quote_coverage(self) -> None:
        item = _item(gold_quotes=["revenue grew 14 percent", "absent span"])
        hits = [_hit(3, text="Segment revenue grew 14 percent year over year.")]
        assert quote_coverage(hits, item) == 0.5

    def test_quote_coverage_without_quotes_is_one(self) -> None:
        assert quote_coverage([_hit(3)], _item()) == 1.0


class TestAnswerMetrics:
    """Citation accuracy and refusal correctness."""

    def test_citation_accuracy_counts_verified(self) -> None:
        assert citation_accuracy([{"verified": True}, {"verified": False}]) == 0.5
        assert citation_accuracy([]) == 1.0

    def test_location_accuracy(self) -> None:
        item = _item()
        assert citation_location_accuracy([{"page": 3}], item) == 1.0
        assert citation_location_accuracy([{"page": 7}], item) == 0.0

    def test_not_found_correctness(self) -> None:
        assert not_found_correct("not_found", _item(answerable=False))
        assert not_found_correct("refused", _item(answerable=False))
        assert not not_found_correct("answered", _item(answerable=False))
        assert not_found_correct("answered", _item(answerable=True))
        assert not not_found_correct("not_found", _item(answerable=True))


class TestMetricSet:
    """Aggregation and percentiles."""

    def test_means_and_counts(self) -> None:
        metrics = MetricSet()
        metrics.add("recall", 1.0)
        metrics.add("recall", 0.0)
        assert metrics.means()["recall"] == 0.5
        assert metrics.counts()["recall"] == 2

    def test_percentile_is_recorded(self) -> None:
        metrics = MetricSet()
        value = metrics.percentile("p95", [10.0, 20.0, 30.0, 40.0], 0.95)
        assert value == 40.0
        assert "p95" in metrics.means()

    def test_empty_percentile_is_zero(self) -> None:
        assert MetricSet().percentile("p95", [], 0.95) == 0.0
