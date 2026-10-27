"""Evaluation CLI.

Two modes:

``retrieval``  no LLM calls at all. Measures recall, MRR, precision and whether
               the gold quotes were actually retrieved. Fast and free, so CI runs
               it on every pull request with a minimum-recall gate.
``full``       adds generation, citation verification and the judge metrics.
               Costs roughly one and a half LLM calls per item, so it runs
               nightly.

Usage::

    python -m emrag.eval.run_eval --mode retrieval --golden tests/golden/v1.jsonl
    python -m emrag.eval.run_eval --mode full --min-faithfulness 0.85 --report out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from emrag.api.deps import AppState, build_state
from emrag.core.errors import QuotaExhausted
from emrag.core.types import Principal
from emrag.eval.judges import judge_faithfulness, judge_relevance
from emrag.eval.metrics import (
    GoldItem,
    MetricSet,
    RetrievalHit,
    citation_accuracy,
    citation_location_accuracy,
    load_golden,
    mrr_at_k,
    not_found_correct,
    precision_at_k,
    quote_coverage,
    recall_at_k,
)
from emrag.generation.answerer import AnswerOptions
from emrag.observability.logging import configure_logging, get_logger
from emrag.retrieval import filters as filter_builder
from emrag.retrieval.retriever import RetrievalOptions
from emrag.settings import Settings, get_settings

logger = get_logger(__name__)

DEFAULT_GOLDEN = "tests/golden/v1.jsonl"
# Full mode is a quality measurement, not a latency one, and the free tiers need
# room to fall back between providers; the interactive 4.5 s deadline is far too
# tight for that. Latency is still reported separately.
DEFAULT_EVAL_DEADLINE_MS = 20_000
QUOTA_RETRIES = 3
MAX_QUOTA_WAIT_S = 90.0


def _principal(item: GoldItem) -> Principal:
    """Build an evaluation principal with full read access to one tenant."""
    return Principal(
        subject="eval",
        tenant_id=item.tenant_id,
        groups=["all"],
        clearance="confidential",
        role="admin",
        auth_method="api_key",
    )


async def evaluate(
    state: AppState,
    items: list[GoldItem],
    *,
    mode: str,
    top_k: int,
    judge: bool,
    deadline_ms: int = DEFAULT_EVAL_DEADLINE_MS,
) -> dict[str, Any]:
    """Run the evaluation and return a report dictionary.

    One item failing never loses the run: a quota exhaustion waits for the
    provider's retry hint and tries the item again a few times, and any other
    error is recorded against the item and the run moves on. The first real
    full-mode run died on item 9 of 42 with nothing written.
    """
    metrics = MetricSet()
    latencies: list[float] = []
    per_item: list[dict[str, Any]] = []
    errors = 0
    cfg = state.settings

    for item in items:
        principal = _principal(item)
        retrieval_filter = filter_builder.from_request(
            principal, doc_ids=item.expected_doc_ids or None
        )
        started = time.perf_counter()
        try:
            record = await _evaluate_item(
                state,
                item,
                retrieval_filter,
                metrics,
                mode=mode,
                top_k=top_k,
                judge=judge,
                deadline_ms=deadline_ms,
            )
        except Exception as exc:  # noqa: BLE001 - recorded per item, run continues
            errors += 1
            logger.warning("eval.item_failed", item=item.id, error=str(exc)[:200])
            record = {"id": item.id, "error": str(exc)[:200]}
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        record["latency_ms"] = int(elapsed_ms)
        per_item.append(record)

    p50 = metrics.percentile("latency_p50_ms", latencies, 0.5)
    p95 = metrics.percentile("latency_p95_ms", latencies, 0.95)

    return {
        "mode": mode,
        "items": len(items),
        "errors": errors,
        "top_k": top_k,
        "backend": cfg.vector_backend,
        "embed_model": state.embedder.model_name,
        "metrics": metrics.means(),
        "counts": metrics.counts(),
        "latency": {"p50_ms": int(p50), "p95_ms": int(p95)},
        "per_item": per_item,
    }


async def _evaluate_item(
    state: AppState,
    item: GoldItem,
    retrieval_filter: Any,
    metrics: MetricSet,
    *,
    mode: str,
    top_k: int,
    judge: bool,
    deadline_ms: int,
) -> dict[str, Any]:
    """Score one item, waiting out quota exhaustion rather than failing on it."""
    for attempt in range(QUOTA_RETRIES + 1):
        try:
            return await _score_item(
                state,
                item,
                retrieval_filter,
                metrics,
                mode=mode,
                top_k=top_k,
                judge=judge,
                deadline_ms=deadline_ms,
            )
        except QuotaExhausted as exc:
            if attempt == QUOTA_RETRIES:
                raise
            wait = min(float(exc.retry_after_s or 60), MAX_QUOTA_WAIT_S)
            logger.warning("eval.quota_wait", item=item.id, wait_s=wait, attempt=attempt + 1)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def _score_item(
    state: AppState,
    item: GoldItem,
    retrieval_filter: Any,
    metrics: MetricSet,
    *,
    mode: str,
    top_k: int,
    judge: bool,
    deadline_ms: int,
) -> dict[str, Any]:
    """Run retrieval or generation for one item and add its metrics."""
    cfg = state.settings
    if mode == "retrieval":
        outcome = await state.retriever.retrieve(
            item.question,
            retrieval_filter,
            RetrievalOptions(
                top_k=top_k,
                candidates_k=cfg.candidates_k,
                rewrite="off",
                parent_expansion=False,
                rerank=cfg.rerank_enabled,
                token_budget=cfg.evidence_token_budget,
                gate_tau=cfg.retrieval_gate_tau,
            ),
        )
        hits = [
            RetrievalHit(
                chunk_id=c.chunk.chunk_id,
                doc_id=c.chunk.doc_id,
                page=c.chunk.page,
                text=c.chunk.text,
            )
            for c in outcome.candidates
        ]
        record: dict[str, Any] = {"id": item.id, "hits": len(hits)}
        if item.answerable:
            metrics.add(f"recall@{top_k}", recall_at_k(hits, item, top_k))
            metrics.add(f"mrr@{top_k}", mrr_at_k(hits, item, top_k))
            metrics.add(f"precision@{top_k}", precision_at_k(hits, item, top_k))
            metrics.add("quote_coverage", quote_coverage(hits, item))
            record["recall"] = recall_at_k(hits, item, top_k)
        else:
            metrics.add("not_found_gate", 0.0 if outcome.gate_passed else 1.0)
            record["gate_passed"] = outcome.gate_passed
    else:
        result = await state.answerer.answer(
            item.question,
            retrieval_filter,
            retrieval=RetrievalOptions(
                top_k=top_k,
                candidates_k=cfg.candidates_k,
                rewrite="auto",
                parent_expansion=True,
                rerank=cfg.rerank_enabled,
                token_budget=cfg.evidence_token_budget,
                gate_tau=cfg.retrieval_gate_tau,
            ),
            options=AnswerOptions(deadline_ms=deadline_ms, debug=False),
        )
        payload = result.model_dump(mode="json")
        citations = payload.get("citations") or []
        claims = (payload.get("answer") or {}).get("claims") or []

        metrics.add("citation_accuracy", citation_accuracy(citations))
        metrics.add("citation_location_accuracy", citation_location_accuracy(citations, item))
        metrics.add("not_found_correct", float(not_found_correct(result.status, item)))
        metrics.add("answered", float(result.status in ("answered", "partial")))
        record = {
            "id": item.id,
            "status": result.status,
            "citations": len(citations),
            "claims": len(claims),
        }
        if judge and claims and state.router.has_tier("judge"):
            faithfulness = await judge_faithfulness(state.router, claims, citations)
            relevance = await judge_relevance(
                state.router, item.question, (payload.get("answer") or {}).get("markdown", "")
            )
            if faithfulness == faithfulness:  # not NaN
                metrics.add("faithfulness", faithfulness)
                record["faithfulness"] = round(faithfulness, 3)
            if relevance == relevance:
                metrics.add("answer_relevance", relevance)
                record["relevance"] = relevance
    return record


def _print_report(report: dict[str, Any]) -> None:
    """Print a compact human-readable summary."""
    sys.stdout.write(
        f"\nmode={report['mode']} items={report['items']} top_k={report['top_k']} "
        f"backend={report['backend']}\n"
    )
    width = max((len(k) for k in report["metrics"]), default=10)
    for name, value in report["metrics"].items():
        sys.stdout.write(f"  {name.ljust(width)}  {value}\n")
    latency = report["latency"]
    sys.stdout.write(f"  latency p50={latency['p50_ms']}ms p95={latency['p95_ms']}ms\n")
    if report.get("errors"):
        sys.stdout.write(
            f"  ERRORS: {report['errors']} item(s) failed; see per_item in the report\n"
        )
    sys.stdout.write("\n")


def _check_gates(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Return the list of threshold failures."""
    failures: list[str] = []
    metrics = report["metrics"]
    gates = [
        (f"recall@{args.top_k}", args.min_recall),
        ("citation_accuracy", args.min_citation_accuracy),
        ("faithfulness", args.min_faithfulness),
    ]
    for name, minimum in gates:
        if minimum is None or name not in metrics:
            continue
        if metrics[name] < minimum:
            failures.append(f"{name}={metrics[name]} below minimum {minimum}")
    if args.max_p95_ms is not None and report["latency"]["p95_ms"] > args.max_p95_ms:
        failures.append(f"p95={report['latency']['p95_ms']}ms above maximum {args.max_p95_ms}ms")
    return failures


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="emrag-eval", description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N items")
    parser.add_argument("--no-judge", action="store_true", help="skip judge metrics in full mode")
    parser.add_argument(
        "--deadline-ms",
        type=int,
        default=DEFAULT_EVAL_DEADLINE_MS,
        help="per-answer deadline in full mode (generous by default: quality, not latency)",
    )
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--min-citation-accuracy", type=float, default=None)
    parser.add_argument("--min-faithfulness", type=float, default=None)
    parser.add_argument("--max-p95-ms", type=int, default=None)
    return parser


async def _main(args: argparse.Namespace) -> int:
    settings: Settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    golden_path = Path(args.golden)
    if not await asyncio.to_thread(golden_path.exists):
        sys.stderr.write(f"golden set not found: {golden_path}\n")
        return 2
    items = load_golden(golden_path)
    if args.limit:
        items = items[: args.limit]
    if not items:
        sys.stderr.write("golden set is empty\n")
        return 2

    state = build_state(settings)
    try:
        await state.store.ensure_schema(settings.embed_dim, sparse=True)
        await state.registry.refresh()
        report = await evaluate(
            state,
            items,
            mode=args.mode,
            top_k=args.top_k,
            judge=not args.no_judge,
            deadline_ms=args.deadline_ms,
        )
    finally:
        await state.aclose()

    _print_report(report)
    if args.report:
        await asyncio.to_thread(args.report.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            args.report.write_text, json.dumps(report, indent=2), encoding="utf-8"
        )
        sys.stdout.write(f"report written to {args.report}\n")

    failures = _check_gates(report, args)
    for failure in failures:
        sys.stderr.write(f"GATE FAILED: {failure}\n")
    return 1 if failures else 0


def main() -> int:
    """Entry point for the ``emrag-eval`` console script."""
    args = build_parser().parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
