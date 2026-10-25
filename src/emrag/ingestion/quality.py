"""Ingestion quality gates.

A document reaches ``ready`` only when it is actually answerable: nearly every
page produced a chunk, the metadata contract holds, and vector counts match
chunk counts. Otherwise it becomes ``ready_degraded`` with a report naming the
pages that failed, which is far more useful than a silent partial index.
"""

from __future__ import annotations

from emrag.core.types import Chunk, PageInfo, QualityReport
from emrag.ingestion.metadata import check_invariants
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

MIN_PAGE_COVERAGE = 0.95


def evaluate(
    chunks: list[Chunk],
    pages: list[PageInfo],
    *,
    parser_degraded_reasons: list[str] | None = None,
) -> QualityReport:
    """Build the quality report for one ingested document."""
    report = QualityReport(pages_total=len(pages))
    report.degraded_reasons = list(parser_degraded_reasons or [])
    report.pages_blank = sum(1 for p in pages if p.blank)
    report.ocr_pages = sum(1 for p in pages if p.scanned)

    pages_with_chunks = {c.page for c in chunks}
    report.pages_with_chunks = len(pages_with_chunks)
    expected = {p.page for p in pages if not p.blank}
    report.pages_failed = sorted(expected - pages_with_chunks)

    if not chunks:
        report.passed = False
        report.degraded_reasons.append("no chunks produced")
        return report

    coverage = report.pages_with_chunks / max(1, len(expected))
    if coverage < MIN_PAGE_COVERAGE:
        report.degraded_reasons.append(
            f"page coverage {coverage:.0%} below {MIN_PAGE_COVERAGE:.0%}"
        )

    invariants = check_invariants(chunks, page_count=max((p.page for p in pages), default=1))
    report.degraded_reasons.extend(invariants[:10])

    report.passed = not report.degraded_reasons
    if not report.passed:
        logger.warning(
            "ingest.quality_degraded",
            reasons=report.degraded_reasons[:5],
            failed_pages=report.pages_failed[:10],
        )
    return report


def verify_upsert(expected: int, actual: int) -> list[str]:
    """Compare chunk count with upserted point count."""
    if expected == actual:
        return []
    return [f"upserted {actual} points for {expected} chunks"]
