"""Document triage.

Decides, per page, whether text can be extracted natively or the page must be
OCR'd, and classifies the document as ``native``, ``scanned`` or ``mixed``. The
parser chain and the cost of ingestion both follow from this.
"""

from __future__ import annotations

from typing import Any

from emrag.core.types import DocClass, PageInfo
from emrag.observability.logging import get_logger

logger = get_logger(__name__)


def triage_pdf(
    pdf_bytes: bytes,
    *,
    min_chars_native: int = 50,
    image_coverage_scanned: float = 0.6,
) -> tuple[list[PageInfo], DocClass]:
    """Classify every page of a PDF and the document as a whole.

    A page is treated as scanned when it yields almost no characters *and* is
    mostly covered by images. A page with neither text nor images is blank.
    """
    from emrag.pdfcompat import fitz

    pages: list[PageInfo] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc, start=1):
            rect = page.rect
            text = page.get_text("text") or ""
            char_count = len(text.strip())
            coverage = _image_coverage(page, rect.width * rect.height)
            scanned = char_count < min_chars_native and coverage >= image_coverage_scanned
            blank = char_count < 5 and coverage < 0.02
            pages.append(
                PageInfo(
                    page=index,
                    width_pt=float(rect.width),
                    height_pt=float(rect.height),
                    char_count=char_count,
                    image_coverage=round(coverage, 4),
                    scanned=scanned,
                    blank=blank,
                )
            )

    doc_class = classify_document(pages)
    logger.info(
        "ingest.triage",
        pages=len(pages),
        doc_class=doc_class,
        scanned_pages=sum(1 for p in pages if p.scanned),
    )
    return pages, doc_class


def classify_document(pages: list[PageInfo]) -> DocClass:
    """Aggregate per-page classes into a document class."""
    considered = [p for p in pages if not p.blank]
    if not considered:
        return "native"
    scanned = sum(1 for p in considered if p.scanned)
    if scanned == 0:
        return "native"
    if scanned == len(considered):
        return "scanned"
    return "mixed"


def _image_coverage(page: Any, page_area: float) -> float:
    """Fraction of the page covered by raster images, clamped to [0, 1]."""
    if page_area <= 0:
        return 0.0
    covered = 0.0
    try:
        for info in page.get_image_info():
            bbox = info.get("bbox")
            if not bbox:
                continue
            width = max(0.0, bbox[2] - bbox[0])
            height = max(0.0, bbox[3] - bbox[1])
            covered += width * height
    except Exception as exc:  # noqa: BLE001 - malformed image dictionaries are common
        logger.debug("ingest.image_info_failed", error=str(exc)[:160])
        return 0.0
    return min(1.0, covered / page_area)


def page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages without a full triage pass."""
    from emrag.pdfcompat import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return int(doc.page_count)
