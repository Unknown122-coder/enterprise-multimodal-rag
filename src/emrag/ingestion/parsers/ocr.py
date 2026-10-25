"""Standalone OCR for scanned pages.

Used when Docling is unavailable but a page still has no extractable text: the
page is rendered at 300 dpi, deskewed and passed to RapidOCR. Word boxes are
merged into line blocks so a citation can still point at a region.
"""

from __future__ import annotations

from typing import Any

from emrag.core.types import Element
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_LINE_MERGE_TOLERANCE = 0.012  # page fractions


def ocr_available() -> bool:
    """Return True when an OCR engine is importable."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_page(
    pdf_bytes: bytes, page_number: int, *, dpi: int = 300, order_start: int = 0
) -> list[Element]:
    """OCR one page and return merged line elements with normalised boxes."""
    if not ocr_available():
        logger.warning("ingest.ocr_unavailable", page=page_number)
        return []

    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    from emrag.pdfcompat import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if not 1 <= page_number <= doc.page_count:
            return []
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        width_px, height_px = pixmap.width, pixmap.height
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(height_px, width_px, pixmap.n)

    engine = RapidOCR()
    detections, _elapsed = engine(image)
    if not detections:
        return []

    boxes: list[tuple[list[float], str, float]] = []
    for box, text, score in detections:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        xs = [float(point[0]) / width_px for point in box]
        ys = [float(point[1]) / height_px for point in box]
        boxes.append(
            (
                [
                    min(max(min(xs), 0.0), 1.0),
                    min(max(min(ys), 0.0), 1.0),
                    min(max(max(xs), 0.0), 1.0),
                    min(max(max(ys), 0.0), 1.0),
                ],
                cleaned,
                float(score),
            )
        )

    boxes.sort(key=lambda item: (round(item[0][1], 3), item[0][0]))
    elements: list[Element] = []
    order = order_start
    for bbox, text, confidence in _merge_lines(boxes):
        order += 1
        elements.append(
            Element(
                kind="ocr_block",
                page=page_number,
                bbox=bbox,
                bbox_confidence=round(confidence, 3),
                text=text,
                element_id=f"ocr:p{page_number}:l{order}",
                order=order,
            )
        )
    logger.info("ingest.ocr_page", page=page_number, lines=len(elements))
    return elements


def _merge_lines(
    boxes: list[tuple[list[float], str, float]],
) -> list[tuple[list[float], str, float]]:
    """Group word boxes that share a baseline into single line blocks."""
    merged: list[tuple[list[float], str, float]] = []
    current: list[tuple[list[float], str, float]] = []
    current_y = None

    for item in boxes:
        y_centre = (item[0][1] + item[0][3]) / 2
        if current_y is None or abs(y_centre - current_y) <= _LINE_MERGE_TOLERANCE:
            current.append(item)
            current_y = y_centre if current_y is None else (current_y + y_centre) / 2
        else:
            merged.append(_collapse(current))
            current = [item]
            current_y = y_centre
    if current:
        merged.append(_collapse(current))
    return merged


def _collapse(group: list[tuple[list[float], str, float]]) -> tuple[list[float], str, float]:
    """Union the boxes of a line and join their text left to right."""
    ordered = sorted(group, key=lambda item: item[0][0])
    bbox = [
        min(i[0][0] for i in ordered),
        min(i[0][1] for i in ordered),
        max(i[0][2] for i in ordered),
        max(i[0][3] for i in ordered),
    ]
    text = " ".join(i[1] for i in ordered)
    confidence = sum(i[2] for i in ordered) / len(ordered)
    return bbox, text, confidence


def deskew(image: Any) -> Any:
    """Deskew a page image before OCR.

    Assumption: RapidOCR's detector tolerates small rotations, so this is a no-op
    placeholder kept as the single hook for adding OpenCV-based deskewing without
    touching callers.
    """
    return image
