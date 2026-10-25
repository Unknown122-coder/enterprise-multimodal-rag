"""Docling parser (preferred engine for the full pipeline).

Docling contributes what PyMuPDF cannot: a learned layout model for reading
order, TableFormer for table structure, an OCR backend for scanned pages, and
per-item page plus bounding-box provenance. It is heavy (PyTorch, ONNX models),
so it is an optional dependency enabled with ``USE_DOCLING=true`` and installed
only in the ingestion environment.

Docling bounding boxes use a bottom-left origin in PDF points; this module
converts them to the top-left-origin page fractions the chunk contract requires.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from emrag.core.types import Element, PageInfo
from emrag.ingestion.parsers.pymupdf_parser import ParseResult, render_page
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_LABEL_MAP = {
    "title": "heading",
    "section_header": "heading",
    "page_header": "paragraph",
    "page_footer": "paragraph",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list_item",
    "caption": "caption",
    "table": "table",
    "picture": "figure",
    "code": "paragraph",
    "formula": "paragraph",
}


def docling_available() -> bool:
    """Return True when the optional docling dependency is importable."""
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def parse_pdf(
    pdf_bytes: bytes,
    *,
    pages: list[PageInfo] | None = None,
    needs_ocr: bool = False,
    render_pages: bool = True,
    ocr_engine: str = "rapidocr",
) -> ParseResult:
    """Parse a PDF with Docling, raising ``ImportError`` if it is not installed."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = needs_ocr
    options.do_table_structure = True
    options.table_structure_options.do_cell_matching = True
    options.generate_page_images = False
    options.generate_picture_images = True
    if needs_ocr:
        _apply_ocr_options(options, ocr_engine)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "input.pdf"
        source.write_bytes(pdf_bytes)
        conversion = converter.convert(source)

    document = conversion.document
    result = ParseResult(engine=f"docling{'+ocr' if needs_ocr else ''}")
    result.pages = list(pages or _pages_from_docling(document))
    page_geometry = {p.page: (p.width_pt, p.height_pt) for p in result.pages}

    order = 0
    for item, _level in document.iterate_items():
        label = str(getattr(item, "label", "") or "").lower()
        kind = _LABEL_MAP.get(label)
        if kind is None:
            continue
        page_no, bbox = _locate(item, page_geometry)
        if page_no is None:
            continue
        order += 1
        element = Element(
            kind=kind,  # type: ignore[arg-type]
            page=page_no,
            bbox=bbox,
            text=_item_text(item),
            level=_heading_level(item) if kind == "heading" else None,
            element_id=str(getattr(item, "self_ref", f"docling:{order}")),
            order=order,
        )
        if kind == "table":
            element.table_cells = _table_cells(item)
            element.table_caption = _caption_text(item, document)
        if kind == "figure":
            image_key = f"p{page_no:03d}_fig{order}.png"
            image_bytes = _picture_bytes(item, document)
            if image_bytes:
                result.images[image_key] = image_bytes
                element.image_bytes_ref = image_key
            element.table_caption = None
            element.text = _caption_text(item, document) or ""
        result.elements.append(element)

    if needs_ocr:
        for element in result.elements:
            if element.kind == "paragraph" and element.page in _scanned_pages(result.pages):
                element.kind = "ocr_block"
                element.bbox_confidence = 0.85

    if render_pages:
        result.page_renders = _render_all(pdf_bytes)

    result.elements.sort(key=lambda e: (e.page, e.order))
    logger.info(
        "ingest.docling_parsed",
        elements=len(result.elements),
        images=len(result.images),
        ocr=needs_ocr,
    )
    return result


def _apply_ocr_options(options: Any, engine: str) -> None:
    """Select an OCR backend, degrading to Docling's default if unavailable."""
    try:
        if engine == "rapidocr":
            from docling.datamodel.pipeline_options import RapidOcrOptions

            options.ocr_options = RapidOcrOptions(force_full_page_ocr=True)
        elif engine == "easyocr":
            from docling.datamodel.pipeline_options import EasyOcrOptions

            options.ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        elif engine == "tesseract":
            from docling.datamodel.pipeline_options import TesseractCliOcrOptions

            options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    except ImportError as exc:
        logger.warning("ingest.ocr_backend_unavailable", engine=engine, error=str(exc)[:160])


def _pages_from_docling(document: Any) -> list[PageInfo]:
    """Build page geometry from the Docling document."""
    out: list[PageInfo] = []
    for page_no, page in sorted((document.pages or {}).items()):
        size = getattr(page, "size", None)
        width = float(getattr(size, "width", 612.0) or 612.0)
        height = float(getattr(size, "height", 792.0) or 792.0)
        out.append(PageInfo(page=int(page_no), width_pt=width, height_pt=height))
    return out or [PageInfo(page=1, width_pt=612.0, height_pt=792.0)]


def _locate(item: Any, geometry: dict[int, tuple[float, float]]) -> tuple[int | None, list[float]]:
    """Return the item's page and a top-left-origin normalised bbox."""
    provenance = list(getattr(item, "prov", []) or [])
    if not provenance:
        return None, [0.0, 0.0, 1.0, 1.0]
    first = provenance[0]
    page_no = int(getattr(first, "page_no", 0) or 0)
    if page_no <= 0:
        return None, [0.0, 0.0, 1.0, 1.0]
    width, height = geometry.get(page_no, (612.0, 792.0))
    bbox = getattr(first, "bbox", None)
    if bbox is None:
        return page_no, [0.0, 0.0, 1.0, 1.0]

    left = float(getattr(bbox, "l", 0.0))
    right = float(getattr(bbox, "r", width))
    top = float(getattr(bbox, "t", height))
    bottom = float(getattr(bbox, "b", 0.0))
    origin = str(getattr(getattr(bbox, "coord_origin", None), "value", "BOTTOMLEFT")).upper()

    if "BOTTOM" in origin:
        y0 = (height - max(top, bottom)) / height
        y1 = (height - min(top, bottom)) / height
    else:
        y0 = min(top, bottom) / height
        y1 = max(top, bottom) / height

    normalised = [
        min(max(min(left, right) / width, 0.0), 1.0),
        min(max(y0, 0.0), 1.0),
        min(max(max(left, right) / width, 0.0), 1.0),
        min(max(y1, 0.0), 1.0),
    ]
    if normalised[0] > normalised[2]:
        normalised[0], normalised[2] = normalised[2], normalised[0]
    if normalised[1] > normalised[3]:
        normalised[1], normalised[3] = normalised[3], normalised[1]
    return page_no, normalised


def _item_text(item: Any) -> str:
    """Best-effort text of a Docling item."""
    for attribute in ("text", "orig"):
        value = getattr(item, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _heading_level(item: Any) -> int:
    """Heading level, defaulting to 2 when Docling does not provide one."""
    level = getattr(item, "level", None)
    if level is None:
        return 2
    try:
        return max(1, min(8, int(level)))
    except (TypeError, ValueError):
        return 2


def _table_cells(item: Any) -> list[list[str]] | None:
    """Extract a table as a row-major grid of strings."""
    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None)
    if grid:
        return [[str(getattr(cell, "text", "") or "").strip() for cell in row] for row in grid]
    try:
        frame = item.export_to_dataframe()
    except Exception:  # noqa: BLE001 - pandas export is optional
        return None
    header = [str(c) for c in frame.columns]
    rows = [[str(v) for v in record] for record in frame.itertuples(index=False)]
    return [header, *rows]


def _caption_text(item: Any, document: Any) -> str | None:
    """Return an item's caption text when present."""
    try:
        caption = item.caption_text(document)
    except Exception:  # noqa: BLE001
        caption = None
    if isinstance(caption, str) and caption.strip():
        return caption.strip()
    return None


def _picture_bytes(item: Any, document: Any) -> bytes | None:
    """Return a picture item's PNG bytes."""
    try:
        image = item.get_image(document)
    except Exception:  # noqa: BLE001
        return None
    if image is None:
        return None
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _scanned_pages(pages: list[PageInfo]) -> set[int]:
    return {p.page for p in pages if p.scanned}


def _render_all(pdf_bytes: bytes) -> dict[int, bytes]:
    """Render every page to WebP using PyMuPDF (cheaper than Docling's renderer)."""
    from emrag.pdfcompat import fitz

    out: dict[int, bytes] = {}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc, start=1):
            out[index] = render_page(page)
    return out
