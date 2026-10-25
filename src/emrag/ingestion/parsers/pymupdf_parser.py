"""PyMuPDF parser.

Always available (pure wheel, no model downloads) and therefore the fallback for
the whole chain as well as the engine for the lite ingestion path. It recovers
text blocks with coordinates, infers headings from font size, extracts raster
images and renders page previews.

Layout fidelity is lower than Docling's: there is no learned reading order and no
table structure. Documents parsed this way are flagged ``quality: degraded``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from emrag.core.types import Element, PageInfo
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_MIN_BLOCK_CHARS = 2
_HEADING_SIZE_RATIO = 1.18
_PREVIEW_DPI = 110


@dataclass
class ParseResult:
    """Everything a parser produces for one document."""

    elements: list[Element] = field(default_factory=list)
    pages: list[PageInfo] = field(default_factory=list)
    images: dict[str, bytes] = field(default_factory=dict)
    page_renders: dict[int, bytes] = field(default_factory=dict)
    engine: str = "pymupdf"
    degraded_reasons: list[str] = field(default_factory=list)


def parse_pdf(
    pdf_bytes: bytes,
    *,
    pages: list[PageInfo] | None = None,
    render_pages: bool = True,
    extract_images: bool = True,
    max_images_per_page: int = 12,
) -> ParseResult:
    """Parse a PDF into ordered elements with normalised bounding boxes."""
    from emrag.pdfcompat import fitz

    result = ParseResult(engine="pymupdf")
    page_infos = {p.page: p for p in (pages or [])}
    order = 0

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        median_size = _median_font_size(doc)
        for index, page in enumerate(doc, start=1):
            rect = page.rect
            width, height = float(rect.width), float(rect.height)
            info = page_infos.get(index)
            if info is None:
                text = page.get_text("text") or ""
                info = PageInfo(
                    page=index, width_pt=width, height_pt=height, char_count=len(text.strip())
                )
            result.pages.append(info)

            for block in _text_blocks(page):
                text = block["text"].strip()
                if len(text) < _MIN_BLOCK_CHARS:
                    continue
                order += 1
                is_heading = (
                    block["max_size"] >= median_size * _HEADING_SIZE_RATIO
                    and len(text) < 160
                    and text.count("\n") == 0
                )
                result.elements.append(
                    Element(
                        kind="heading" if is_heading else "paragraph",
                        page=index,
                        bbox=_normalise(block["bbox"], width, height),
                        text=text,
                        level=_heading_level(block["max_size"], median_size)
                        if is_heading
                        else None,
                        element_id=f"pymupdf:p{index}:b{block['index']}",
                        order=order,
                    )
                )

            if extract_images:
                for seq, (image_key, image_bytes, bbox) in enumerate(
                    _page_images(doc, page, index, max_images_per_page)
                ):
                    order += 1
                    result.images[image_key] = image_bytes
                    result.elements.append(
                        Element(
                            kind="figure",
                            page=index,
                            bbox=_normalise(bbox, width, height),
                            text="",
                            element_id=f"pymupdf:p{index}:i{seq}",
                            image_bytes_ref=image_key,
                            order=order,
                        )
                    )

            if render_pages:
                result.page_renders[index] = render_page(page, dpi=_PREVIEW_DPI)

    result.elements.sort(key=lambda e: (e.page, e.order))
    return result


def render_page(page: Any, *, dpi: int = _PREVIEW_DPI) -> bytes:
    """Render one page to WebP bytes for the citation viewer."""
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    png_bytes: bytes = pixmap.tobytes("png")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(png_bytes)) as img:
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="WEBP", quality=72, method=4)
            return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - Pillow/WebP may be unavailable
        logger.debug("ingest.webp_unavailable", error=str(exc)[:160])
        return png_bytes


def render_page_at(pdf_bytes: bytes, page_number: int, *, dpi: int = _PREVIEW_DPI) -> bytes:
    """Render a single page of a PDF held in memory."""
    from emrag.pdfcompat import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if not 1 <= page_number <= doc.page_count:
            raise ValueError(f"page {page_number} out of range (1..{doc.page_count})")
        return render_page(doc[page_number - 1], dpi=dpi)


def _text_blocks(page: Any) -> list[dict[str, Any]]:
    """Return text blocks with their maximum font size, in layout order."""
    blocks: list[dict[str, Any]] = []
    data = page.get_text("dict")
    for index, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        max_size = 0.0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            lines.append("".join(span.get("text", "") for span in spans))
            for span in spans:
                max_size = max(max_size, float(span.get("size", 0.0)))
        text = "\n".join(lines)
        if text.strip():
            blocks.append(
                {
                    "index": index,
                    "text": text,
                    "bbox": tuple(block.get("bbox", (0, 0, 0, 0))),
                    "max_size": max_size,
                }
            )
    blocks.sort(key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
    return blocks


def _page_images(
    doc: Any, page: Any, page_number: int, limit: int
) -> list[tuple[str, bytes, tuple[float, float, float, float]]]:
    """Extract embedded raster images with their placement rectangles."""
    out: list[tuple[str, bytes, tuple[float, float, float, float]]] = []
    seen: set[int] = set()
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:  # noqa: BLE001
        infos = []
    for info in infos:
        if len(out) >= limit:
            break
        xref = int(info.get("xref", 0))
        if xref <= 0 or xref in seen:
            continue
        seen.add(xref)
        try:
            raw = doc.extract_image(xref)
        except Exception as exc:  # noqa: BLE001 - broken image streams are common
            logger.debug("ingest.image_extract_failed", xref=xref, error=str(exc)[:120])
            continue
        image_bytes = raw.get("image")
        if not image_bytes:
            continue
        bbox = tuple(float(v) for v in info.get("bbox", (0, 0, 0, 0)))
        key = f"p{page_number:03d}_x{xref}.{raw.get('ext', 'png')}"
        out.append((key, image_bytes, bbox))  # type: ignore[arg-type]
    return out


def _median_font_size(doc: Any) -> float:
    """Median span size across a sample of pages, used as the body-text baseline."""
    sizes: list[float] = []
    sample = min(doc.page_count, 12)
    for index in range(sample):
        data = doc[index].get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = float(span.get("size", 0.0))
                    if size > 0:
                        sizes.append(size)
    if not sizes:
        return 11.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _heading_level(size: float, median: float) -> int:
    """Map a font size to a heading level between 1 and 4."""
    ratio = size / max(median, 1e-6)
    if ratio >= 1.8:
        return 1
    if ratio >= 1.5:
        return 2
    if ratio >= 1.3:
        return 3
    return 4


def _normalise(bbox: tuple[float, ...], width: float, height: float) -> list[float]:
    """Convert a PDF-point bbox to page fractions, clamped to [0, 1]."""
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 1.0, 1.0]
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    out = [
        min(max(x0 / width, 0.0), 1.0),
        min(max(y0 / height, 0.0), 1.0),
        min(max(x1 / width, 0.0), 1.0),
        min(max(y1 / height, 0.0), 1.0),
    ]
    if out[0] > out[2]:
        out[0], out[2] = out[2], out[0]
    if out[1] > out[3]:
        out[1], out[3] = out[3], out[1]
    return out
