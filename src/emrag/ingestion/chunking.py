"""Structure-aware chunker.

Rules that make citations exact:

* a child chunk never spans a page, so ``page`` on a citation is always precise;
* tables and figures are atomic chunks, never merged with prose;
* parents are heading-bounded sections, stored for context expansion but never
  embedded;
* the embedded text is prefixed with the heading breadcrumb while ``text`` keeps
  the raw passage, because quotes are verified against ``text``.

Chunk sequence numbers are document-wide and follow reading order, which keeps
``chunk_id`` stable across re-ingestion of identical bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from emrag import INGEST_VERSION
from emrag.core.ids import chunk_id as make_chunk_id
from emrag.core.ids import figure_id, table_id
from emrag.core.ids import parent_id as make_parent_id
from emrag.core.textnorm import normalize_display, text_hash
from emrag.core.tokens import count_tokens, truncate_to_tokens
from emrag.core.types import (
    Chunk,
    Classification,
    Element,
    PageInfo,
    Parent,
    PipelineProfile,
    utcnow,
)
from emrag.ingestion.figures import FigureDescription
from emrag.ingestion.tables import cells_to_markdown, split_table, summary_line
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_PROSE_KINDS = frozenset({"paragraph", "list_item", "ocr_block"})
_MAX_CHUNK_CHARS = 32_000


@dataclass
class ChunkingConfig:
    """Chunk size policy, loaded from ``configs/default.yaml``."""

    child_target_tokens: int = 350
    child_min_tokens: int = 80
    child_max_tokens: int = 512
    child_overlap_ratio: float = 0.15
    parent_max_tokens: int = 1500
    table_max_tokens: int = 800
    figure_max_tokens: int = 400

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ChunkingConfig:
        """Build a config from a YAML mapping, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DocumentContext:
    """Per-document values stamped onto every chunk."""

    doc_id: str
    tenant_id: str
    filename: str
    page_count: int
    title: str | None = None
    source_uri: str | None = None
    acl_groups: list[str] = field(default_factory=lambda: ["all"])
    classification: Classification = "internal"
    tags: list[str] = field(default_factory=list)
    pipeline_profile: PipelineProfile = "full"
    embed_model: str = "unknown"
    sparse_model: str = "unknown"
    lang: str = "en"
    ingest_version: str = INGEST_VERSION
    ingested_at: datetime = field(default_factory=utcnow)


@dataclass
class ChunkingResult:
    """Chunks plus the parents that contain them."""

    chunks: list[Chunk] = field(default_factory=list)
    parents: list[Parent] = field(default_factory=list)
    table_count: int = 0
    figure_count: int = 0


def chunk_document(
    elements: list[Element],
    pages: list[PageInfo],
    context: DocumentContext,
    *,
    config: ChunkingConfig | None = None,
    figure_descriptions: list[FigureDescription] | None = None,
) -> ChunkingResult:
    """Convert parsed elements into parent sections and citable child chunks."""
    cfg = config or ChunkingConfig()
    page_geometry = {p.page: p for p in pages}
    descriptions = {d.image_key: d for d in (figure_descriptions or [])}

    result = ChunkingResult()
    sequence = 0
    section_seq = 0
    heading_stack: list[tuple[int, str]] = []
    pending_prose: list[Element] = []
    current_parent: _ParentBuilder | None = None

    def section_path() -> list[str]:
        return [title for _, title in heading_stack][-8:]

    def flush_prose() -> None:
        nonlocal sequence, pending_prose
        if not pending_prose or current_parent is None:
            pending_prose = []
            return
        for page, group in _group_by_page(pending_prose):
            for text, bbox, confidence, element_ids in _pack_prose(group, cfg):
                sequence += 1
                chunk = _build_chunk(
                    context=context,
                    sequence=sequence,
                    page=page,
                    bbox=bbox,
                    bbox_confidence=confidence,
                    modality="ocr_text" if _is_ocr(group) else "text",
                    text=text,
                    section=section_path(),
                    parent=current_parent.parent_id,
                    element_ids=element_ids,
                    geometry=page_geometry.get(page),
                )
                if chunk is not None:
                    result.chunks.append(chunk)
                    current_parent.add(chunk, text)
        pending_prose = []

    def open_section(level: int, title: str) -> None:
        nonlocal section_seq, current_parent
        flush_prose()
        if current_parent is not None:
            result.parents.extend(current_parent.finish(cfg.parent_max_tokens))
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        section_seq += 1
        current_parent = _ParentBuilder(
            parent_id=make_parent_id(context.doc_id, section_seq),
            doc_id=context.doc_id,
            tenant_id=context.tenant_id,
            section_path=section_path(),
        )

    open_section(1, "Document")

    for element in elements:
        if element.kind == "heading" and element.text.strip():
            open_section(element.level or 2, normalize_display(element.text)[:200])
            continue
        if element.kind == "caption":
            pending_prose.append(element)
            continue
        if element.kind in _PROSE_KINDS:
            pending_prose.append(element)
            continue
        if element.kind == "table":
            flush_prose()
            assert current_parent is not None
            table_chunks, sequence = _table_chunks(
                element,
                context,
                cfg,
                section_path(),
                current_parent.parent_id,
                page_geometry,
                sequence,
            )
            for chunk in table_chunks:
                result.chunks.append(chunk)
                current_parent.add(chunk, chunk.text)
                result.table_count += 1
            continue
        if element.kind == "figure":
            flush_prose()
            assert current_parent is not None
            description = descriptions.get(element.image_bytes_ref or "")
            if description is not None and description.decorative:
                continue
            sequence += 1
            figure = _figure_chunk(
                element,
                description,
                context,
                cfg,
                section_path(),
                current_parent.parent_id,
                sequence,
                page_geometry,
            )
            if figure is not None:
                result.chunks.append(figure)
                current_parent.add(figure, figure.text)
                result.figure_count += 1
            continue

    flush_prose()
    if current_parent is not None:
        result.parents.extend(current_parent.finish(cfg.parent_max_tokens))

    _link_siblings(result.chunks)
    result.parents = [p for p in result.parents if p.child_ids]
    logger.info(
        "ingest.chunked",
        chunks=len(result.chunks),
        parents=len(result.parents),
        tables=result.table_count,
        figures=result.figure_count,
    )
    return result


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _ParentBuilder:
    """Accumulates a section's children and text."""

    def __init__(
        self, parent_id: str, doc_id: str, tenant_id: str, section_path: list[str]
    ) -> None:
        self.parent_id = parent_id
        self._doc_id = doc_id
        self._tenant_id = tenant_id
        self._section_path = list(section_path)
        self._child_ids: list[str] = []
        self._texts: list[str] = []
        self._page_start: int | None = None
        self._page_end: int | None = None

    def add(self, chunk: Chunk, text: str) -> None:
        """Register a child chunk."""
        self._child_ids.append(chunk.chunk_id)
        self._texts.append(text)
        self._page_start = (
            chunk.page if self._page_start is None else min(self._page_start, chunk.page)
        )
        self._page_end = chunk.page if self._page_end is None else max(self._page_end, chunk.page)

    def finish(self, max_tokens: int) -> list[Parent]:
        """Emit the parent, split into parts when the section is very long."""
        if not self._child_ids:
            return []
        full_text = "\n\n".join(self._texts)
        page_start = self._page_start or 1
        page_end = self._page_end or page_start
        if count_tokens(full_text) <= max_tokens:
            return [
                Parent(
                    parent_id=self.parent_id,
                    doc_id=self._doc_id,
                    tenant_id=self._tenant_id,
                    section_path=self._section_path,
                    page_start=page_start,
                    page_end=page_end,
                    text=full_text,
                    token_count=count_tokens(full_text),
                    child_ids=list(self._child_ids),
                )
            ]

        parts: list[Parent] = []
        buffer: list[str] = []
        buffer_ids: list[str] = []
        tokens = 0
        part_index = 1
        for text, child_id in zip(self._texts, self._child_ids, strict=True):
            size = count_tokens(text)
            if buffer and tokens + size > max_tokens:
                parts.append(self._part(part_index, buffer, buffer_ids, page_start, page_end))
                part_index += 1
                buffer, buffer_ids, tokens = [], [], 0
            buffer.append(text)
            buffer_ids.append(child_id)
            tokens += size
        if buffer:
            parts.append(self._part(part_index, buffer, buffer_ids, page_start, page_end))
        return parts

    def _part(
        self, index: int, texts: list[str], ids: list[str], page_start: int, page_end: int
    ) -> Parent:
        text = "\n\n".join(texts)
        return Parent(
            parent_id=f"{self.parent_id}:{index}",
            doc_id=self._doc_id,
            tenant_id=self._tenant_id,
            section_path=self._section_path,
            page_start=page_start,
            page_end=page_end,
            text=text,
            token_count=count_tokens(text),
            child_ids=list(ids),
        )


def _group_by_page(elements: list[Element]) -> list[tuple[int, list[Element]]]:
    """Group consecutive elements by page, preserving order."""
    groups: list[tuple[int, list[Element]]] = []
    for element in elements:
        if groups and groups[-1][0] == element.page:
            groups[-1][1].append(element)
        else:
            groups.append((element.page, [element]))
    return groups


def _pack_prose(
    elements: list[Element], cfg: ChunkingConfig
) -> list[tuple[str, list[float], float, list[str]]]:
    """Pack prose elements into target-sized chunks with sentence overlap."""
    packed: list[tuple[str, list[float], float, list[str]]] = []
    buffer: list[Element] = []
    tokens = 0

    def emit(group: list[Element], overlap: str = "") -> None:
        if not group:
            return
        body = "\n\n".join(normalize_display(e.text) for e in group if e.text.strip())
        if not body.strip():
            return
        text = f"{overlap}\n\n{body}".strip() if overlap else body
        text = truncate_to_tokens(text, cfg.child_max_tokens)[:_MAX_CHUNK_CHARS]
        packed.append(
            (
                text,
                _union_bbox(group),
                min((e.bbox_confidence for e in group), default=1.0),
                [e.element_id for e in group if e.element_id],
            )
        )

    overlap_text = ""
    for element in elements:
        size = count_tokens(element.text)
        if size > cfg.child_max_tokens:
            emit(buffer, overlap_text)
            overlap_text = _tail_sentences(buffer, cfg)
            buffer, tokens = [], 0
            for piece in _split_long(element, cfg):
                emit([piece], overlap_text)
                overlap_text = _tail_sentences([piece], cfg)
            continue
        if buffer and tokens + size > cfg.child_target_tokens:
            emit(buffer, overlap_text)
            overlap_text = _tail_sentences(buffer, cfg)
            buffer, tokens = [], 0
        buffer.append(element)
        tokens += size

    if buffer:
        if packed and tokens < cfg.child_min_tokens:
            # Merge a tiny trailing chunk backwards rather than emit a stub.
            last_text, last_bbox, last_conf, last_ids = packed.pop()
            merged = "\n\n".join(
                [last_text, *[normalize_display(e.text) for e in buffer if e.text.strip()]]
            )
            packed.append(
                (
                    truncate_to_tokens(merged, cfg.child_max_tokens)[:_MAX_CHUNK_CHARS],
                    _union_bbox_raw([last_bbox, _union_bbox(buffer)]),
                    min(last_conf, min((e.bbox_confidence for e in buffer), default=1.0)),
                    [*last_ids, *[e.element_id for e in buffer if e.element_id]],
                )
            )
        else:
            emit(buffer, overlap_text)
    return packed


def _split_long(element: Element, cfg: ChunkingConfig) -> list[Element]:
    """Split an oversized element on sentence boundaries."""
    sentences = _sentences(normalize_display(element.text))
    pieces: list[Element] = []
    buffer: list[str] = []
    tokens = 0
    for sentence in sentences:
        size = count_tokens(sentence)
        if buffer and tokens + size > cfg.child_target_tokens:
            pieces.append(element.model_copy(update={"text": " ".join(buffer)}))
            buffer, tokens = [], 0
        buffer.append(sentence)
        tokens += size
    if buffer:
        pieces.append(element.model_copy(update={"text": " ".join(buffer)}))
    return pieces or [element]


def _tail_sentences(elements: list[Element], cfg: ChunkingConfig) -> str:
    """Return the trailing sentences to repeat as overlap in the next chunk."""
    if cfg.child_overlap_ratio <= 0 or not elements:
        return ""
    budget = int(cfg.child_target_tokens * cfg.child_overlap_ratio)
    if budget <= 0:
        return ""
    text = normalize_display(elements[-1].text)
    out: list[str] = []
    total = 0
    for sentence in reversed(_sentences(text)):
        size = count_tokens(sentence)
        if total + size > budget and out:
            break
        out.insert(0, sentence)
        total += size
    return " ".join(out)


def _sentences(text: str) -> list[str]:
    """Split text into sentences without a heavyweight NLP dependency."""
    import re

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()] or ([text] if text.strip() else [])


def _is_ocr(elements: list[Element]) -> bool:
    return any(e.kind == "ocr_block" for e in elements)


def _union_bbox(elements: list[Element]) -> list[float]:
    """Union the boxes of a group of elements."""
    if not elements:
        return [0.0, 0.0, 1.0, 1.0]
    return _union_bbox_raw([list(e.bbox) for e in elements])


def _union_bbox_raw(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 1.0, 1.0]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _table_chunks(
    element: Element,
    context: DocumentContext,
    cfg: ChunkingConfig,
    section: list[str],
    parent: str,
    geometry: dict[int, PageInfo],
    start_sequence: int,
) -> tuple[list[Chunk], int]:
    """Build one chunk per table part, returning them and the new sequence counter."""
    cells = element.table_cells or ([[element.text]] if element.text.strip() else [])
    if not cells:
        return [], start_sequence
    identifier = table_id(context.doc_id, element.page, element.order % 100)
    parts = split_table(cells, max_tokens=cfg.table_max_tokens)
    caption = element.table_caption
    prefix = summary_line(cells, caption, section[-1] if section else None)

    out: list[Chunk] = []
    sequence = start_sequence
    for index, part in enumerate(parts, start=1):
        body = cells_to_markdown(part)
        text = f"{prefix}\n\n{body}"
        sequence += 1
        chunk = _build_chunk(
            context=context,
            sequence=sequence,
            page=element.page,
            bbox=list(element.bbox),
            bbox_confidence=element.bbox_confidence,
            modality="table",
            text=text,
            section=section,
            parent=parent,
            element_ids=[element.element_id] if element.element_id else [],
            geometry=geometry.get(element.page),
            table_identifier=identifier,
            table_part=f"{index}/{len(parts)}" if len(parts) > 1 else None,
            caption=caption,
            max_tokens=cfg.table_max_tokens,
        )
        if chunk is not None:
            out.append(chunk)
    return out, sequence


def _figure_chunk(
    element: Element,
    description: FigureDescription | None,
    context: DocumentContext,
    cfg: ChunkingConfig,
    section: list[str],
    parent: str,
    sequence: int,
    geometry: dict[int, PageInfo],
) -> Chunk | None:
    """Build the single chunk for a figure."""
    text = (
        description.to_text()
        if description is not None
        else (element.text or f"Figure on page {element.page}.")
    )
    if not text.strip():
        return None
    image_key = element.image_bytes_ref or figure_id(context.doc_id, element.page, sequence % 100)
    return _build_chunk(
        context=context,
        sequence=sequence,
        page=element.page,
        bbox=list(element.bbox),
        bbox_confidence=element.bbox_confidence,
        modality="figure",
        text=text,
        section=section,
        parent=parent,
        element_ids=[element.element_id] if element.element_id else [],
        geometry=geometry.get(element.page),
        image_ref=f"figures/{image_key}",
        caption=description.title if description else element.table_caption,
        max_tokens=cfg.figure_max_tokens,
    )


def _build_chunk(
    *,
    context: DocumentContext,
    sequence: int,
    page: int,
    bbox: list[float],
    bbox_confidence: float,
    modality: str,
    text: str,
    section: list[str],
    parent: str,
    element_ids: list[str],
    geometry: PageInfo | None,
    table_identifier: str | None = None,
    table_part: str | None = None,
    image_ref: str | None = None,
    caption: str | None = None,
    max_tokens: int = 512,
) -> Chunk | None:
    """Assemble and validate one chunk, returning ``None`` if it is empty."""
    body = truncate_to_tokens(text.strip(), max_tokens)[:_MAX_CHUNK_CHARS]
    if not body:
        return None
    try:
        return Chunk(
            chunk_id=make_chunk_id(context.doc_id, page, sequence),
            doc_id=context.doc_id,
            tenant_id=context.tenant_id,
            filename=context.filename,
            title=context.title,
            source_uri=context.source_uri,
            page=page,
            page_count=context.page_count,
            bbox=bbox,
            bbox_confidence=bbox_confidence,
            page_width_pt=geometry.width_pt if geometry else None,
            page_height_pt=geometry.height_pt if geometry else None,
            section_path=list(section),
            element_ids=element_ids,
            modality=modality,  # type: ignore[arg-type]
            text=body,
            text_hash=text_hash(body),
            token_count=max(1, min(1024, count_tokens(body))),
            lang=context.lang,
            table_id=table_identifier,
            table_part=table_part,
            image_ref=image_ref,
            caption=caption,
            parent_id=parent,
            acl_groups=list(context.acl_groups),
            classification=context.classification,
            ingest_version=context.ingest_version,
            pipeline_profile=context.pipeline_profile,
            embed_model=context.embed_model,
            sparse_model=context.sparse_model,
            ingested_at=context.ingested_at,
            tags=list(context.tags),
        )
    except Exception as exc:  # noqa: BLE001 - a single bad chunk must not fail the document
        logger.warning("ingest.chunk_rejected", page=page, sequence=sequence, error=str(exc)[:240])
        return None


def _link_siblings(chunks: list[Chunk]) -> None:
    """Fill ``prev_id`` and ``next_id`` in reading order."""
    for index, chunk in enumerate(chunks):
        chunk.prev_id = chunks[index - 1].chunk_id if index > 0 else None
        chunk.next_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None


def embedding_text(chunk: Chunk) -> str:
    """Return the contextual text actually sent to the embedding model."""
    breadcrumb = " > ".join(chunk.section_path)
    return f"{breadcrumb}\n\n{chunk.text}" if breadcrumb else chunk.text
