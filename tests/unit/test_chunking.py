"""Unit tests for the structure-aware chunker and table serialisation."""

from __future__ import annotations

from emrag.core.ids import doc_id_for_bytes
from emrag.core.textnorm import text_hash
from emrag.core.types import Element, PageInfo
from emrag.ingestion import tables
from emrag.ingestion.chunking import (
    ChunkingConfig,
    DocumentContext,
    chunk_document,
    embedding_text,
)
from emrag.ingestion.figures import FigureDescription
from emrag.ingestion.metadata import check_invariants

DOC_ID = doc_id_for_bytes(b"chunking-fixture")


def _context(page_count: int = 2) -> DocumentContext:
    return DocumentContext(
        doc_id=DOC_ID,
        tenant_id="demo",
        filename="report.pdf",
        page_count=page_count,
        title="Report",
        embed_model="test-embed",
        sparse_model="test-sparse",
    )


def _pages(count: int = 2) -> list[PageInfo]:
    return [PageInfo(page=i, width_pt=612.0, height_pt=792.0) for i in range(1, count + 1)]


def _paragraph(page: int, order: int, text: str, top: float = 0.1) -> Element:
    return Element(
        kind="paragraph",
        page=page,
        bbox=[0.1, top, 0.9, min(1.0, top + 0.08)],
        text=text,
        element_id=f"e{page}-{order}",
        order=order,
    )


def _heading(page: int, order: int, text: str, level: int = 1) -> Element:
    return Element(
        kind="heading",
        page=page,
        bbox=[0.1, 0.05, 0.9, 0.09],
        text=text,
        level=level,
        element_id=f"h{page}-{order}",
        order=order,
    )


class TestProseChunking:
    """Prose chunking must stay inside one page and carry section context."""

    def test_chunks_never_span_pages(self) -> None:
        elements = [
            _heading(1, 1, "Section One"),
            _paragraph(1, 2, "Alpha sentence about revenue. " * 6),
            _paragraph(2, 3, "Beta sentence about margin. " * 6),
        ]
        result = chunk_document(elements, _pages(), _context(), config=ChunkingConfig())
        assert result.chunks
        pages = {c.page for c in result.chunks}
        assert pages == {1, 2}
        for chunk in result.chunks:
            assert chunk.chunk_id.startswith(DOC_ID)
            assert chunk.page in (1, 2)

    def test_section_path_tracks_heading_hierarchy(self) -> None:
        elements = [
            _heading(1, 1, "3 Financial Review", level=1),
            _heading(1, 2, "3.2 Segment Results", level=2),
            _paragraph(1, 3, "Segment revenue grew 14 percent year over year in this period."),
        ]
        result = chunk_document(elements, _pages(1), _context(1))
        assert result.chunks
        path = result.chunks[-1].section_path
        assert path[-2:] == ["3 Financial Review", "3.2 Segment Results"]

    def test_embedding_text_prefixes_breadcrumb(self) -> None:
        elements = [
            _heading(1, 1, "Outlook"),
            _paragraph(1, 2, "Management expects revenue between 188 and 194 million dollars."),
        ]
        result = chunk_document(elements, _pages(1), _context(1))
        chunk = result.chunks[-1]
        prepared = embedding_text(chunk)
        assert prepared.startswith("Document > Outlook") or "Outlook" in prepared
        assert chunk.text in prepared

    def test_oversized_paragraph_is_split(self) -> None:
        long_text = " ".join(f"Sentence number {i} about the quarter." for i in range(120))
        result = chunk_document(
            [_heading(1, 1, "Long"), _paragraph(1, 2, long_text)],
            _pages(1),
            _context(1),
            config=ChunkingConfig(child_target_tokens=80, child_max_tokens=120),
        )
        assert len(result.chunks) > 1
        assert all(c.token_count <= 120 for c in result.chunks)

    def test_text_hash_matches_text(self) -> None:
        result = chunk_document(
            [_heading(1, 1, "H"), _paragraph(1, 2, "Revenue reached 48.2 million dollars.")],
            _pages(1),
            _context(1),
        )
        for chunk in result.chunks:
            assert chunk.text_hash == text_hash(chunk.text)

    def test_siblings_are_linked_in_reading_order(self) -> None:
        elements = [_heading(1, 1, "H")] + [
            _paragraph(1, i + 2, f"Paragraph {i} with enough words to form its own chunk. " * 8)
            for i in range(3)
        ]
        result = chunk_document(
            elements,
            _pages(1),
            _context(1),
            config=ChunkingConfig(child_target_tokens=40, child_max_tokens=90, child_min_tokens=10),
        )
        assert len(result.chunks) > 1
        assert result.chunks[0].prev_id is None
        assert result.chunks[-1].next_id is None
        for earlier, later in zip(result.chunks, result.chunks[1:], strict=False):
            assert earlier.next_id == later.chunk_id
            assert later.prev_id == earlier.chunk_id

    def test_invariants_hold(self) -> None:
        elements = [
            _heading(1, 1, "Section"),
            _paragraph(1, 2, "First body paragraph with several words in it."),
            _paragraph(2, 3, "Second body paragraph on the following page."),
        ]
        result = chunk_document(elements, _pages(), _context())
        assert check_invariants(result.chunks, page_count=2) == []


class TestTableChunking:
    """Tables are atomic, prefixed for retrieval, and split with repeated headers."""

    def _table_element(self, rows: int = 3) -> Element:
        cells = [["Segment", "Revenue", "Growth"]]
        for i in range(rows):
            cells.append([f"Segment {i}", f"{10 + i}.5", f"{i}%"])
        return Element(
            kind="table",
            page=1,
            bbox=[0.1, 0.2, 0.9, 0.6],
            table_cells=cells,
            table_caption="Segment revenue by line of business",
            element_id="t1",
            order=2,
        )

    def test_table_becomes_its_own_chunk(self) -> None:
        result = chunk_document(
            [_heading(1, 1, "Segments"), self._table_element()], _pages(1), _context(1)
        )
        table_chunks = [c for c in result.chunks if c.modality == "table"]
        assert len(table_chunks) == 1
        chunk = table_chunks[0]
        assert chunk.table_id
        assert "Segment revenue by line of business" in chunk.text
        assert "| Segment | Revenue | Growth |" in chunk.text
        assert result.table_count == 1

    def test_large_table_splits_and_repeats_header(self) -> None:
        element = self._table_element(rows=120)
        result = chunk_document(
            [_heading(1, 1, "Segments"), element],
            _pages(1),
            _context(1),
            config=ChunkingConfig(table_max_tokens=200),
        )
        parts = [c for c in result.chunks if c.modality == "table"]
        assert len(parts) > 1
        assert {c.table_id for c in parts} == {parts[0].table_id}
        assert all(c.table_part for c in parts)
        assert all("| Segment | Revenue | Growth |" in c.text for c in parts)
        assert {c.chunk_id for c in parts}.__len__() == len(parts)

    def test_markdown_escapes_pipes(self) -> None:
        rendered = tables.cells_to_markdown([["a|b"], ["c"]])
        assert "a\\|b" in rendered

    def test_structured_form(self) -> None:
        structured = tables.to_structured([["A", "B"], ["1", "2"]], caption="Cap")
        assert structured == {
            "columns": ["A", "B"],
            "rows": [["1", "2"]],
            "caption": "Cap",
            "n_rows": 1,
            "n_cols": 2,
        }


class TestFigureChunking:
    """Figures become one chunk whose text is the generated description."""

    def _figure_element(self) -> Element:
        return Element(
            kind="figure",
            page=1,
            bbox=[0.2, 0.3, 0.8, 0.7],
            image_bytes_ref="p001_fig1.png",
            element_id="f1",
            order=2,
        )

    def test_described_figure_is_chunked_with_image_ref(self) -> None:
        description = FigureDescription(
            image_key="p001_fig1.png",
            page=1,
            bbox=[0.2, 0.3, 0.8, 0.7],
            kind="chart",
            title="Revenue by quarter",
            description="A bar chart of quarterly revenue.",
            data_points=[{"label": "Q3", "value": "48.2", "unit": "USD m"}],
        )
        result = chunk_document(
            [_heading(1, 1, "Charts"), self._figure_element()],
            _pages(1),
            _context(1),
            figure_descriptions=[description],
        )
        figures = [c for c in result.chunks if c.modality == "figure"]
        assert len(figures) == 1
        assert figures[0].image_ref == "figures/p001_fig1.png"
        assert "Revenue by quarter" in figures[0].text
        assert "48.2" in figures[0].text

    def test_decorative_figure_is_dropped(self) -> None:
        description = FigureDescription(
            image_key="p001_fig1.png",
            page=1,
            bbox=[0.2, 0.3, 0.8, 0.7],
            kind="other",
            description="Company logo, decorative.",
            decorative=True,
        )
        result = chunk_document(
            [_heading(1, 1, "Charts"), self._figure_element()],
            _pages(1),
            _context(1),
            figure_descriptions=[description],
        )
        assert not [c for c in result.chunks if c.modality == "figure"]


class TestParents:
    """Parents cover their children and are never embedded."""

    def test_parents_reference_their_children(self) -> None:
        elements = [
            _heading(1, 1, "One"),
            _paragraph(1, 2, "Body of section one with sufficient length to chunk. " * 4),
            _heading(2, 3, "Two"),
            _paragraph(2, 4, "Body of section two with sufficient length to chunk. " * 4),
        ]
        result = chunk_document(elements, _pages(), _context())
        assert len(result.parents) >= 2
        child_ids = {c.chunk_id for c in result.chunks}
        for parent in result.parents:
            assert parent.child_ids
            assert set(parent.child_ids) <= child_ids
            assert parent.parent_id.startswith(DOC_ID)
        parent_ids = {p.parent_id for p in result.parents}
        for chunk in result.chunks:
            assert any(chunk.parent_id.startswith(p.split(":")[0]) for p in parent_ids)
