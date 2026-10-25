"""Table serialisation.

Each table becomes GitHub-flavoured markdown (what gets embedded and shown to the
model) plus a structured JSON form kept in the artifacts. A short synthetic
summary line is prepended before embedding, because a bare markdown grid
retrieves poorly for natural-language questions.

Large tables are split by row groups, repeating the header on every part so each
part is independently citable and independently answerable.
"""

from __future__ import annotations

from typing import Any

from emrag.core.tokens import count_tokens


def cells_to_markdown(cells: list[list[str]]) -> str:
    """Render a row-major grid as a markdown table."""
    if not cells:
        return ""
    width = max(len(row) for row in cells)
    padded = [[_clean(cell) for cell in row] + [""] * (width - len(row)) for row in cells]
    header, *body = padded
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def summary_line(cells: list[list[str]], caption: str | None, section: str | None) -> str:
    """Build the retrieval prefix for a table."""
    label = caption or section or "data table"
    columns = ", ".join(_clean(c) for c in (cells[0] if cells else []) if _clean(c))
    rows = max(0, len(cells) - 1)
    parts = [f"Table: {label}."]
    if columns:
        parts.append(f"Columns: {columns}.")
    parts.append(f"{rows} data rows.")
    return " ".join(parts)


def to_structured(cells: list[list[str]], caption: str | None = None) -> dict[str, Any]:
    """Return the JSON form stored in ``elements.json``."""
    if not cells:
        return {"columns": [], "rows": [], "caption": caption, "n_rows": 0, "n_cols": 0}
    header, *body = cells
    columns = [_clean(c) for c in header]
    rows = [[_clean(c) for c in row] for row in body]
    return {
        "columns": columns,
        "rows": rows,
        "caption": caption,
        "n_rows": len(rows),
        "n_cols": len(columns),
    }


def split_table(cells: list[list[str]], *, max_tokens: int = 800) -> list[list[list[str]]]:
    """Split a table into parts under ``max_tokens``, repeating the header.

    Returns a single-element list when the table already fits.
    """
    if not cells:
        return []
    if count_tokens(cells_to_markdown(cells)) <= max_tokens:
        return [cells]

    header, *body = cells
    header_tokens = count_tokens(cells_to_markdown([header]))
    budget = max(1, max_tokens - header_tokens)

    parts: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_tokens = 0
    for row in body:
        row_tokens = count_tokens(" | ".join(_clean(c) for c in row)) + 2
        if current and current_tokens + row_tokens > budget:
            parts.append([header, *current])
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += row_tokens
    if current:
        parts.append([header, *current])
    return parts


def extract_with_pdfplumber(pdf_bytes: bytes, page_number: int) -> list[list[list[str]]]:
    """Fallback table extraction for native PDFs when TableFormer is unsure.

    Returns a list of tables, each a row-major grid. Empty when pdfplumber is not
    installed (it is an ingest-only dependency).
    """
    try:
        import io

        import pdfplumber
    except ImportError:
        return []

    tables: list[list[list[str]]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if not 1 <= page_number <= len(pdf.pages):
            return []
        page = pdf.pages[page_number - 1]
        for raw in page.extract_tables() or []:
            grid = [[_clean(cell or "") for cell in row] for row in raw if any(row)]
            if len(grid) >= 2:
                tables.append(grid)
    return tables


def _clean(value: Any) -> str:
    """Flatten a cell value to single-line markdown-safe text."""
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ").strip()
