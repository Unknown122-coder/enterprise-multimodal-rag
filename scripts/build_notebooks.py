"""Convert the ``py:percent`` notebook sources into ``.ipynb`` files.

The ``.py`` files are the source of truth because they diff and review cleanly;
the ``.ipynb`` files are the artefacts Colab opens. Running this keeps the two in
step without taking a jupytext dependency.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
MARKDOWN_MARKER = "# %% [markdown]"
CODE_MARKER = "# %%"


def split_cells(source: str) -> list[tuple[str, str]]:
    """Split percent-format source into ``(kind, text)`` cells."""
    cells: list[tuple[str, str]] = []
    kind = "code"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip("\n")
        if text.strip():
            cells.append((kind, text))

    for line in source.splitlines():
        stripped = line.rstrip()
        if stripped == MARKDOWN_MARKER or stripped.startswith(MARKDOWN_MARKER + " "):
            flush()
            kind, buffer = "markdown", []
            continue
        if stripped == CODE_MARKER or stripped.startswith(CODE_MARKER + " "):
            flush()
            kind, buffer = "code", []
            continue
        buffer.append(line)
    flush()
    return cells


def to_notebook(cells: list[tuple[str, str]]) -> dict[str, object]:
    """Build a minimal nbformat 4 document."""
    out: list[dict[str, object]] = []
    for kind, text in cells:
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#").lstrip()
                for line in text.splitlines()
            )
            out.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": _lines(body),
                }
            )
        else:
            out.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": _lines(text),
                }
            )
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _lines(text: str) -> list[str]:
    """Split text into nbformat source lines, each keeping its newline."""
    raw = text.splitlines()
    return [line + "\n" for line in raw[:-1]] + ([raw[-1]] if raw else [])


def main() -> int:
    """Convert every ``*.py`` notebook source in ``notebooks/``."""
    sources = sorted(NOTEBOOK_DIR.glob("*.py"))
    if not sources:
        sys.stderr.write(f"no notebook sources found in {NOTEBOOK_DIR}\n")
        return 2
    for source in sources:
        notebook = to_notebook(split_cells(source.read_text(encoding="utf-8")))
        target = source.with_suffix(".ipynb")
        target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        sys.stdout.write(f"{source.name} -> {target.name} ({len(notebook['cells'])} cells)\n")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
