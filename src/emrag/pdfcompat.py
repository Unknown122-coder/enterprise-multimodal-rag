"""PyMuPDF import shim.

PyMuPDF 1.25 renamed the module from ``fitz`` to ``pymupdf`` and emits a
FutureWarning on the old name. Importing through this shim keeps both versions
working and keeps the deprecation warning out of every log and test run.

The module is deliberately re-exported as ``Any``: PyMuPDF ships no type
annotations, so without this every call site would need a ``no-untyped-call``
suppression. Confining that to one line keeps the rest of the package strict.

Usage::

    from emrag.pdfcompat import fitz
"""

from __future__ import annotations

from typing import Any

try:  # PyMuPDF >= 1.25
    import pymupdf as _pymupdf
except ImportError:  # pragma: no cover - PyMuPDF < 1.25
    import fitz as _pymupdf  # type: ignore[no-redef]

fitz: Any = _pymupdf

__all__ = ["fitz"]
