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

import importlib
from typing import Any

# importlib rather than two import statements: whether mypy sees the second
# import as a redefinition depends on which PyMuPDF modules exist in the
# environment, so a "type: ignore" there is needed on one machine and flagged
# as unused on another. This form needs none.
try:  # PyMuPDF >= 1.25
    _pymupdf = importlib.import_module("pymupdf")
except ImportError:  # pragma: no cover - PyMuPDF < 1.25
    _pymupdf = importlib.import_module("fitz")

fitz: Any = _pymupdf

__all__ = ["fitz"]
