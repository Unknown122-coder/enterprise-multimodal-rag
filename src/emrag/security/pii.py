"""PII detection and tenant policy.

Detection is regex based, which is adequate for the structured identifiers that
appear in corporate documents and needs no extra dependency. Three policies:

``allow``  index as-is, flag the chunk so the UI can warn;
``mask``   replace identifiers *before* embedding, so neither the index nor the
           model ever sees them (quotes then verify against the masked text,
           which is why masking must happen before hashing);
``drop``   discard the chunk entirely.
"""

from __future__ import annotations

import re
from typing import Literal

from emrag.core.textnorm import text_hash
from emrag.core.tokens import count_tokens
from emrag.core.types import Chunk

PiiMode = Literal["allow", "mask", "drop"]

PII_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("email", "[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    (
        "phone",
        "[PHONE]",
        re.compile(
            r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]\d{3,4}(?!\d)"
        ),
    ),
    ("ssn", "[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("iban", "[IBAN]", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("credit_card", "[CARD]", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("aadhaar", "[ID]", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),
    ("pan", "[ID]", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
)


def detect(text: str) -> list[str]:
    """Return the names of PII kinds present in ``text``."""
    found: list[str] = []
    for name, _placeholder, pattern in PII_PATTERNS:
        if pattern.search(text):
            found.append(name)
    return found


def mask(text: str) -> tuple[str, list[str]]:
    """Replace every detected identifier with a placeholder."""
    masked = text
    found: list[str] = []
    for name, placeholder, pattern in PII_PATTERNS:
        masked, count = pattern.subn(placeholder, masked)
        if count:
            found.append(name)
    return masked, found


def apply_pii_policy(chunk: Chunk, mode: PiiMode) -> tuple[Chunk | None, list[str]]:
    """Apply a tenant's PII policy to one chunk.

    Returns the chunk (possibly rewritten) and the risk flags to record. A
    ``None`` chunk means the policy dropped it.
    """
    kinds = detect(chunk.text)
    if not kinds:
        return chunk, []

    flags = [f"pii:{k}" for k in kinds]
    if mode == "drop":
        return None, flags
    if mode == "mask":
        masked, _ = mask(chunk.text)
        chunk.text = masked
        chunk.text_hash = text_hash(masked)
        chunk.token_count = max(1, min(1024, count_tokens(masked)))
        chunk.pii = False
        return chunk, [*flags, "pii:masked"]

    chunk.pii = True
    return chunk, flags
