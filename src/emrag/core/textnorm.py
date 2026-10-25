"""Text normalisation.

The *same* normalisation serves (a) content hashing and (b) citation quote
verification. Keeping one implementation is a build non-negotiable: if hashing
and verification disagreed, a quote could verify against text that was never
indexed.

:func:`normalize_with_offsets` is the primitive, and :func:`normalize` is defined
in terms of it, so the two can never drift. The offsets let the verifier report a
fuzzy match using the document's own characters rather than the normalised form.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Zero-width, bidirectional-override and soft-hyphen code points: carriers for
# text a reader cannot see but a model still reads.
_INVISIBLE: frozenset[int] = frozenset(
    {0x00AD, 0xFEFF, *range(0x200B, 0x2010), *range(0x202A, 0x202F), *range(0x2060, 0x2065)}
)
# Curly quotes and the dash family folded to ASCII, so a model paraphrase using
# straight punctuation still matches the source.
_FOLD: dict[int, str] = {
    0x2018: "'",
    0x2019: "'",
    0x201A: "'",
    0x201B: "'",
    0x201C: '"',
    0x201D: '"',
    0x201E: '"',
    **dict.fromkeys(range(8208, 8214), "-"),
}
_WS = re.compile(r"\s+")
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")


def normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalise ``text`` and record the source index of every output character.

    Pipeline: join hyphenated line breaks, NFKC, drop invisible characters, fold
    quotes and dashes, collapse whitespace runs to one space, casefold, and trim
    the ends. ``offsets[i]`` is the index in ``text`` that produced output
    character ``i``.
    """
    out: list[str] = []
    offsets: list[int] = []
    previous_space = True
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        # A hyphen that only exists because the word wrapped: drop it and the
        # whitespace that follows, so "rev-\nenue" normalises to "revenue".
        if char == "-" and _is_hyphen_break(text, index):
            index += 1
            while index < length and text[index].isspace():
                index += 1
            continue

        for piece in unicodedata.normalize("NFKC", char):
            code_point = ord(piece)
            if code_point in _INVISIBLE:
                continue
            if piece.isspace():
                if previous_space:
                    continue
                out.append(" ")
                offsets.append(index)
                previous_space = True
                continue
            folded = _FOLD.get(code_point, piece).casefold()
            for final in folded:
                out.append(final)
                offsets.append(index)
            previous_space = False
        index += 1

    while out and out[-1] == " ":
        out.pop()
        offsets.pop()
    return "".join(out), offsets


def normalize(text: str) -> str:
    """Return the canonical form of ``text`` for hashing and quote matching."""
    return normalize_with_offsets(text)[0]


def normalize_display(text: str) -> str:
    """Normalise whitespace and typography but preserve case, for shown text."""
    out = unicodedata.normalize("NFKC", text)
    out = _HYPHEN_BREAK.sub(r"\1\2", out)
    out = "".join(c for c in out if ord(c) not in _INVISIBLE)
    out = "".join(_FOLD.get(ord(c), c) for c in out)
    return _WS.sub(" ", out).strip()


def text_hash(text: str) -> str:
    """Return ``sha256:<hex>`` of the normalised text."""
    digest = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def word_count(text: str) -> int:
    """Count whitespace-separated words in the normalised text."""
    normalised = normalize(text)
    return len(normalised.split()) if normalised else 0


def _is_hyphen_break(text: str, index: int) -> bool:
    """True when the hyphen at ``index`` is a line-wrap artefact."""
    if index == 0 or not text[index - 1].isalnum():
        return False
    position = index + 1
    saw_newline = False
    while position < len(text) and text[position].isspace():
        if text[position] == "\n":
            saw_newline = True
        position += 1
    return saw_newline and position < len(text) and text[position].isalnum()
