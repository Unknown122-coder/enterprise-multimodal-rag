"""Token counting.

``cl100k_base`` is used as a provider-neutral approximation. It is within a few
percent of Llama and Gemini tokenisers for English prose, which is accurate enough
for budgeting and quota pre-estimation.
"""

from __future__ import annotations

import functools

_FALLBACK_CHARS_PER_TOKEN = 4


@functools.lru_cache(maxsize=1)
def _encoder() -> object | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except (ImportError, OSError, ValueError):  # pragma: no cover - offline or no cache
        return None


def count_tokens(text: str) -> int:
    """Return an approximate token count for ``text``."""
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
    return len(enc.encode(text, disallowed_special=()))  # type: ignore[attr-defined]


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to ``max_tokens``, preferring a sentence boundary."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    enc = _encoder()
    if enc is None:
        hard = text[: max_tokens * _FALLBACK_CHARS_PER_TOKEN]
    else:
        ids = enc.encode(text, disallowed_special=())[:max_tokens]  # type: ignore[attr-defined]
        hard = enc.decode(ids)  # type: ignore[attr-defined]
    for sep in (". ", "! ", "? ", "\n"):
        idx = hard.rfind(sep)
        if idx > len(hard) * 0.6:
            return hard[: idx + 1].rstrip()
    return hard.rstrip()
