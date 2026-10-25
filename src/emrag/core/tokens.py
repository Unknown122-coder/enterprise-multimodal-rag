"""Token counting.

``cl100k_base`` is used as a provider-neutral approximation. It is within a few
percent of the Llama and Gemini tokenisers for English prose, which is accurate
enough for context budgeting and quota pre-estimation.

**The tokeniser is load-bearing, not cosmetic.** Chunk boundaries are decided by
token counts, chunk sequence numbers follow from those boundaries, and chunk
identifiers follow from the sequence numbers. Two deployments that count tokens
differently therefore produce different chunk ids for the same document, which
breaks the guarantee that a vector store is rebuildable from artifacts.

So the fallback here is deliberately loud. ``tiktoken`` fetches its vocabulary
over the network on first use, which on a cold free-tier host costs about a
minute, and behind an egress firewall fails outright. The Dockerfile bakes the
vocabulary into the image and :func:`warm_up` loads it during start-up, so
neither happens in a request. If the real tokeniser is nonetheless unavailable,
:func:`encoder_available` reports false, readiness says so, and a warning is
logged once rather than the system quietly switching to a different measure.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

ENCODING_NAME = "cl100k_base"
# Mean characters per token for English prose, used only by the fallback.
_FALLBACK_CHARS_PER_TOKEN = 4

_warned = False
_warn_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def _encoder() -> Any | None:
    """Load the tokeniser once, returning ``None`` when it is unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as exc:  # noqa: BLE001 - any failure means the fallback
        _warn_once(str(exc)[:200])
        return None


def _warn_once(reason: str) -> None:
    """Log the degradation a single time, with its consequence spelled out."""
    global _warned
    with _warn_lock:
        if _warned:
            return
        _warned = True
    from emrag.observability.logging import get_logger

    get_logger(__name__).error(
        "tokens.encoder_unavailable",
        encoding=ENCODING_NAME,
        reason=reason,
        consequence=(
            "falling back to a character heuristic; chunk boundaries and therefore "
            "chunk ids will differ from a deployment with the real tokeniser"
        ),
        remedy="bake the tiktoken cache into the image or set TIKTOKEN_CACHE_DIR",
    )


def encoder_available() -> bool:
    """Whether the real tokeniser is loaded, as opposed to the fallback."""
    return _encoder() is not None


def warm_up() -> bool:
    """Load the tokeniser during start-up. Returns whether it is the real one.

    Called from the application lifespan so the first request never pays the
    vocabulary load, and so a missing vocabulary is visible at boot.
    """
    return encoder_available()


def count_tokens(text: str) -> int:
    """Return an approximate token count for ``text``."""
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
    return len(enc.encode(text, disallowed_special=()))


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
        hard = enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])
    for sep in (". ", "! ", "? ", "\n"):
        idx = hard.rfind(sep)
        if idx > len(hard) * 0.6:
            return hard[: idx + 1].rstrip()
    return hard.rstrip()


def reset_for_tests() -> None:
    """Clear the cached encoder and the warning latch."""
    global _warned
    _encoder.cache_clear()
    _warned = False
