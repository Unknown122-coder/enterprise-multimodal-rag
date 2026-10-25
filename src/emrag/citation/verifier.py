"""Citation verification.

This is the component that makes the traceability guarantee real. No model is
trusted to cite honestly; every citation is checked mechanically:

1. the alias must be one actually shown to the model for this request, which
   makes an invented source impossible;
2. the quote must appear inside that block's source text, compared after the
   shared normalisation (so OCR spacing, curly quotes and hyphenated line breaks
   do not cause false rejections);
3. a near match above the configured threshold is accepted, and the returned
   quote is **replaced with the real span** from the source, so what the user
   sees is always text that exists in the document.

A claim that ends up with no verified citation is dropped. The answer that
reaches the user therefore cannot contain an unsupported sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import Indel

from emrag.citation.contract import EvidenceBundle
from emrag.core.textnorm import (
    normalize,
    normalize_display,
    normalize_with_offsets,
    word_count,
)
from emrag.core.types import Candidate
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 90
MIN_QUOTE_WORDS = 4
MAX_QUOTE_CHARS = 200


@dataclass
class VerifiedCitation:
    """One checked citation."""

    alias: str
    candidate: Candidate | None
    quote: str
    verified: bool
    match_score: float
    reason: str = ""


@dataclass
class VerifiedClaim:
    """One claim with only its surviving citations."""

    index: int
    text: str
    citations: list[VerifiedCitation] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        """True when at least one citation verified."""
        return any(c.verified for c in self.citations)


@dataclass
class VerificationReport:
    """Aggregate outcome of verifying a model answer."""

    claims: list[VerifiedClaim] = field(default_factory=list)
    dropped_claims: list[VerifiedClaim] = field(default_factory=list)
    rejected_citations: int = 0
    total_citations: int = 0

    @property
    def supported_claims(self) -> list[VerifiedClaim]:
        """Claims that kept at least one verified citation."""
        return [c for c in self.claims if c.supported]

    @property
    def honesty_rate(self) -> float:
        """Share of the model's citations that verified, before filtering."""
        if not self.total_citations:
            return 1.0
        return (self.total_citations - self.rejected_citations) / self.total_citations


class CitationVerifier:
    """Verifies model citations against the evidence actually retrieved."""

    def __init__(
        self,
        *,
        threshold: int = DEFAULT_THRESHOLD,
        min_quote_words: int = MIN_QUOTE_WORDS,
        max_quote_chars: int = MAX_QUOTE_CHARS,
    ) -> None:
        self._threshold = threshold
        self._min_words = min_quote_words
        self._max_chars = max_quote_chars

    def verify_answer(
        self, claims: list[dict[str, Any]], bundle: EvidenceBundle
    ) -> VerificationReport:
        """Verify every citation of every claim in a model answer."""
        report = VerificationReport()
        for index, raw_claim in enumerate(claims):
            text = str(raw_claim.get("text", "") or "").strip()
            if not text:
                continue
            claim = VerifiedClaim(index=index, text=normalize_display(text))
            for raw_citation in raw_claim.get("citations") or []:
                if not isinstance(raw_citation, dict):
                    continue
                report.total_citations += 1
                citation = self.verify_one(
                    str(raw_citation.get("id", "")),
                    str(raw_citation.get("quote", "")),
                    bundle,
                )
                if citation.verified:
                    claim.citations.append(citation)
                else:
                    report.rejected_citations += 1
                    logger.info(
                        "citation.rejected",
                        alias=citation.alias,
                        reason=citation.reason,
                        score=round(citation.match_score, 3),
                    )
            if claim.supported:
                report.claims.append(claim)
            else:
                report.dropped_claims.append(claim)
                logger.info("citation.claim_dropped", claim_index=index)
        return report

    def verify_one(self, alias: str, quote: str, bundle: EvidenceBundle) -> VerifiedCitation:
        """Verify a single ``(alias, quote)`` pair."""
        alias = alias.strip()
        candidate = bundle.alias_to_candidate.get(alias)
        if candidate is None:
            return VerifiedCitation(
                alias=alias,
                candidate=None,
                quote=quote[: self._max_chars],
                verified=False,
                match_score=0.0,
                reason="unknown_alias",
            )

        cleaned = normalize_display(quote)[: self._max_chars]
        if word_count(cleaned) < self._min_words:
            return VerifiedCitation(
                alias=alias,
                candidate=candidate,
                quote=cleaned,
                verified=False,
                match_score=0.0,
                reason="quote_too_short",
            )

        source = candidate.chunk.text
        matched, score = match_quote(cleaned, source, threshold=self._threshold)
        if matched is None:
            return VerifiedCitation(
                alias=alias,
                candidate=candidate,
                quote=cleaned,
                verified=False,
                match_score=score,
                reason="quote_not_in_source",
            )
        return VerifiedCitation(
            alias=alias,
            candidate=candidate,
            quote=matched,
            verified=True,
            match_score=score,
        )


def match_quote(
    quote: str, source: str, *, threshold: int = DEFAULT_THRESHOLD
) -> tuple[str | None, float]:
    """Locate ``quote`` inside ``source``.

    Returns the matching span **as it appears in the source** and a score in
    ``[0, 1]``. An exact normalised substring scores 1.0; otherwise the best
    fuzzy alignment is used, and only accepted above ``threshold``.
    """
    if not quote.strip() or not source.strip():
        return None, 0.0

    normal_quote = normalize(quote)
    normal_source, offsets = normalize_with_offsets(source)

    position = normal_source.find(normal_quote)
    if position >= 0:
        return _slice_original(source, offsets, position, len(normal_quote)), 1.0

    ratio = fuzz.partial_ratio(normal_quote, normal_source)
    if ratio < threshold:
        return None, ratio / 100.0

    window = _best_window(normal_quote, normal_source)
    if window is None:
        return None, ratio / 100.0
    start, length = window
    return _slice_original(source, offsets, start, length), ratio / 100.0


def _best_window(needle: str, haystack: str) -> tuple[int, int] | None:
    """Find the window of ``haystack`` closest to ``needle`` by edit distance."""
    length = len(needle)
    if length == 0 or len(haystack) < length // 2:
        return None
    best_start = None
    best_distance = None
    step = max(1, length // 8)
    for window_length in (length, int(length * 1.15) + 1):
        for start in range(0, max(1, len(haystack) - window_length + 1), step):
            window = haystack[start : start + window_length]
            distance = Indel.distance(needle, window)
            if best_distance is None or distance < best_distance:
                best_distance, best_start = distance, start
    if best_start is None:
        return None
    # Refine locally around the coarse best position.
    refine_from = max(0, best_start - step)
    refine_to = min(len(haystack) - 1, best_start + step)
    for start in range(refine_from, refine_to + 1):
        window = haystack[start : start + length]
        distance = Indel.distance(needle, window)
        if best_distance is None or distance < best_distance:
            best_distance, best_start = distance, start
    return best_start, length


def _slice_original(source: str, offsets: list[int], start: int, length: int) -> str:
    """Map a normalised-space span back onto the original text."""
    if not offsets:
        return source[:MAX_QUOTE_CHARS]
    start = max(0, min(start, len(offsets) - 1))
    end_index = min(start + length - 1, len(offsets) - 1)
    begin = offsets[start]
    finish = offsets[end_index] + 1
    return normalize_display(source[begin:finish])[:MAX_QUOTE_CHARS]
