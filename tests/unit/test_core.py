"""Unit tests for identifier derivation, text normalisation and token budgeting."""

from __future__ import annotations

import pytest

from emrag.core import ids, textnorm, tokens


class TestIdentifiers:
    """Identifier derivation must be deterministic and validated."""

    def test_doc_id_is_content_addressed(self) -> None:
        first = ids.doc_id_for_bytes(b"hello world")
        second = ids.doc_id_for_bytes(b"hello world")
        assert first == second
        assert ids.DOC_ID_RE.match(first)
        assert first != ids.doc_id_for_bytes(b"hello worlds")

    def test_chunk_id_format_and_parsing(self) -> None:
        doc_id = ids.doc_id_for_bytes(b"x")
        chunk_id = ids.chunk_id(doc_id, 12, 7)
        assert chunk_id == f"{doc_id}:p012:c0007"
        assert ids.CHUNK_ID_RE.match(chunk_id)
        assert ids.doc_id_of_chunk(chunk_id) == doc_id
        assert ids.page_of_chunk(chunk_id) == 12

    def test_point_id_is_stable_uuid5(self) -> None:
        doc_id = ids.doc_id_for_bytes(b"x")
        chunk_id = ids.chunk_id(doc_id, 1, 1)
        assert ids.point_id(chunk_id) == ids.point_id(chunk_id)
        assert ids.point_id(chunk_id) != ids.point_id(ids.chunk_id(doc_id, 1, 2))

    @pytest.mark.parametrize(
        "doc_id,page,seq",
        [("bad", 1, 1), ("d_0000000000000000", 0, 1), ("d_0000000000000000", 1, 10_000)],
    )
    def test_chunk_id_rejects_bad_input(self, doc_id: str, page: int, seq: int) -> None:
        with pytest.raises(ids.IdError):
            ids.chunk_id(doc_id, page, seq)

    def test_parent_id_supports_parts(self) -> None:
        doc_id = ids.doc_id_for_bytes(b"x")
        assert ids.PARENT_ID_RE.match(ids.parent_id(doc_id, 5))
        assert ids.PARENT_ID_RE.match(ids.parent_id(doc_id, 5, 2))

    def test_tenant_id_validation(self) -> None:
        assert ids.validate_tenant_id("acme-corp") == "acme-corp"
        for bad in ("Acme", "a", "has_underscore", "-leading"):
            with pytest.raises(ids.IdError):
                ids.validate_tenant_id(bad)


class TestTextNormalisation:
    """Normalisation is shared by hashing and quote verification."""

    def test_collapses_whitespace_and_case(self) -> None:
        assert textnorm.normalize("  Hello   World \n") == "hello world"

    def test_folds_curly_quotes_and_dashes(self) -> None:
        curly = "the “annual” report — revised"
        straight = 'the "annual" report - revised'
        assert textnorm.normalize(curly) == textnorm.normalize(straight)

    def test_strips_invisible_characters(self) -> None:
        assert textnorm.normalize("rev​enue") == "revenue"

    def test_joins_hyphenated_line_breaks(self) -> None:
        assert textnorm.normalize("rev-\nenue grew") == "revenue grew"

    def test_hash_is_stable_and_prefixed(self) -> None:
        digest = textnorm.text_hash("Revenue grew 14%")
        assert digest.startswith("sha256:")
        assert digest == textnorm.text_hash("  revenue   grew 14%  ")

    def test_display_normalisation_preserves_case(self) -> None:
        assert textnorm.normalize_display("  Revenue   Grew ") == "Revenue Grew"

    def test_word_count(self) -> None:
        assert textnorm.word_count("Segment revenue grew 14 percent") == 5


class TestTokens:
    """Token counting and truncation back the context budget."""

    def test_count_is_monotonic(self) -> None:
        short = tokens.count_tokens("revenue")
        long = tokens.count_tokens("revenue " * 50)
        assert 0 < short < long

    def test_empty_text_is_zero(self) -> None:
        assert tokens.count_tokens("") == 0

    def test_truncate_respects_budget(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        truncated = tokens.truncate_to_tokens(text, 6)
        assert tokens.count_tokens(truncated) <= 6
        assert truncated

    def test_truncate_noop_when_within_budget(self) -> None:
        text = "Short enough."
        assert tokens.truncate_to_tokens(text, 100) == text

    def test_truncate_prefers_sentence_boundary(self) -> None:
        text = "Alpha beta gamma delta. Epsilon zeta eta theta iota kappa lambda mu nu."
        truncated = tokens.truncate_to_tokens(text, 8)
        assert truncated.endswith(".") or truncated == text


class TestNormaliseWithOffsets:
    """The offset-tracking primitive must agree with ``normalize`` exactly."""

    @pytest.mark.parametrize(
        "text",
        [
            "Simple text",
            "  Leading and trailing   ",
            "rev-\nenue grew 14%",
            "the “annual” report — revised",
            "zero​width​joined",
            "tabs\tand\nnewlines   collapsed",
            "",
            "   ",
        ],
    )
    def test_matches_normalize(self, text: str) -> None:
        normalised, offsets = textnorm.normalize_with_offsets(text)
        assert normalised == textnorm.normalize(text)
        assert len(offsets) == len(normalised)
        assert all(0 <= o < max(1, len(text)) for o in offsets)

    def test_offsets_are_non_decreasing(self) -> None:
        _, offsets = textnorm.normalize_with_offsets("Revenue  grew\n14 percent")
        assert offsets == sorted(offsets)

    def test_offsets_point_at_the_right_source_span(self) -> None:
        source = "Revenue grew 14 percent year over year."
        normalised, offsets = textnorm.normalize_with_offsets(source)
        start = normalised.index("14 percent")
        end = start + len("14 percent") - 1
        assert source[offsets[start] : offsets[end] + 1] == "14 percent"
