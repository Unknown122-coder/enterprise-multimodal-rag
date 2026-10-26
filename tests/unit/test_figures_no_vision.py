"""Figures without a vision model must not become placeholder chunks.

Seen on the first real document: a one-page letter with a logo and a signature
produced two chunks reading "Figure (other). Figure on page 1." They carried no
information, competed with the real passage in retrieval, and could have been
cited. Without a vision model the right outcome is to keep the image in the
artifacts for a later full ingest and index nothing.
"""

from __future__ import annotations

import pytest

from emrag.core.ids import doc_id_for_bytes
from emrag.core.types import Element, PageInfo
from emrag.ingestion.chunking import ChunkingConfig, DocumentContext, chunk_document
from emrag.ingestion.figures import FigureDescription
from emrag.ingestion.pipeline import placeholder_descriptions


def _figure(page: int = 1, caption: str = "") -> Element:
    return Element(
        kind="figure",
        page=page,
        bbox=[0.1, 0.1, 0.4, 0.3],
        text=caption,
        image_bytes_ref=f"p{page:03d}_x7.png",
        element_id="f1",
        order=3,
    )


def test_uncaptioned_figure_is_decorative_without_vision() -> None:
    [desc] = placeholder_descriptions([_figure()])
    assert isinstance(desc, FigureDescription)
    assert desc.decorative is True


def test_captioned_figure_survives_without_vision() -> None:
    [desc] = placeholder_descriptions([_figure(caption="Figure 2: revenue by segment")])
    assert desc.decorative is False
    assert "revenue by segment" in desc.description


def test_chunker_drops_decorative_placeholders() -> None:
    doc_id = doc_id_for_bytes(b"no-vision")
    ctx = DocumentContext(
        doc_id=doc_id,
        tenant_id="demo",
        filename="letter.pdf",
        page_count=1,
        embed_model="t",
        sparse_model="t",
    )
    elements = [
        Element(
            kind="heading", page=1, bbox=[0.1, 0.05, 0.9, 0.1], text="Letter", level=1, order=1
        ),
        Element(
            kind="paragraph",
            page=1,
            bbox=[0.1, 0.2, 0.9, 0.5],
            text="This is to certify that the person held the role of engineer.",
            order=2,
        ),
        _figure(),
    ]
    result = chunk_document(
        elements,
        [PageInfo(page=1, width_pt=612, height_pt=792)],
        ctx,
        config=ChunkingConfig(),
        figure_descriptions=placeholder_descriptions([_figure()]),
    )
    assert [c.modality for c in result.chunks] == ["text"]
    assert result.figure_count == 0


@pytest.mark.parametrize("logit,expect", [(-2.1, True), (-8.0, False), (5.0, True)])
def test_default_gate_passes_real_relevant_logits(logit: float, expect: bool) -> None:
    # Calibrated on a real one-page letter: the correct passage scored -2.08,
    # the irrelevant ones below -10. The default must admit the former.
    import math

    from emrag.retrieval.retriever import RetrievalOptions

    tau = RetrievalOptions().gate_tau
    assert (1 / (1 + math.exp(-logit)) >= tau) is expect
