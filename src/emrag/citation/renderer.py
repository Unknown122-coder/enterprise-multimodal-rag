"""Answer rendering.

Turns verified claims into the response the client sees: markdown with numbered
references, a deduplicated citation list, and page or figure URLs so a viewer can
show the exact region of the source page.

Numbering is assigned here, not by the model, so references are always dense and
in reading order regardless of what the model emitted.
"""

from __future__ import annotations

from emrag.citation.verifier import VerifiedClaim
from emrag.core.types import Answer, Candidate, Citation, Claim

PAGE_URL = "/v1/documents/{doc_id}/pages/{page}"
FIGURE_URL = "/v1/documents/{doc_id}/figures/{image_ref}"


def render(
    claims: list[VerifiedClaim], *, include_page_images: bool = True
) -> tuple[Answer, list[Citation]]:
    """Render verified claims into an answer plus its citation list."""
    citations: list[Citation] = []
    by_key: dict[tuple[str, str], int] = {}
    rendered_claims: list[Claim] = []
    sentences: list[str] = []

    for position, claim in enumerate(claims):
        refs: list[int] = []
        for citation in claim.citations:
            if not citation.verified or citation.candidate is None:
                continue
            key = (citation.candidate.chunk.chunk_id, citation.quote)
            ref = by_key.get(key)
            if ref is None:
                ref = len(citations) + 1
                by_key[key] = ref
                citations.append(
                    _to_citation(
                        ref,
                        citation.candidate,
                        citation.quote,
                        citation.match_score,
                        include_page_images=include_page_images,
                    )
                )
            if ref not in refs:
                refs.append(ref)

        text = claim.text.strip()
        if not text:
            continue
        markers = "".join(f" [{r}]" for r in refs)
        sentences.append(f"{text}{markers}")
        rendered_claims.append(Claim(index=position, text=text, citation_refs=refs))

    return Answer(markdown=" ".join(sentences).strip(), claims=rendered_claims), citations


def render_nearest(
    candidates: list[Candidate], *, limit: int = 3, include_page_images: bool = True
) -> list[Citation]:
    """Render the best non-answering candidates for a ``not_found`` response.

    No quotes are attached: nothing was verified, so nothing is presented as
    supporting text. Only locations are offered, so the user can look themselves.
    """
    out: list[Citation] = []
    for index, candidate in enumerate(candidates[:limit], start=1):
        citation = _to_citation(
            index, candidate, None, 0.0, include_page_images=include_page_images
        )
        citation.verified = False
        out.append(citation)
    return out


def _to_citation(
    ref: int,
    candidate: Candidate,
    quote: str | None,
    match_score: float,
    *,
    include_page_images: bool,
) -> Citation:
    """Build one citation object from a candidate chunk."""
    chunk = candidate.chunk
    page_url = None
    if include_page_images:
        bbox = ",".join(f"{c:.4f}" for c in chunk.bbox)
        page_url = PAGE_URL.format(doc_id=chunk.doc_id, page=chunk.page) + f"?bbox={bbox}"
    image_url = None
    if chunk.modality == "figure" and chunk.image_ref:
        reference = chunk.image_ref.split("/")[-1]
        image_url = FIGURE_URL.format(doc_id=chunk.doc_id, image_ref=reference)

    return Citation(
        ref=ref,
        chunk_id=chunk.chunk_id,
        document_id=chunk.doc_id,
        filename=chunk.filename,
        title=chunk.title,
        page=chunk.page,
        bbox=list(chunk.bbox),
        modality=chunk.modality,
        section_path=list(chunk.section_path),
        quote=quote,
        verified=quote is not None,
        match_score=round(match_score, 4),
        retrieval_score=round(candidate.effective_score, 4),
        page_image_url=page_url,
        image_url=image_url,
    )


def renumber_streamed(body: str, alias_to_ref: dict[str, int]) -> str:
    """Rewrite inline ``[c3]`` markers in streamed markdown as dense ``[1]`` refs."""
    import re

    def _swap(match: re.Match[str]) -> str:
        alias = match.group(1)
        ref = alias_to_ref.get(alias)
        return f" [{ref}]" if ref else ""

    return re.sub(r"\s*\[([cs]\d{1,2})\]", _swap, body).strip()
