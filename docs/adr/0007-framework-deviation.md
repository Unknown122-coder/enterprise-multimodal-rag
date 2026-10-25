# 0007 The runtime does not depend on LlamaIndex

**Status** accepted · **Date** 2026-10-25 · **Amends** [0001](0001-framework.md)

## Context

ADR 0001 chose LlamaIndex, with a boundary: LlamaIndex would own data
structures, parsers, vector-store adapters and post-processors, while the LLM
router, the citation contract, the verifier and the API stayed ours.

In implementation, that boundary moved. Every component on the serving path is
ours: the chunker, both vector-store adapters, the retrieval orchestration and
the evaluation metrics. LlamaIndex is not imported anywhere the API or the
ingestion pipeline runs.

This was not planned. It happened because the citation contract turned out to
constrain almost everything:

- **Chunking.** A chunk must never span a page and must carry a normalised
  bounding box, because that is what a citation points at. The chunker also has
  to keep tables and figures atomic, repeat table headers across splits, and
  track a heading breadcrumb. Expressing that through a generic node parser was
  more work than writing it, and the result would still have needed validating
  against the same contract.
- **Vector store.** Qdrant's named dense and sparse vectors on one point, with
  server-side reciprocal rank fusion and a tenant payload index, is specific
  enough that the adapter was going to be ours either way. The same interface
  then had to express Pinecone's namespaces and flat metadata.
- **Verification.** The verifier needs the exact text that was indexed, byte for
  byte, and the same normalisation used for hashing. Anything that transforms
  text between indexing and verification breaks the guarantee.

## Decision

Keep the implementation as it is, and be explicit that LlamaIndex is not a
runtime dependency.

`src/emrag/llm/llamaindex_adapter.py` is provided and supported: it exposes the
quota-aware router as a LlamaIndex `CustomLLM`, and the configured embedder as a
`BaseEmbedding`. Anyone who wants to use LlamaIndex components alongside this
system routes through those, so the quota ledger stays authoritative. The module
imports LlamaIndex lazily and nothing in the serving path imports it.

## Alternatives

**Retrofit LlamaIndex now.** Rejected. It would add a heavy dependency and
rewrite working, tested components, for no behaviour the system lacks. The
honest cost is a dependency we would then have to pin against, and the honest
benefit is familiarity rather than function.

**Drop the adapter and the dependency entirely.** Tempting for simplicity, but
it strands anyone who wants LlamaIndex's evaluators or query engines, and the
likely result is that they call a provider directly and bypass the ledger.

## Consequences

- The dependency footprint is much smaller: no LlamaIndex, no PyTorch in the API
  image. The image carries three ONNX models and nothing else heavy.
- Every component on the citation path is testable in isolation and has tests
  that assert the contract directly.
- Work that LlamaIndex would have provided is now ours to maintain: the Docling
  element mapping, both store adapters, the fusion and reranking glue.
- Contributors expecting a LlamaIndex codebase will not find one. The README and
  the blueprint's decision table are the places this must be stated plainly, and
  are.
- If a future requirement genuinely needs a LlamaIndex component, the adapter is
  the seam to add it through.
