# 0001 Use LlamaIndex, not LangChain

**Status** accepted · **Date** 2026-10-25

## Context

The system is document-and-citation-centric. The hard part is not chaining calls;
it is carrying page numbers, bounding boxes, section paths and parent-child
relationships intact from a PDF through chunking and retrieval into a rendered
citation. Whatever framework we pick has to not lose that metadata.

## Decision

LlamaIndex core, plus its Qdrant, Pinecone and Docling adapters and its
evaluation module.

The boundary is explicit. LlamaIndex owns data structures, parsers, vector-store
adapters and post-processors. **We own** the LLM provider router, the citation
contract, the verifier and the API. Quota handling and output grammar are the two
things most likely to need exact control on a free tier, so they stay ours.

## Alternatives

**LangChain plus LangGraph.** Comparable capability, but its document model is
thinner, orchestration needs a second package, and its abstraction churn has been
higher. Neither helps with the metadata problem.

**No framework.** Genuinely tempting: we would write the parsers and adapters
ourselves and own everything. Rejected because the Docling and Qdrant hybrid
adapters are non-trivial and maintained, and re-implementing them buys nothing.

## Consequences

- The node metadata and relationship model maps almost directly onto the chunk
  contract, which is the point.
- Maintained adapters for Docling and Qdrant hybrid search, including
  server-side fusion.
- A heavy dependency that must be pinned; minor releases have broken imports
  before.
- Contributors need to know which side of the boundary they are on. The rule is:
  if it touches a provider quota or a citation, it is ours.
