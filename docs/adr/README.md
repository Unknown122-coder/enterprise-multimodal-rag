# Architecture decision records

One file per decision that would be expensive to reverse. Each states the
decision, what else was considered, why this won, and what it costs. The full
decision table is in `ARCHITECTURE_BLUEPRINT.md` section 13; these expand the
ones where the trade-off is not obvious.

| ADR | Decision |
|-----|----------|
| [0001](0001-framework.md) | LlamaIndex over LangChain |
| [0002](0002-embeddings.md) | Local fastembed over a hosted embedding API |
| [0003](0003-hosting.md) | Hugging Face Spaces over Render |
| [0004](0004-artifacts-as-truth.md) | Artifacts, not the vector store, are the source of truth |
| [0005](0005-citation-verification.md) | Deterministic quote verification over model self-reporting |
| [0006](0006-multi-tenancy.md) | One collection with a tenant payload index |
| [0007](0007-framework-deviation.md) | The runtime does not depend on LlamaIndex (amends 0001) |
