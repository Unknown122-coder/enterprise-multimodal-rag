# 0002 Embed locally with fastembed, not a hosted API

**Status** accepted · **Date** 2026-10-25

## Context

Every query must be embedded. On a free tier the question is not which model is
most accurate but what happens at the end of the day when a quota is spent.

A hosted embedding API couples the **query path** to a daily request limit. When
that limit is reached, search stops working. Not degrades: stops.

## Decision

`BAAI/bge-base-en-v1.5` at 768 dimensions via fastembed (ONNX, CPU), with
`Qdrant/bm25` for the sparse companion. Identical artefacts on Colab and on the
API host, so ingestion and query vectors are byte-compatible.

## Alternatives

**Gemini `gemini-embedding-001`.** Good quality, and the Matryoshka output
truncates to 768 so the schema would be unchanged. Rejected as the default
because of the quota coupling above. Kept as a real, implemented profile
(`EMBED_BACKEND=gemini`) for hosts too small to run a local model, with the
trade-off documented in the module.

**`bge-m3`.** Better, multilingual, 1024 dimensions. Too heavy for two vCPUs
inside a 5-second p95 budget. This is the upgrade path when multilingual matters.

## Consequences

- The query path consumes no API quota. Search keeps working when generation
  quota is exhausted; the service answers `not_found` rather than failing.
- Roughly 450 MB of host RAM, which is why hosting needs more than 512 MB and
  therefore drives ADR 0003.
- English-centric retrieval.
- Models are baked into the container image, so a cold start never waits on a
  download.
- Thirty thousand chunks is about 92 MB of dense vectors, around six times
  headroom on the free cluster.
