# 0006 One collection, tenant in the payload

**Status** accepted · **Date** 2026-10-25

## Context

Several tenants share one free Qdrant cluster with 1 GB of RAM. Isolation has to
be airtight, and it has to fit.

## Decision

A single collection, `chunks_v1`, with `tenant_id` in the payload and a keyword
index marked `is_tenant=True` so each tenant's points are co-located on disk.

Enforcement is at one point in the code. `build_retrieval_filter` takes a
`Principal` and is the only way to construct a `RetrievalFilter`; it always emits
a tenant clause, an ACL-group clause and a classification ceiling. The tenant
comes from the credential, never from the request body. A unit test asserts the
tenant clause is present in the compiled filter, so removing it fails the build.

Pinecone maps the same model onto namespaces, which gives isolation for free, and
still asserts `tenant_id` in metadata as defence in depth.

## Alternatives

**A collection per tenant.** Cleanest isolation, and the obvious choice with
memory to spare. Each Qdrant collection carries fixed overhead, so a handful of
tenants would exhaust 1 GB. This is the right move past roughly 50 tenants or
when one tenant dominates, and the migration is a config change plus a replay.

**Separate clusters per tenant.** One free cluster per account; unmanageable.

## Consequences

- Fits comfortably: 30,000 chunks is about 160 MB of RAM.
- Filter correctness becomes load-bearing, so it has dedicated tests including
  an end-to-end check that a second tenant retrieves nothing.
- A tenant cannot be deleted by dropping a collection; deletion is a filtered
  delete plus artifact removal.
- Noisy-neighbour effects are possible, since all tenants share one index graph.
