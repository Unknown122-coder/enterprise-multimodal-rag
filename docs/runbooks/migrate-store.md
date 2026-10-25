# Runbook: migrate between Qdrant and Pinecone

**Time** 15 to 30 minutes. No re-embedding and no model calls.

The two backends differ in ways the adapter hides: tenancy is a namespace rather
than a payload filter, dense and sparse live in separate indexes so fusion moves
client-side, and metadata must be flat. The risk is therefore not data loss but a
**silent recall regression** from a filter-syntax difference. The verification
step is the point of this runbook.

## Steps

1. Create the target account and credentials, and set them in your shell. Do not
   change the deployed environment yet.

   ```bash
   export PINECONE_API_KEY=...
   export PINECONE_INDEX_DENSE=emrag-dense-v1
   export PINECONE_INDEX_SPARSE=emrag-sparse-v1
   ```

2. Replay and compare, with both backends reachable:

   ```bash
   python scripts/migrate_store.py \
     --from qdrant --to pinecone \
     --tenant all \
     --verify-golden tests/golden/v1.jsonl \
     --max-recall-delta 0.02
   ```

   The script creates the target schema, replays artifacts, checks per-document
   counts, then measures recall on both backends and fails if the target is more
   than two points worse.

3. If it fails on recall rather than counts, the filter translation is the first
   place to look: `build_pinecone_filter` against `build_qdrant_filter`. Compare
   a single query's candidates on both backends with `POST /v1/retrieve`.

4. Flip the deployment: set `VECTOR_BACKEND=pinecone` plus the Pinecone
   variables, and restart.

5. Confirm, then leave the source backend in place for a few days before
   deleting it.

## Notes

- Pinecone Starter cannot delete by metadata filter, so document deletion lists
  ids first and deletes in batches. Deleting a large document is slower there.
- `export` is not implemented for Pinecone. Migration away from it replays
  artifacts, which is the supported path in either direction.
- Nothing above this layer changes. If application code needed editing, the
  abstraction has leaked and that is the bug to fix.
