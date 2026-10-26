# Runbook: rebuild the vector store

**When** the cluster was suspended or deleted, counts disagree with the registry,
or the collection name changed for a schema migration.

**Time** under ten minutes for 30,000 chunks. No parsing, no model calls, no cost.

This works because artifacts hold the vectors (see ADR 0004).

## Steps

1. Confirm the artifact store is intact. This is the thing that must survive:

   ```bash
   python -c "
   import asyncio
   from emrag.ingestion.artifacts import ArtifactReader
   from emrag.storage.artifact_store import build_artifact_store
   reader = ArtifactReader(build_artifact_store())
   print(asyncio.run(reader.list_documents()))
   "
   ```

   If that list is empty, stop. There is nothing to replay and the corpus must be
   re-ingested from source documents.

2. Create the collection and its payload indexes:

   ```bash
   python scripts/bootstrap_store.py
   ```

3. Replay, verifying as you go:

   ```bash
   python scripts/upsert_artifacts.py --tenant all --verify --refresh-manifest
   ```

   It prints per-document counts and exits non-zero if any document's point count
   does not match its chunk count.

4. Refresh the API registry and confirm readiness:

   ```bash
   curl -X POST https://<host>/v1/internal/hf-webhook -H "X-Webhook-Secret: $HF_WEBHOOK_SECRET"
   curl -sS https://<host>/readyz
   ```

5. Spot-check retrieval with a question you know the answer to:

   ```bash
   curl -sS -X POST https://<host>/v1/retrieve \
     -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"question":"<something you know is in the corpus>","top_k":5}'
   ```

## Upgrading the local Qdrant image

Qdrant's on-disk format is not readable across large version jumps: a 1.20
server refused a volume written by 1.12 with "unknown variant `on_disk`" and
exited. Do not try to upgrade the local container in place. Wipe its volume and
replay, which is the procedure above and takes under a minute for a small
corpus:

```bash
docker compose rm -sf qdrant
docker volume rm "$(docker volume ls --format '{{.Name}}' | grep qdrant_data)"
docker compose up -d qdrant
python scripts/bootstrap_store.py
python scripts/upsert_artifacts.py --tenant all --verify --refresh-manifest
```

Keep the `qdrant-client` pin and the compose image within one minor version of
each other; the client warns, loudly, when they drift. Qdrant Cloud manages its
own upgrades and is unaffected.

## Schema migration variant

To change the collection schema without downtime:

1. Set `COLLECTION_NAME=chunks_v2` in your shell only, and run steps 2 and 3.
2. Verify recall against the new collection with the evaluation harness.
3. Flip `COLLECTION_NAME` in the deployment environment and restart.
4. Delete the old collection once you are satisfied.

## Preventing it

`.github/workflows/keepalive.yml` issues a trivial count every six hours, which
is what stops a free cluster idling into suspension. If it has been failing, fix
that first.
