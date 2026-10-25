# Runbook: ingest a batch of documents

**Time** 10 minutes of attention, then one to five hours unattended per 5,000 pages.

## Before you start

- Confirm the vector store is reachable: `python scripts/bootstrap_store.py`.
- Check the vision budget: `GET /v1/usage` as an admin. Captioning uses about one
  request per four figures. If `rpd_used` is already near its limit, either wait
  or accept the local vision fallback on Colab.

## Steps

1. Users upload through `POST /v1/documents`. Anything over `MAX_PAGES_LITE`
   pages, or scanned, is queued rather than ingested inline.
2. Open `notebooks/01_ingest_colab.ipynb` in Colab. **Runtime, Change runtime
   type, T4 GPU.**
3. Pin `REVISION` to the release tag you intend to ingest with. This value is
   recorded on every chunk as `ingest_version`.
4. Run the install, secrets, build and warm-up cells.
5. Run the queue cell. Confirm the document count matches what you expect.
6. Set `LIMIT = 2`, run the drain cell, and **read the output**. Check chunk
   counts per document and any degraded reasons before committing to the batch.
7. Set `LIMIT = 0` and run the drain cell again.
8. Run the verification cells.

## Verifying

Three things must agree, and all three are in the notebook:

- registry document count against vector count;
- per-document chunk count against what the store holds;
- a real query returning the page you expect.

Then refresh the API's registry:

```bash
curl -X POST https://<host>/v1/internal/hf-webhook -H "X-Webhook-Secret: $HF_WEBHOOK_SECRET"
```

## If something goes wrong

| symptom | do this |
|---------|---------|
| Colab disconnected mid-batch | re-run the drain cell; completed documents are skipped and the interrupted one restarts |
| job stuck in `processing` | re-run the drain cell; the claim is replaced and the upsert is idempotent |
| `status: ready_degraded` | read `quality_report.degraded_reasons` on the document; usually a parser fallback or an un-OCR'd page |
| vision quota exhausted | figures get a minimal description instead of a caption; re-ingest those documents tomorrow for full captions |
| vector count below chunk count | `python scripts/upsert_artifacts.py --tenant all --verify` |
