# 0004 Artifacts are the source of truth, not the vector store

**Status** accepted · **Date** 2026-10-25

## Context

A free Qdrant cluster can be suspended after inactivity, and a free Pinecone
index can be reaped. If the vectors were the only record of an ingestion run,
losing them would mean re-parsing and re-captioning the whole corpus: hours of
Colab time and a large bite out of the vision quota.

Separately, we want to switch vector backends without re-embedding, and to make
re-ingestion idempotent.

## Decision

Every ingestion writes a complete artifact folder per document to a private
Hugging Face dataset repository: source bytes, parsed elements, chunks **with
their vectors** in Parquet, parent sections, the SQLite query shard, page renders,
figure crops and the quality report.

Identifiers are content-derived: `doc_id` from the file bytes, `chunk_id` from
the document, page and sequence, and the vector-store key as a UUIDv5 of the
chunk id.

## Alternatives

**Vector store as truth.** Simpler, one place to look, and the backup is whatever
the provider gives you. On a free tier that is nothing.

**Postgres for everything.** The right answer with a budget, and the first paid
upgrade. Free Postgres offerings pause, which is the same problem one layer down.

## Consequences

- A lost cluster costs minutes: `scripts/upsert_artifacts.py` replays stored
  vectors, with no parsing and no model calls.
- Backend migration is a replay plus a recall comparison
  (`scripts/migrate_store.py`), not a re-embed.
- Re-ingesting identical bytes produces identical ids, so upserts are idempotent
  and an interrupted Colab session resumes rather than duplicates.
- Storage is duplicated: vectors live in Parquet and in the store.
- Registry writes are repository commits, so document create and delete take one
  to two seconds. Acceptable for a corpus that changes rarely.
