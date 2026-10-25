# Enterprise Multimodal RAG System

Question answering over corporate documents with citations that are **verified
against the source text before they are returned**.

Every claim in an answer names a document, a page, a chunk and a quote that
provably appears in that chunk. A claim whose citation cannot be verified is
dropped rather than shown. When the corpus does not support an answer, the
response status is `not_found`, never prose.

Runs entirely on free tiers: Qdrant Cloud, Groq, Gemini, Google Colab, Hugging
Face Spaces and a private Hugging Face dataset repository.

- **Design**: [`ARCHITECTURE_BLUEPRINT.md`](ARCHITECTURE_BLUEPRINT.md)
- **Decisions**: [`docs/adr/`](docs/adr/)
- **Operations**: [`docs/runbooks/`](docs/runbooks/)

> **On frameworks.** The blueprint selected LlamaIndex. The implementation does
> not depend on it: the chunker, both vector-store adapters, retrieval and the
> evaluation metrics are all in this repository, because the citation contract
> constrained them too tightly to express through a generic node parser.
> `emrag.llm.llamaindex_adapter` exposes the quota-aware router and the
> configured embedder to LlamaIndex for anyone who wants its components, and
> nothing on the serving path imports it. The reasoning is in
> [ADR 0007](docs/adr/0007-framework-deviation.md).

---

## Contents

1. [How it works](#how-it-works)
2. [Quick start](#quick-start)
3. [Free-tier account setup](#free-tier-account-setup)
4. [Colab ingestion walkthrough](#colab-ingestion-walkthrough)
5. [API examples](#api-examples)
6. [The citation guarantee](#the-citation-guarantee)
7. [Configuration](#configuration)
8. [Deployment](#deployment)
9. [Evaluation](#evaluation)
10. [Development](#development)
11. [Known limits](#known-limits)

---

## How it works

Two paths, one contract.

**Ingestion** runs offline in Colab, where a GPU and the heavy parser are free.
Docling recovers layout, reading order, table structure and OCR; figures are
described by a vision model; a structure-aware chunker produces page-bounded
chunks with bounding boxes; everything is written to a private dataset repository
as the durable record, and the vectors are upserted to Qdrant.

**Query** runs in a FastAPI service. A question is embedded locally, retrieved
with hybrid dense plus BM25 search fused server-side, reranked by a cross
encoder, then gated: if the best evidence is too weak, the service answers
`not_found` **without calling a model at all**. Otherwise the evidence is handed
to an LLM under a strict JSON grammar, and every citation it returns is checked
against the chunk text before the response is rendered.

```
upload ──> inbox ──> [Colab] parse ─ caption ─ chunk ─ embed ──> artifacts + Qdrant
                                                                      │
question ──> embed ──> hybrid search ──> rerank ──> gate ──> LLM ──> verify ──> answer
```

Two properties follow from this shape, and both are load-bearing:

- **Artifacts are the source of truth.** The vector store can be rebuilt from
  them in minutes, with identical identifiers, which is what makes a suspended
  free cluster an inconvenience rather than data loss.
- **Identifiers are content-derived.** Re-ingesting the same bytes produces the
  same chunk ids, so upserts are idempotent and a disconnected Colab session
  resumes instead of duplicating.

---

## Quick start

Requires Python 3.11 (3.12 also works) and Docker for the local Qdrant.

```bash
git clone <this repo> && cd enterprise-multimodal-rag
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Start a local Qdrant and the API:

```bash
docker compose up -d qdrant
python scripts/bootstrap_store.py          # create the collection and its indexes
uvicorn emrag.api.main:app --reload --port 8000
```

In another shell, ingest a document and ask a question:

```bash
# Upload (short native PDFs are ingested inline)
curl -sS -X POST http://localhost:8000/v1/documents \
  -F "file=@report.pdf" \
  -F 'metadata={"title":"Q3 Report","tags":["2025"]}'

# Ask
curl -sS -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How much did segment revenue grow?"}' | jq
```

`ALLOW_DEV_ANON=true` in `.env` lets you call the API without a credential as
tenant `demo`. It is rejected when `APP_ENV=prod`.

Optional web interface:

```bash
pip install -e ".[ui]"
EMRAG_API_URL=http://localhost:8000 streamlit run ui/app.py
```

Everything in one container pair, instead of the above:

```bash
docker compose up --build        # API on http://localhost:8000
```

---

## Free-tier account setup

Five accounts, none of which need a card. Figures change; verify the current
limits in each console and record them in `configs/providers.yaml`, which is the
only place the system reads them from.

### 1. Qdrant Cloud (vector store)

1. Sign up at <https://cloud.qdrant.io> and create a **free cluster** (1 GB RAM,
   4 GB disk, one node).
2. Copy the cluster URL and create an API key.
3. Put them in `.env` as `QDRANT_URL` and `QDRANT_API_KEY`.
4. Run `python scripts/bootstrap_store.py`.

A free cluster can be suspended after prolonged inactivity. The keepalive
workflow prevents that, and `scripts/upsert_artifacts.py` rebuilds it if it
happens anyway.

### 2. Groq (text generation)

1. Sign up at <https://console.groq.com> and create an API key.
2. Set `GROQ_API_KEY`.

Groq provides the fast answer tier. The largest model has the smallest daily
token budget, which is why it is the `quality` tier rather than the default.

### 3. Google AI Studio (vision and the volume tier)

1. Create a key at <https://aistudio.google.com/apikey>.
2. Set `GEMINI_API_KEY`.

Gemini describes figures during ingestion, serves as the high-volume answer tier,
and acts as the evaluation judge.

### 4. Hugging Face (artifact store and hosting)

1. Create an account at <https://huggingface.co>.
2. Create a **private dataset** repository, for example `your-org/emrag-store`.
3. Create a write token at <https://huggingface.co/settings/tokens>.
4. Set `HF_TOKEN`, `HF_DATASET_REPO` and `ARTIFACT_BACKEND=hf`.
5. For hosting, create a **Docker Space** and add the same secrets there.

### 5. Google Colab (ingestion)

No setup beyond a Google account. Add the secrets above to the Colab secrets
panel; the notebook reads them from there and never writes them anywhere.

### Issue an API key

```bash
python scripts/make_api_key.py --tenant acme --role writer --group finance
```

It prints the key once and the Argon2 hash to store. Put the tenant record in
`configs/tenants.yaml` locally, or serialise it into `TENANTS_JSON` in
production. Only the hash is ever stored.

---

## Colab ingestion walkthrough

The notebook is deliberately thin: all logic lives in the package, so the version
that ran an ingestion is a git tag recorded on every chunk.

1. Open `notebooks/01_ingest_colab.ipynb` in Colab.
2. **Runtime, Change runtime type, T4 GPU.** CPU works but is three to five
   times slower.
3. Add the secrets listed in the notebook to the Colab secrets panel (the key
   icon), and grant the notebook access to each.
4. Set `REPO_URL` and `REVISION` in the first cell. Pin `REVISION` to a tag.
5. Run the install cell. Two to four minutes; Docling pulls PyTorch.
6. Run through to the queue cell to see what is waiting. Jobs appear there when
   someone calls `POST /v1/documents` with a document too large or too scanned
   for the inline path.
7. Set `LIMIT = 2` and run the drain cell first. Inspect the output before
   committing to a long session.
8. Set `LIMIT = 0` and run it again to process everything.
9. Run the verification cells: registry counts against vector counts, then a real
   query to confirm the right page comes back.

Throughput on a T4 is roughly 0.5 to 2 pages per second for native PDFs and 0.25
to 0.5 with OCR, so 5,000 pages fits inside one session. Checkpointing is per
document, so a disconnect costs one document.

To ingest files without the API, upload them to `/content/samples` in Colab and
run the local-files cell.

---

## API examples

Base path `/v1`. Interactive docs at `/docs`; the schema is also checked into
`docs/api/openapi.json`.

### Upload

```bash
curl -sS -X POST http://localhost:8000/v1/documents \
  -H "Authorization: Bearer $EMRAG_KEY" \
  -F "file=@annual-report.pdf" \
  -F 'metadata={"title":"Annual Report 2025","tags":["finance","2025"],"classification":"internal","acl_groups":["finance"]}'
```

```json
{
  "document_id": "d_3fa9c2e7b1d04a55",
  "job_id": "job_8c1f22a0d3e74b9a5f61",
  "pipeline": "lite",
  "status": "queued",
  "detail": "12 native pages ingested inline"
}
```

### Watch ingestion

```bash
curl -sS http://localhost:8000/v1/jobs/job_8c1f22a0d3e74b9a5f61 -H "Authorization: Bearer $EMRAG_KEY"
curl -sS http://localhost:8000/v1/documents/d_3fa9c2e7b1d04a55 -H "Authorization: Bearer $EMRAG_KEY"
```

### Ask

```bash
curl -sS -X POST http://localhost:8000/v1/query \
  -H "Authorization: Bearer $EMRAG_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How much did segment revenue grow in Q3 2025?",
    "filters": {"tags": ["2025"], "modalities": ["text", "table"]},
    "options": {"model_tier": "fast", "top_k": 8, "deadline_ms": 4500}
  }'
```

```json
{
  "request_id": "req_01j8x4m2p9",
  "status": "answered",
  "answer": {
    "markdown": "Segment revenue grew 14 percent year over year. [1]",
    "claims": [{"index": 0, "text": "Segment revenue grew 14 percent year over year.", "citation_refs": [1]}]
  },
  "citations": [
    {
      "ref": 1,
      "chunk_id": "d_3fa9c2e7b1d04a55:p012:c0007",
      "document_id": "d_3fa9c2e7b1d04a55",
      "filename": "annual-report.pdf",
      "page": 12,
      "bbox": [0.071, 0.312, 0.929, 0.588],
      "modality": "text",
      "section_path": ["3 Financial Review", "3.2 Segment Results"],
      "quote": "Segment revenue grew 14 percent year over year",
      "verified": true,
      "match_score": 1.0,
      "page_image_url": "/v1/documents/d_3fa9c2e7b1d04a55/pages/12?bbox=0.0710,0.3120,0.9290,0.5880"
    }
  ],
  "retrieval": {"strategy": "hybrid_rrf+rerank", "candidates": 40, "used": 6, "latency_ms": 712},
  "generation": {"provider": "groq", "model": "llama-3.3-70b-versatile", "prompt_tokens": 3120, "completion_tokens": 212},
  "latency_ms": 2460
}
```

Branch on `status`, not on whether `answer` is empty:

| status | meaning |
|--------|---------|
| `answered` | every claim verified |
| `partial` | some claims were dropped as unverifiable; what remains is verified |
| `not_found` | the corpus does not support an answer; `nearest_sources` lists where to look |
| `refused` | the request itself was rejected, with a `reason` code |

### See the source

```bash
# Page image with the cited region highlighted
curl -sS "http://localhost:8000/v1/documents/d_3fa9c2e7b1d04a55/pages/12?bbox=0.071,0.312,0.929,0.588" \
  -H "Authorization: Bearer $EMRAG_KEY" -o page12.webp

# The chunk a citation points at
curl -sS http://localhost:8000/v1/chunks/d_3fa9c2e7b1d04a55:p012:c0007 -H "Authorization: Bearer $EMRAG_KEY"
```

### Stream

```bash
curl -sS -N -X POST http://localhost:8000/v1/query/stream \
  -H "Authorization: Bearer $EMRAG_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Summarise the outlook section"}'
```

Events arrive in order: `retrieval`, many `answer_delta`, then `citations`,
`verification`, optionally `revision`, then `usage` and `done`. **If a
`revision` event arrives, replace the text you displayed with its payload**: it
means verification removed claims after streaming them.

### Retrieval only

Useful for debugging and for building a source browser without spending quota.

```bash
curl -sS -X POST http://localhost:8000/v1/retrieve \
  -H "Authorization: Bearer $EMRAG_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"question":"free cash flow","top_k":5}'
```

### Quota and health

```bash
curl -sS http://localhost:8000/v1/usage -H "Authorization: Bearer $ADMIN_KEY"   # per-model budgets
curl -sS http://localhost:8000/v1/metrics -H "Authorization: Bearer $ADMIN_KEY" # counters, latency
curl -sS http://localhost:8000/healthz
curl -sS http://localhost:8000/readyz
```

### Errors

Every error is an RFC 9457 problem document carrying the request id:

```json
{
  "type": "https://emrag.dev/errors/quota_exhausted",
  "title": "Service Unavailable",
  "status": 503,
  "detail": "all models for tier 'fast' are unavailable (groq/llama-4-scout:tpd)",
  "code": "QUOTA_EXHAUSTED",
  "request_id": "req_01j8x4m2p9",
  "retry_after_s": 1830
}
```

Codes: `VALIDATION_ERROR`, `UNAUTHENTICATED`, `FORBIDDEN`, `NOT_FOUND`,
`PAYLOAD_TOO_LARGE`, `UNSUPPORTED_MEDIA`, `RATE_LIMITED`, `QUOTA_EXHAUSTED`,
`MODEL_UNAVAILABLE`, `STORE_UNAVAILABLE`, `INGEST_FAILED`, `DEADLINE_EXCEEDED`.

`QUOTA_EXHAUSTED` means a configured model is out of budget and carries a
`retry_after_s`. `MODEL_UNAVAILABLE` means no model is configured for the
requested tier, so waiting will not help: set a provider key.

---

## The citation guarantee

Nothing reaches a user unless a quote supporting it provably exists in the cited
chunk. Four mechanisms, in order of how much they contribute:

1. **The retrieval gate.** When the best reranked score is below the configured
   threshold, the service returns `not_found` and never calls a model. This is
   the cheapest and most effective guard, because the common path to an
   ungrounded answer is a model being handed weak evidence and trying anyway.
2. **A constrained grammar.** The model must emit claims, each with an evidence
   alias and a verbatim quote. Aliases are short labels (`c1`, `c2`) for the
   passages actually retrieved, so there is no plausible-looking identifier to
   invent.
3. **Mechanical verification.** Each citation is checked by
   `emrag.citation.verifier`: the alias must be one shown for this request, and
   the quote must appear in that chunk's text after normalisation. A near match
   above threshold is accepted and **the quote is replaced with the real span
   from the source**, so the user always sees text that exists in the document.
4. **A drop policy.** A claim with no verified citation is removed. If every
   claim is removed, the status becomes `not_found`.

Normalisation is shared with content hashing, in one function, so a quote can
never verify against text that was not indexed. That invariant is a test.

Against prompt injection: retrieved content is wrapped in delimited blocks and
declared as data; the model has no tools; output is constrained to the grammar;
suspicious passages are flagged at ingestion and stripped of invisible
characters; and above all, an injected instruction that produces an unquotable
claim cannot survive verification. Questions that attack the system are refused
before retrieval.

---

## Configuration

Secrets and deployment choices come from the environment; tuning comes from YAML.

| file | holds |
|------|-------|
| `.env` | secrets and deployment choices, see `.env.example` |
| `configs/default.yaml` | chunk sizes, retrieval k values, thresholds, budgets |
| `configs/providers.yaml` | the model catalogue with published rate limits |
| `configs/tenants.yaml` | tenants, hashed API keys, groups, PII policy |

Settings are validated at startup and the process fails fast on a bad
combination. `APP_ENV=prod` additionally requires tenants to be configured,
anonymous access to be off, and at least one LLM key.

The knobs worth knowing:

| variable | default | effect |
|----------|---------|--------|
| `RETRIEVAL_GATE_TAU` | `0.15` | higher refuses more; calibrate with notebook 03 |
| `EVIDENCE_TOKEN_BUDGET` | `2500` | context given to the model |
| `TOP_K` / `CANDIDATES_K` | `8` / `40` | passages used and retrieved |
| `RERANK_ENABLED` | `true` | the cross encoder; off saves ~350 ms, costs precision |
| `MAX_PAGES_LITE` | `20` | above this, uploads queue for Colab |
| `VECTOR_BACKEND` | `qdrant` | or `pinecone` |
| `QDRANT_URL` | `http://localhost:6333` | a server URL, `local://<dir>` for on-disk, `:memory:` for tests |
| `EMBED_BACKEND` | `fastembed` | or `gemini` for small hosts, `deterministic` for tests |

---

## Deployment

### Hugging Face Spaces (recommended)

Sixteen gigabytes of RAM is what makes local embedding and reranking possible, so
the query path consumes no API quota. It is the reason this is the primary target.

Step by step, including the GitHub secrets and repository variables the
workflows need: [`docs/runbooks/deploy.md`](docs/runbooks/deploy.md). In short:

1. Create a **Docker** Space.
2. Add the secrets from [account setup](#free-tier-account-setup), plus
   `APP_ENV=prod`, `ALLOW_DEV_ANON=false`, `ARTIFACT_BACKEND=hf`.
3. Set the `HF_SPACE` repository variable (`org/space`) and the `HF_TOKEN`
   secret in GitHub.
4. Push a tag: `git tag v1.0.0 && git push --tags`. The deploy workflow pushes
   the tree, swaps in the Space README front matter, and waits for `/healthz`.
5. Register the artifact-store webhook, or documents ingested in Colab stay
   invisible to the API until it restarts.

Models are baked into the image, so a cold start never waits on a download.
`/readyz` stays red until warm-up finishes, which makes a cold start visible
rather than mysterious.

### Render (fallback)

`render.yaml` is a working blueprint, but read `docs/adr/0003-hosting.md` first:
512 MB cannot hold the local models, so that profile switches to Gemini
embeddings and disables reranking. The query path then consumes an API quota and
retrieval precision drops.

### Cold starts and sleep

| host | sleeps after | wake |
|------|--------------|------|
| Hugging Face Spaces | ~48 h idle | 30 to 60 s |
| Render free | ~15 min idle | 30 to 60 s |

`.github/workflows/keepalive.yml` pings the API and the vector store every six
hours. For Render, lower the cron to every ten minutes.

### CI/CD

| workflow | trigger | does |
|----------|---------|------|
| `ci.yml` | push, PR | ruff, mypy, the test suite, contract check, Docker build and probe |
| `deploy-hf-space.yml` | tag `v*` | push to the Space, wait for health |
| `keepalive.yml` | every 6 h | keep the host and cluster warm |
| `eval-nightly.yml` | nightly | full evaluation with judges, gated on thresholds |
| `security.yml` | push, weekly | secret scan, dependency audit |

Tests run fully offline: a deterministic embedder, in-process Qdrant and a
scripted LLM, so CI needs no API keys and costs no quota.

---

## Evaluation

```bash
# Free, fast, runs in CI
python -m emrag.eval.run_eval --mode retrieval --golden tests/golden/v1.jsonl --min-recall 0.85

# Spends quota; nightly
python -m emrag.eval.run_eval --mode full --min-faithfulness 0.85 --report var/eval.json
```

| metric | target | what it tells you |
|--------|--------|-------------------|
| `recall@8` | ≥ 0.90 | did retrieval find the right page |
| `mrr@8` | ≥ 0.70 | how high up |
| `quote_coverage` | ≥ 0.90 | was the exact span needed to cite actually retrieved |
| `citation_accuracy` | 1.00 | enforced by construction; anything less is a bug |
| `faithfulness` | ≥ 0.90 | judge: is each claim entailed by its quote |
| `not_found_correct` | ≥ 0.90 | refusal behaviour on the unanswerable subset |
| p95 latency | < 5 s | warm instance, end to end |

Start with retrieval. If recall is low, no prompt work will fix the answers and
generation spend is wasted. `notebooks/03_eval_offline.ipynb` prints the failing
questions and sweeps the gate threshold, which is the tuning that matters most.

`tests/golden/v1.jsonl` ships as a 20-item starter against the fixture
documents. Build the real 100-item set with `notebooks/02_build_golden_set.ipynb`,
which has a model draft questions and a human accept them, and verifies every
gold quote against the corpus before writing it. See
[`tests/golden/README.md`](tests/golden/README.md).

---

## Development

```bash
make install     # editable install with dev extras
make dev         # plus pre-commit hooks
make lint        # ruff check and format check
make type        # mypy
make test        # unit and integration, offline
make test-int    # integration only
make eval        # retrieval evaluation
make openapi     # export docs/api/openapi.json
make up          # docker compose: api + qdrant
```

### Layout

```
src/emrag/
├── core/          pure domain: types, ids, text normalisation, tokens, errors
├── ingestion/     triage, parsers, tables, figures, chunking, quality, artifacts
├── embeddings/    fastembed (default), gemini, deterministic (tests)
├── vectorstore/   one interface, Qdrant and Pinecone behind it, plus replay/migration
├── retrieval/     filters, hybrid search, rerank, rewrite, context budget
├── citation/      output contract, verifier, status policy, renderer
├── generation/    the answerer that ties it together
├── security/      auth, ACL, upload validation, injection, PII
├── storage/       artifact store, registry, per-document SQLite shards
├── observability/ structured logging, metrics, optional OTel tracing
├── eval/          metrics, judges, run_eval CLI
└── api/           routers, schemas, middleware, dependency wiring
```

### Non-negotiables

Changing any of these needs a very good reason, and each has a test guarding it:

1. The tenant comes from the credential, never the request. Every retrieval
   filter carries a tenant clause.
2. No answer text is returned without at least one verified citation.
3. A chunk never spans a page, and always has a page and a bounding box.
4. Hashing and quote verification share one normalisation function.
5. Provider rate limits live in `configs/providers.yaml`, never in code.
6. Ingestion logic lives in the package, not in notebooks.
7. The vector store is rebuildable from artifacts with identical identifiers.
8. Secrets come from the environment; logs redact secrets and, by default,
   query text.

### Testing

```bash
pytest -q                          # everything, offline
pytest -m "not integration" -q     # units only
pytest tests/unit/test_verifier.py # the citation guarantee
```

Integration tests run the real pipeline against an in-process Qdrant with a
scripted LLM, so retrieval, filters and verification are genuinely exercised.

---

## Known limits

**Operational**

- Heavy ingestion is operator-driven. Someone opens the notebook; there is no
  always-on worker on the free tier. The first paid step is a background worker
  polling the same inbox contract.
- Registry writes go through the artifact store, so document create and delete
  take one to two seconds. A real database is the first paid upgrade.
- Rate limiting is in-process, which is correct for one instance and wrong for
  several.

**Capacity**

- Combined free LLM quotas support roughly 1,000 to 1,500 answers per day. The
  largest model allows only about 25 of those, which is why it is not the
  default tier. `GET /v1/usage` shows what remains.
- The free vector cluster holds about 150,000 chunks with quantisation, well
  past the 5,000-page design target.
- Figure captioning is capped by the vision quota. A first-day bulk ingestion of
  thousands of figures will span several days, or fall back to the local vision
  model on Colab.

**Quality**

- English-centric. The embedder and reranker are English models; multilingual
  means switching to bge-m3 and re-embedding.
- Bounding boxes on OCR pages are advisory. `bbox_confidence` carries the OCR
  engine's confidence, and highlights on scanned pages can be approximate.
- Without Docling (the lite path, or when it fails) there is no table structure.
  Those documents are marked `ready_degraded` with the reason recorded.
- A legitimate paraphrase with no quotable span is dropped. The system prefers
  silence to an unverifiable sentence, which occasionally loses a correct answer.

**Security**

- Prompt-injection defence is layered, not absolute. The guarantee is narrower
  and stronger: an injected claim cannot be shown unless it is quotable from the
  corpus.
- PII detection is regex based. It catches structured identifiers, not names.
- Malware checking is a hash reputation lookup, not scanning.

**Free-tier figures in this document and in `configs/providers.yaml` are
indicative.** Verify them in each console before sizing anything.

---

## Licence

Apache-2.0.
