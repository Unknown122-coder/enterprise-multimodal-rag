# %% [markdown]
# # Offline ingestion worker (Google Colab)
#
# This notebook is deliberately thin. Every pipeline decision lives in the
# `emrag` package, so the version that ran an ingestion is a git tag recorded on
# every chunk as `ingest_version`, not a copy-pasted cell.
#
# **What it does**
#
# 1. installs the repository at a pinned revision;
# 2. authenticates to the artifact store (a private Hugging Face dataset repo)
#    and to the vector store;
# 3. claims queued ingestion jobs from `inbox/`, one document at a time;
# 4. parses, captions figures, chunks, embeds, writes artifacts and upserts
#    vectors;
# 5. checkpoints after every document, so a disconnect costs one document.
#
# **Runtime**: a T4 GPU makes OCR and local vision roughly three times faster.
# CPU works and is the fallback. Either way, set the runtime before running.
#
# > Free Colab disconnects after about 90 minutes idle and caps a session at
# > 12 hours. Because checkpointing is per document and identifiers are derived
# > from content, re-running after a disconnect resumes rather than duplicates.

# %% [markdown]
# ## 1. Install
#
# Pin `REVISION` to a tag for reproducible ingestion. Installing takes two to
# four minutes; Docling pulls PyTorch.

# %%
REPO_URL = "https://github.com/your-org/enterprise-multimodal-rag.git"  # noqa: E501
REVISION = "v1.0.0"  # tag, branch or commit

# %%
# ruff: noqa: E402, F821
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    subprocess.run(["git", "clone", "--quiet", REPO_URL, "/content/emrag"], check=True)
    subprocess.run(["git", "-C", "/content/emrag", "checkout", "--quiet", REVISION], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", "/content/emrag[ingest]"],
        check=True,
    )
    os.chdir("/content/emrag")
    print("installed", REVISION)
else:
    print("running outside Colab; assuming the package is already installed")

# %% [markdown]
# ## 2. Secrets
#
# Store these in the Colab secrets panel (the key icon in the left sidebar), not
# in the notebook. None of them are ever written to the artifact store.
#
# | secret | purpose |
# |--------|---------|
# | `HF_TOKEN` | read and write the private dataset repo |
# | `HF_DATASET_REPO` | e.g. `your-org/emrag-store` |
# | `QDRANT_URL` | free cluster endpoint |
# | `QDRANT_API_KEY` | free cluster key |
# | `GEMINI_API_KEY` | figure captioning (vision tier) |
# | `GROQ_API_KEY` | optional vision fallback |

# %%
def load_secrets() -> None:
    """Copy Colab secrets into the environment the settings object reads."""
    names = [
        "HF_TOKEN",
        "HF_DATASET_REPO",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    ]
    if IN_COLAB:
        from google.colab import userdata

        for name in names:
            try:
                value = userdata.get(name)
            except Exception:
                value = None
            if value:
                os.environ[name] = value
    missing = [n for n in ("HF_TOKEN", "HF_DATASET_REPO", "QDRANT_URL") if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"missing required secrets: {missing}")


# The ingest environment stores artifacts in Hugging Face and uses the heavy parser.
os.environ["ARTIFACT_BACKEND"] = "hf"
os.environ["VECTOR_BACKEND"] = "qdrant"
os.environ["USE_DOCLING"] = "true"
os.environ["EMBED_BACKEND"] = "fastembed"
os.environ["APP_ENV"] = "prod"
os.environ["ALLOW_DEV_ANON"] = "false"
os.environ.setdefault("TENANTS_JSON", '[{"tenant_id": "demo", "keys": []}]')

load_secrets()
print("secrets loaded")

# %% [markdown]
# ## 3. Build the pipeline
#
# `build_state` wires exactly the same objects the API uses, so a document
# ingested here behaves identically to one ingested by the lite path.

# %%
import asyncio

from emrag import INGEST_VERSION
from emrag.api.deps import build_state
from emrag.observability.logging import configure_logging
from emrag.settings import get_settings

configure_logging("INFO", json_output=False)
settings = get_settings()
state = build_state(settings)

await state.store.ensure_schema(settings.embed_dim, sparse=True)  # type: ignore[top-level-await]
print("ingest_version", INGEST_VERSION)
print("embed model  ", state.embedder.model_name)
print("vision tier  ", state.router.models_by_tier().get("vision"))

# %% [markdown]
# ## 4. Warm the models
#
# Downloads the ONNX embedding and sparse models once, so per-document timings
# below reflect real throughput rather than a cold start.

# %%
state.embedder.warm_up()
print("models warm")

# %% [markdown]
# ## 5. Inspect the queue
#
# Jobs are created by `POST /v1/documents` on the API, which puts the bytes in
# `inbox/{tenant}/{doc_id}.pdf` and a queued job record alongside.

# %%
jobs = await state.registry.list_queued_jobs()  # type: ignore[top-level-await]
print(f"{len(jobs)} queued job(s)")
for job in jobs[:20]:
    print(f"  {job.job_id}  {job.tenant_id}/{job.doc_id}  pages={job.progress.pages_total}")

# %% [markdown]
# ## 6. Drain the queue
#
# One document per iteration. After each one the artifacts are uploaded as a
# single commit and the vectors are upserted, so the work is durable before the
# next document starts.
#
# Set `LIMIT` to a small number on the first run to check output quality before
# committing a long session.

# %%
LIMIT = 0  # 0 = process everything

from emrag.ingestion.cli import drain_jobs

processed = await drain_jobs(state, None, LIMIT)  # type: ignore[top-level-await]
print(f"processed {processed} document(s)")

# %% [markdown]
# ## 7. Ingest local files instead
#
# Useful for a first run: upload a few PDFs to the Colab file browser, then
# ingest them directly without going through the API.

# %%
from pathlib import Path

from emrag.ingestion.pipeline import IngestRequest

LOCAL_DIR = Path("/content/samples")
TENANT = "demo"

if LOCAL_DIR.exists():
    for path in sorted(LOCAL_DIR.glob("*.pdf")):
        request = IngestRequest(
            tenant_id=TENANT,
            filename=path.name,
            data=path.read_bytes(),
            tags=["colab"],
            classification="internal",
            acl_groups=["all"],
            profile="full",
        )
        result = await state.pipeline.run(request)  # type: ignore[top-level-await]
        flag = " (degraded)" if result.degraded else ""
        print(
            f"{path.name}: {result.chunks} chunks, {result.upserted} vectors, "
            f"{result.duration_s:.1f}s{flag}"
        )
        if result.degraded:
            for reason in result.record.quality_report.degraded_reasons[:5]:
                print(f"    - {reason}")
else:
    print(f"create {LOCAL_DIR} and upload PDFs to use this cell")

# %% [markdown]
# ## 8. Verify
#
# Three checks worth doing before trusting the corpus: the registry agrees with
# the vector store, a sample chunk looks sane, and a real query retrieves the
# page you expect.

# %%
await state.registry.refresh()  # type: ignore[top-level-await]
stats = await state.registry.stats()  # type: ignore[top-level-await]
vectors = await state.store.count()  # type: ignore[top-level-await]
print("documents:", stats["documents"], stats["by_status"])
print("vectors:  ", vectors)

# %%
from emrag.core.types import Principal
from emrag.retrieval import filters as filter_builder
from emrag.retrieval.retriever import RetrievalOptions

QUESTION = "what was total revenue"

principal = Principal(
    subject="colab",
    tenant_id=TENANT,
    groups=["all"],
    clearance="confidential",
    role="admin",
)
outcome = await state.retriever.retrieve(  # type: ignore[top-level-await]
    QUESTION,
    filter_builder.from_request(principal),
    RetrievalOptions(top_k=5, rewrite="off", parent_expansion=False),
)
for candidate in outcome.candidates[:5]:
    chunk = candidate.chunk
    print(
        f"[{candidate.effective_score:.3f}] {chunk.filename} p{chunk.page} "
        f"({chunk.modality}) {chunk.text[:140]!r}"
    )

# %% [markdown]
# ## 9. Notify the API
#
# The API refreshes its registry cache on a webhook from the dataset repo. If the
# webhook is not configured, call the endpoint once by hand after a session.
#
# ```bash
# curl -X POST https://your-space.hf.space/v1/internal/hf-webhook \
#   -H "X-Webhook-Secret: $HF_WEBHOOK_SECRET"
# ```

# %% [markdown]
# ## 10. Shut down
#
# Flushes the quota ledger and closes the vector store client.

# %%
await state.aclose()  # type: ignore[top-level-await]
print("done")

# %% [markdown]
# ## Troubleshooting
#
# | symptom | cause | fix |
# |---------|-------|-----|
# | `missing required secrets` | Colab secrets not granted to this notebook | open the key icon, toggle notebook access |
# | Docling import error | `[ingest]` extra not installed | re-run the install cell |
# | ingestion is very slow | CPU runtime | Runtime, Change runtime type, T4 GPU |
# | `quota exhausted` on captioning | Gemini daily cap reached | re-run tomorrow, or set `GROQ_API_KEY` for the fallback tier |
# | job stuck in `processing` | a previous session died mid-document | re-run the drain cell; the claim is replaced and the upsert is idempotent |
# | vector count below chunk count | upsert interrupted | `python scripts/upsert_artifacts.py --tenant all --verify` |
