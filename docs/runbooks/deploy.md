# Runbook: first deployment

**Time** about 40 minutes, most of it waiting for accounts and the first image
build.

Order matters: the artifact store and vector store must exist before the service
starts, or it comes up with an empty registry and a missing collection.

## 1. Provision

Follow the account setup in the README, then confirm each one works before
moving on:

```bash
python scripts/bootstrap_store.py      # creates the collection and payload indexes
```

That is the only step that fails loudly on a bad Qdrant URL or key, so do it
first.

## 2. Issue credentials

```bash
python scripts/make_api_key.py --tenant acme --role writer --group finance
python scripts/make_api_key.py --tenant acme --role reader
```

Collect the tenant records into a JSON array and keep it for step 4. Only the
Argon2 hashes are stored; the plaintext keys are shown once.

## 3. Verify the published limits

`configs/providers.yaml` ships indicative figures. Open each provider console,
confirm the current requests-per-minute, tokens-per-minute and daily limits, and
edit the file. Nothing else in the system reads a rate limit, so this file being
wrong is the single most likely cause of unexpected 429s.

## 4. Host secrets (Hugging Face Space)

Create a **Docker** Space, then add these under Settings, Variables and secrets.

Secrets:

| secret | value |
|--------|-------|
| `QDRANT_URL` | free cluster endpoint |
| `QDRANT_API_KEY` | free cluster key |
| `GROQ_API_KEY` | from console.groq.com |
| `GEMINI_API_KEY` | from aistudio.google.com/apikey |
| `HF_TOKEN` | write token for the dataset repo |
| `HF_WEBHOOK_SECRET` | any long random string you generate |
| `TENANTS_JSON` | the array from step 2 |
| `JWT_HS256_SECRET` | only if you issue JWTs |

Public variables:

| variable | value |
|----------|-------|
| `APP_ENV` | `prod` |
| `ALLOW_DEV_ANON` | `false` |
| `ARTIFACT_BACKEND` | `hf` |
| `HF_DATASET_REPO` | e.g. `your-org/emrag-store` |

`APP_ENV=prod` refuses to start without tenants configured, anonymous access
disabled and at least one LLM key, so a half-configured Space fails fast rather
than serving unauthenticated.

## 5. GitHub secrets and variables

The workflows need their own copies. Repository Settings, Secrets and variables,
Actions.

Secrets:

| secret | used by |
|--------|---------|
| `HF_TOKEN` | deploy, nightly evaluation |
| `QDRANT_URL`, `QDRANT_API_KEY` | keepalive |
| `QDRANT_URL_STAGING`, `QDRANT_API_KEY_STAGING` | nightly evaluation |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | nightly evaluation |
| `TENANTS_JSON_STAGING` | nightly evaluation |

Variables:

| variable | used by | example |
|----------|---------|---------|
| `HF_SPACE` | deploy | `your-org/emrag` |
| `PUBLIC_API_URL` | keepalive | `https://your-org-emrag.hf.space` |
| `HF_DATASET_REPO_STAGING` | nightly evaluation | `your-org/emrag-store-staging` |

`GITHUB_TOKEN` is provided automatically. The nightly evaluation is the only
workflow that spends provider quota; if you do not want that, disable it rather
than leaving its secrets unset, so the failure is deliberate.

## 6. Deploy

With a GitHub remote:

```bash
git tag v1.0.0 && git push --tags
```

Without one, or to watch each step from your machine, fill `.env.deploy`
(copy `.env.deploy.example`) and run:

```bash
python scripts/deploy_space.py --plan   # check settings, touch nothing
python scripts/deploy_space.py          # repos, artifact sync, cloud collection, secrets, push, wait
```

It does steps 1, 4 and 6 of this runbook in one go and is idempotent, so rerun
it after any failure. `--skip-store` redeploys code only.

The deploy workflow pushes the tree to the Space, swaps in the Space README
front matter, and polls `/healthz` for up to fifteen minutes. The first build is
slow because the image downloads and bakes three ONNX models; later builds reuse
the layer.

## 6b. Deploy to Google Cloud Run instead

Since 2026 Hugging Face requires a PRO subscription for Docker Spaces on the
free CPU tier (`create_repo` returns HTTP 402). Cloud Run is the free-tier
alternative: 2 million requests and 360,000 GB-seconds a month free on any
account, scale to zero between requests. The container needs about 1.2 GB under
load, so it runs with 2 GiB, one vCPU, and a cap of one instance.

Needs a Google Cloud project with billing enabled (the free allowances apply to
paid accounts too), the gcloud CLI logged in, and in `.env.deploy`:
`GCP_PROJECT` (the project ID) and `GCP_REGION`.

```bash
python scripts/deploy_cloud_run.py --plan
python scripts/deploy_cloud_run.py
```

It shares steps 1 to 4 with the Space script, then enables the APIs, pushes the
locally built image to Artifact Registry and deploys. Expected cost outside the
free allowances: image storage above 0.5 GB (about 0.15 USD a month) and
outbound data outside North America (cents). Set a 1 USD budget alert.

Two Cloud Run facts learned on the first deploy:

- Google's front end answers `/healthz` with its own 404 page and never forwards
  it. Probe `/readyz` from outside; `/healthz` is only for the container's own
  health check.
- CPU is throttled while no request is in flight, so a background warm-up took
  ten minutes. The service is deployed with a startup probe on `/readyz`, which
  returns 503 until the models are warm, so the container keeps full CPU until
  then and the first request waits about a minute instead.

The first request after idle still pays that start; the keepalive workflow pings
`/readyz` to keep it rare. Do not print the service spec
(`gcloud run services describe`) into logs: it shows every secret in clear.

## 7. Confirm

```bash
curl -sS https://<host>/healthz
curl -sS https://<host>/readyz    # ready:true only once the models are warm
curl -sS https://<host>/v1/meta   # check `models` lists a model for each tier
```

Check three things in `/readyz`, each a silent failure mode:

- `checks.tokenizer` must be true and `detail.tokenizer` must read
  `cl100k_base`. A `fallback-heuristic` value means the tiktoken vocabulary is
  missing, which changes chunk boundaries and so chunk ids. Documents ingested
  in that state are not interchangeable with ones ingested correctly.
- `detail.artifact_root` must be the path or repository you intended. A
  mis-set root is writable and therefore looks healthy while storing artifacts
  somewhere useless.
- `detail.points` and `detail.documents` should agree with what you expect.

`/v1/meta` showing an empty `models` object means no provider key reached the
container. That is the most common first-deploy mistake, and it surfaces as
`MODEL_UNAVAILABLE` on the first query rather than as a startup failure, because
retrieval still works without a model.

Then ingest one document and ask one question you know the answer to. A verified
citation pointing at the right page is the only proof that the whole chain works.

## 8. Register the webhook

In the dataset repository settings, add a webhook to
`https://<host>/v1/internal/hf-webhook` with the `HF_WEBHOOK_SECRET` as the
`X-Webhook-Secret` header. Without it the API only refreshes its document
registry on restart, so documents ingested in Colab stay invisible until then.

## Rollback

The Space keeps its build history: revert to the previous build in the Space UI,
or push an earlier tag. Nothing in a deployment migrates data, so a rollback is
safe. The one exception is a collection rename, which is covered in
`rebuild-vector-store.md`.
