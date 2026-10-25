# 0003 Host on Hugging Face Spaces, not Render

**Status** accepted · **Date** 2026-10-25

## Context

ADR 0002 puts the embedding and reranking models in the serving process, needing
roughly 1.5 GB of RAM. That single fact decides the host.

| | Spaces free | Render free |
|---|---|---|
| RAM | 16 GB | 512 MB |
| CPU | 2 vCPU | 0.1 shared |
| Sleeps after | ~48 h idle | ~15 min idle |
| Custom domain | paid | free |

## Decision

Hugging Face Spaces, Docker SDK, port 7860. Render is kept as a documented
fallback profile in `render.yaml`, not as a supported default.

## Alternatives

**Render free.** Cannot hold the models. Running there means switching to Gemini
embeddings and disabling the reranker, which reintroduces the quota coupling
ADR 0002 exists to avoid and costs retrieval precision. The profile exists
because someone will need a free custom domain more than they need precision,
and that should be a decision rather than a discovery.

**Fly.io, Railway, Cloud Run.** All viable; none offer 16 GB free.

## Consequences

- Local embedding and reranking work, so the query path stays quota-free.
- A 48-hour sleep window instead of 15 minutes, so keepalive runs every six
  hours rather than every ten minutes.
- No custom domain without paying; the API lives on a `*.hf.space` subdomain.
- The artifact store is already a Hugging Face dataset repository, so secrets and
  tokens are shared with the host.
- Deployment is a git push, which the tag workflow does.
