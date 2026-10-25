---
title: Enterprise Multimodal RAG
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Document question answering with verifiable citations
---

# Enterprise Multimodal RAG

Answers questions over corporate documents with citations verified against the
source text before they are returned. Every claim names a document, page, chunk
and a quote that provably appears in that chunk; when the corpus does not support
an answer the response status is `not_found` rather than prose.

API docs are at `/docs`. Health is at `/healthz`, readiness at `/readyz`.

## Required secrets

Set these in Space settings, under Variables and secrets:

| secret | purpose |
|--------|---------|
| `QDRANT_URL`, `QDRANT_API_KEY` | free Qdrant cluster |
| `GROQ_API_KEY` | text generation |
| `GEMINI_API_KEY` | vision captioning and the volume answer tier |
| `HF_TOKEN`, `HF_DATASET_REPO` | private artifact and registry store |
| `HF_WEBHOOK_SECRET` | registry refresh webhook |
| `TENANTS_JSON` | tenant and API key configuration |

Public variables: `APP_ENV=prod`, `ALLOW_DEV_ANON=false`, `ARTIFACT_BACKEND=hf`.

This file is copied to `README.md` inside the Space by the deploy workflow; the
repository keeps its own README.
