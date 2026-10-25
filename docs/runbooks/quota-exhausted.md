# Runbook: queries return 503 QUOTA_EXHAUSTED

**Meaning** every model in the requested tier is out of quota, cooling after a
429, or failing. Retrieval still works; only generation is blocked.

> If the code is `MODEL_UNAVAILABLE` rather than `QUOTA_EXHAUSTED`, this is
> the wrong runbook: no model is *configured* for that tier. Set
> `GROQ_API_KEY` or `GEMINI_API_KEY` and check `configs/providers.yaml`.
> `GET /v1/meta` shows which tiers currently have models.

## Diagnose

```bash
curl -sS https://<host>/v1/usage -H "Authorization: Bearer $ADMIN_KEY"
```

Read the `reason` suffix in the error detail, which names the model and why:

| reason | meaning | clears in |
|--------|---------|-----------|
| `rpm` | requests this minute | under a minute |
| `tpm` | tokens this minute | under a minute |
| `rpd` | requests today | at UTC midnight |
| `tpd` | tokens today | at UTC midnight |
| `cooling` | a 429 or 5xx was seen | 60 seconds |
| `error` | the provider is failing | until it recovers |

`retry_after_s` on the problem document is the honest estimate.

## Immediate mitigations

1. **Use the volume tier.** If callers are sending `model_tier: "quality"`, that
   model has by far the smallest daily budget. The `fast` tier has roughly five
   times the capacity.
2. **Reduce tokens per query.** Lower `EVIDENCE_TOKEN_BUDGET` to 1,800 and
   `TOP_K` to 6. This is the fastest way to stretch a daily token budget.
3. **Raise the gate.** A higher `RETRIEVAL_GATE_TAU` refuses weak questions
   without a model call. Sweep it with notebook 03 first so you know what it
   costs in recall.
4. **Add a provider.** If only one of `GROQ_API_KEY` or `GEMINI_API_KEY` is set,
   setting the other roughly doubles capacity, and the router will use it with no
   other change.

## Structural fixes

- Check `daily_reserve_ratio` in `configs/providers.yaml`. It holds back 10 % of
  each daily budget so a batch job cannot starve interactive queries. Raise it if
  batch work keeps draining the day.
- Add models to the `fast` tier in the catalogue. The router picks by `priority`,
  lowest first, and needs no code change.
- Confirm the published limits in the catalogue still match the provider console.
  They change.

## Verify the ledger is honest

The ledger persists daily counters, so a restart does not reset them. If usage
looks implausibly low after a restart, check that
`ARTIFACT_LOCAL_ROOT/state/usage.json` is writable; on an ephemeral filesystem it
will not survive, and the ledger will under-count until midnight.
