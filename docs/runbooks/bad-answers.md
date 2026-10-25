# Runbook: answers are wrong, missing, or over-refused

Diagnose in this order. Each step is cheaper than the next, and a failure at one
level makes everything above it meaningless.

## 1. Was the right passage retrieved?

Retrieval first, always. If the evidence was never retrieved, no prompt change
will help.

```bash
curl -sS -X POST https://<host>/v1/retrieve \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"question":"<the failing question>","top_k":10}'
```

Look for the passage you expect in the output.

| finding | likely cause | fix |
|---------|--------------|-----|
| passage absent entirely | not ingested, or ingested degraded | check the document's `quality_report` |
| passage present but ranked low | vocabulary mismatch | enable rewriting (`"rewrite":"on"`), check the reranker is on |
| answer split across two passages | chunk boundary | enable `parent_expansion`, or lower `child_target_tokens` and re-ingest |
| wrong document entirely | filters too loose | pass `doc_ids` or `tags` |

Measure rather than guess:

```bash
python -m emrag.eval.run_eval --mode retrieval --golden tests/golden/v1.jsonl
```

`notebooks/03_eval_offline.ipynb` prints the failing questions with their
expected pages, which is faster than reading averages.

## 2. Was the gate too strict?

A `not_found` with `reason: retrieval_below_threshold` means no model was called.

```bash
curl -sS -X POST https://<host>/v1/query -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d '{"question":"..."}' | jq '.retrieval.gate_score'
```

Compare `gate_score` with `RETRIEVAL_GATE_TAU`. Sweep the threshold with the gate
cell in notebook 03: it reports, per candidate threshold, how many answerable
questions still pass and how many unanswerable ones are blocked. Pick the knee.
Do not tune it from a single example.

## 3. Were claims dropped?

A `partial` status, or `not_found` with `all_claims_unverified`, means the model
cited something that does not exist in the chunk it named.

Turn on `RETRIEVAL_DEBUG=true` and re-ask. The `debug.verification` block shows
`honesty_rate` and the dropped claim texts.

| finding | meaning | fix |
|---------|---------|-----|
| low `honesty_rate` across many queries | the model is paraphrasing inside quotes | the `quality` tier usually cites more faithfully; keep `repair` enabled |
| quotes nearly match | OCR noise beyond the threshold | raise `QUOTE_MATCH_THRESHOLD` cautiously, from 90 to 85, and re-run the full evaluation |
| correct answers dropped as unquotable | the answer genuinely has no verbatim span | expected behaviour; see ADR 0005 |

## 4. Is the answer ungrounded?

If a claim is shown, its quote exists in the cited chunk: that is mechanically
guaranteed. What is not guaranteed is that the quote **entails** the claim.

```bash
python -m emrag.eval.run_eval --mode full --min-faithfulness 0.85
```

A low faithfulness score with high citation accuracy means the model is quoting
real text that does not support what it says. Switch the failing queries to the
`quality` tier and compare.

## 5. Over-refusing

If `not_found_correct` is high on the unanswerable subset but users complain
about refusals on answerable questions, the gate is too strict (step 2) or
retrieval is weak (step 1). Lowering the gate without fixing retrieval trades
refusals for hallucinations.

## What never to do

Do not disable the verifier or the gate to make answers appear. The guarantee in
ADR 0005 is the product. If answers are missing, the fix is upstream in
retrieval, not downstream in verification.
