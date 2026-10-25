# Golden Q&A set

`v1.jsonl` is the evaluation set. One JSON object per line; `//` comments and blank
lines are ignored.

## Fields

| field | meaning |
|-------|---------|
| `id` | stable identifier, referenced in reports |
| `question` | the question as a user would ask it |
| `answerable` | `false` for the adversarial and unanswerable subset |
| `tenant_id` | tenant the question is asked against |
| `category` | `factoid`, `table`, `figure`, `multi_hop`, `cross_doc`, `adversarial` |
| `modality` | expected evidence modality |
| `difficulty` | `easy`, `medium`, `hard` |
| `source_file` | filename the answer lives in, for human reference |
| `expected_doc_ids` | optional; content-addressed ids, filled in after ingestion |
| `expected_pages` | pages that contain the answer; the durable retrieval target |
| `expected_chunk_ids` | optional; most precise target, invalidated by re-chunking |
| `gold_answer` | reference answer, for relevance judging |
| `gold_quotes` | verbatim spans the answer must rest on |
| `notes` | why the item exists, especially for adversarial items |

## Composition target (100 items)

| share | category |
|-------|----------|
| 30 % | factoid text |
| 20 % | table lookup |
| 15 % | figure or chart |
| 10 % | multi-hop within one document |
| 10 % | cross-document |
| 15 % | unanswerable or adversarial |

## Building the real set

`notebooks/02_build_golden_set.ipynb` samples chunks from the ingested corpus, has
a model propose a question, answer and quote for each, then requires a human to
accept or edit it. Every proposed quote is verified against the chunk text before
the item is written, so a gold quote can never be one the corpus does not contain.

## Re-chunking

A chunking change invalidates `expected_chunk_ids`. Re-map by `(expected_pages,
gold_quotes)` rather than regenerating the set, so the questions stay stable
across pipeline versions.

## Running

```bash
python -m emrag.eval.run_eval --mode retrieval --golden tests/golden/v1.jsonl --min-recall 0.85
python -m emrag.eval.run_eval --mode full --min-faithfulness 0.85 --report var/eval.json
```

`v1.jsonl` as shipped is a 20-item starter written against the two fixture
documents in `tests/fixtures/`. Replace it with your own corpus before trusting
the numbers.
