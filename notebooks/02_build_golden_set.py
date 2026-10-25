# %% [markdown]
# # Build the golden Q&A set
#
# A retrieval system without a golden set cannot be improved, only fiddled with.
# This notebook turns an ingested corpus into `tests/golden/v1.jsonl`.
#
# The method matters more than the tooling:
#
# 1. **sample** chunks across documents, pages and modalities, so the set is not
#    all easy first-page factoids;
# 2. **propose** a question, answer and supporting quote per chunk with a model;
# 3. **verify mechanically** that the proposed quote actually appears in the
#    chunk, using the same verifier the serving path uses. A proposal that fails
#    is discarded, never edited into shape;
# 4. **review by hand**. Every item a human accepts becomes gold; the model only
#    drafts.
#
# Target composition is in `tests/golden/README.md`. The adversarial 15 % is
# written by hand at the end, because the interesting unanswerable questions are
# the ones a model would not think to ask.

# %%
# ruff: noqa: E402, F821
import os

os.environ.setdefault("APP_ENV", "prod")
os.environ.setdefault("ALLOW_DEV_ANON", "false")
os.environ.setdefault("TENANTS_JSON", '[{"tenant_id": "demo", "keys": []}]')

import json
import random
from pathlib import Path

from emrag.api.deps import build_state
from emrag.observability.logging import configure_logging
from emrag.settings import get_settings

configure_logging("WARNING", json_output=False)
settings = get_settings()
state = build_state(settings)

TENANT = "demo"
TARGET_ITEMS = 85  # the remaining 15 are the hand-written adversarial subset
OUTPUT = Path("tests/golden/v1.generated.jsonl")
random.seed(17)

# %% [markdown]
# ## 1. Sample chunks
#
# Stratified by modality so tables and figures are represented in proportion to
# the composition target, not in proportion to how common they are.

# %%
QUOTAS = {"text": 0.55, "table": 0.25, "figure": 0.20}

await state.registry.refresh()  # type: ignore[top-level-await]
documents, _ = await state.registry.list_documents(TENANT, limit=200)  # type: ignore[top-level-await]
documents = [d for d in documents if d.status in ("ready", "ready_degraded")]
print(f"{len(documents)} ready document(s)")

by_modality: dict[str, list] = {"text": [], "table": [], "figure": []}
for record in documents:
    cursor = None
    while True:
        chunks, cursor = await state.store.list_chunks(  # type: ignore[top-level-await]
            TENANT, record.doc_id, limit=500, offset=cursor
        )
        for chunk in chunks:
            key = "text" if chunk.modality in ("text", "ocr_text") else chunk.modality
            if key in by_modality and chunk.token_count >= 40:
                by_modality[key].append(chunk)
        if not cursor:
            break

sample = []
for modality, share in QUOTAS.items():
    pool = by_modality[modality]
    random.shuffle(pool)
    sample.extend(pool[: int(TARGET_ITEMS * share)])
random.shuffle(sample)
print({k: len(v) for k, v in by_modality.items()}, "-> sampled", len(sample))

# %% [markdown]
# ## 2. Propose questions
#
# One call per batch of chunks, on the `fast` tier. The prompt asks for a
# question a colleague would actually ask, and for the quote that answers it.

# %%
from emrag.llm.base import LLMMessage, LLMRequest

PROPOSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "question", "answer", "quote", "difficulty"],
                "properties": {
                    "index": {"type": "integer"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "quote": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
            },
        }
    },
}

SYSTEM = (
    "You write evaluation questions for a document search system. For each passage, "
    "write one question a colleague would plausibly ask, the short answer, and the "
    "exact span of the passage that answers it. The quote must be copied verbatim "
    "from the passage, at least six words long. Prefer questions about specific "
    "numbers, dates, names and conditions over vague summaries. Never write a "
    "question the passage does not answer. Reply with JSON only."
)

BATCH = 5
proposals: list[dict] = []

for start in range(0, len(sample), BATCH):
    batch = sample[start : start + BATCH]
    listing = "\n\n".join(
        f"Passage {i} (page {c.page}, {c.modality}):\n{c.text[:1500]}"
        for i, c in enumerate(batch)
    )
    try:
        response = await state.router.complete(  # type: ignore[top-level-await]
            "fast",
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content=SYSTEM),
                    LLMMessage(role="user", content=listing),
                ],
                max_tokens=1200,
                json_schema=PROPOSE_SCHEMA,
                schema_name="golden_proposals",
            ),
        )
        payload = json.loads(response.text or "{}")
    except Exception as exc:
        print(f"batch at {start} failed: {str(exc)[:160]}")
        continue
    for item in payload.get("items", []):
        index = int(item.get("index", -1))
        if 0 <= index < len(batch):
            proposals.append({"chunk": batch[index], "proposal": item})

print(f"{len(proposals)} proposal(s)")

# %% [markdown]
# ## 3. Verify quotes mechanically
#
# The same matcher the serving path uses. A proposal whose quote does not appear
# in its chunk is discarded: if the verifier would reject it at serving time, it
# has no business being gold.

# %%
from emrag.citation.verifier import match_quote

verified = []
for entry in proposals:
    chunk = entry["chunk"]
    quote = str(entry["proposal"].get("quote", ""))
    matched, score = match_quote(quote, chunk.text)
    if matched is None or score < 1.0:
        continue
    entry["proposal"]["quote"] = matched
    verified.append(entry)

print(f"{len(verified)} of {len(proposals)} proposals have verbatim quotes")

# %% [markdown]
# ## 4. Review by hand
#
# Run this cell and answer for each item: `y` accept, `n` reject, or type a
# replacement question. Nothing reaches the golden set without a decision here.
#
# In Colab this is a plain `input()` loop, which is slow but honest. Budget about
# twenty minutes per fifty items.

# %%
accepted: list[dict] = []
REVIEW = True  # set False to accept every verified proposal unreviewed

for position, entry in enumerate(verified, start=1):
    chunk, proposal = entry["chunk"], entry["proposal"]
    print(f"\n--- {position}/{len(verified)}  {chunk.filename} p{chunk.page} ({chunk.modality})")
    print("Q:", proposal["question"])
    print("A:", proposal["answer"])
    print("quote:", proposal["quote"][:160])
    if not REVIEW:
        accepted.append(entry)
        continue
    verdict = input("accept? [y/n/new question] ").strip()
    if verdict.lower() in ("n", "no"):
        continue
    if verdict and verdict.lower() not in ("y", "yes"):
        proposal["question"] = verdict
    accepted.append(entry)

print(f"\naccepted {len(accepted)} item(s)")

# %% [markdown]
# ## 5. Write the file
#
# `expected_chunk_ids` is recorded because it is the most precise retrieval
# target, and `expected_pages` alongside it because pages survive a re-chunking
# that invalidates chunk ids.

# %%
CATEGORY = {"text": "factoid", "ocr_text": "factoid", "table": "table", "figure": "figure"}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8") as handle:
    handle.write("// Generated by notebooks/02_build_golden_set.ipynb. Reviewed by hand.\n")
    for position, entry in enumerate(accepted, start=1):
        chunk, proposal = entry["chunk"], entry["proposal"]
        handle.write(
            json.dumps(
                {
                    "id": f"a{position:03d}",
                    "question": proposal["question"],
                    "answerable": True,
                    "tenant_id": TENANT,
                    "category": CATEGORY.get(chunk.modality, "factoid"),
                    "modality": chunk.modality,
                    "difficulty": proposal.get("difficulty", "medium"),
                    "source_file": chunk.filename,
                    "expected_doc_ids": [chunk.doc_id],
                    "expected_pages": [chunk.page],
                    "expected_chunk_ids": [chunk.chunk_id],
                    "gold_answer": proposal["answer"],
                    "gold_quotes": [proposal["quote"]],
                    "notes": "generated, human reviewed",
                },
                ensure_ascii=False,
            )
            + "\n"
        )

print(f"wrote {OUTPUT} with {len(accepted)} item(s)")

# %% [markdown]
# ## 6. Add the adversarial subset by hand
#
# Fifteen percent of the set should be questions the corpus cannot answer. These
# are worth writing yourself, in roughly equal parts:
#
# * **near miss**: the right topic, the wrong period or entity, where retrieval
#   will happily surface relevant-looking text;
# * **explicitly declined**: something a document says it does not provide, which
#   is the strongest possible test of grounding;
# * **absent**: a plausible fact that simply is not there;
# * **injection**: instructions hidden in a document, and instructions in the
#   question itself;
# * **cross-tenant**: a request for another tenant's data.
#
# Append them to the generated file, then rename it over `v1.jsonl` once the
# retrieval evaluation passes.

# %%
await state.aclose()  # type: ignore[top-level-await]
print("done")
