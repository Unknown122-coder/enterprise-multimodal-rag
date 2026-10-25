# %% [markdown]
# # Offline evaluation
#
# Runs the evaluation harness and shows where the system is weak. Two modes:
#
# * `retrieval` costs nothing and answers "did we find the right page?";
# * `full` adds generation, citation verification and judge scores, at roughly
#   one and a half model calls per item.
#
# Start with `retrieval`. If recall is low, no amount of prompt work will fix the
# answers, and generation spend is wasted.

# %%
# ruff: noqa: E402, F821
import os

os.environ.setdefault("APP_ENV", "prod")
os.environ.setdefault("ALLOW_DEV_ANON", "false")
os.environ.setdefault("TENANTS_JSON", '[{"tenant_id": "demo", "keys": []}]')

from pathlib import Path

from emrag.api.deps import build_state
from emrag.eval.metrics import load_golden
from emrag.eval.run_eval import evaluate
from emrag.observability.logging import configure_logging
from emrag.settings import get_settings

configure_logging("WARNING", json_output=False)
settings = get_settings()
state = build_state(settings)

GOLDEN = Path("tests/golden/v1.jsonl")
items = load_golden(GOLDEN)
print(f"{len(items)} item(s): {sum(1 for i in items if not i.answerable)} unanswerable")

# %% [markdown]
# ## 1. Retrieval evaluation
#
# `recall@8` is the headline number. `quote_coverage` is the one that predicts
# citation quality: it asks whether the exact span an answer needs was retrieved,
# not merely the right page.

# %%
await state.store.ensure_schema(settings.embed_dim, sparse=True)  # type: ignore[top-level-await]
await state.registry.refresh()  # type: ignore[top-level-await]

retrieval_report = await evaluate(  # type: ignore[top-level-await]
    state, items, mode="retrieval", top_k=8, judge=False
)
for name, value in retrieval_report["metrics"].items():
    print(f"{name:28} {value}")
print("latency", retrieval_report["latency"])

# %% [markdown]
# ## 2. Which questions failed?
#
# Read these rather than the averages. A recall failure is usually one of three
# things: vocabulary mismatch the sparse side should have caught, an answer split
# across a chunk boundary, or a question that is genuinely ambiguous.

# %%
failures = [r for r in retrieval_report["per_item"] if r.get("recall") == 0.0]
by_id = {item.id: item for item in items}
print(f"{len(failures)} retrieval failure(s)\n")
for row in failures:
    item = by_id[row["id"]]
    print(f"{item.id}  [{item.category}/{item.difficulty}]  {item.question}")
    print(f"    expected pages {item.expected_pages} in {item.source_file or 'any document'}")
    if item.gold_quotes:
        print(f"    gold quote: {item.gold_quotes[0][:110]}")
    print()

# %% [markdown]
# ## 3. Tune the retrieval gate
#
# The gate decides when to refuse without calling a model. Too low and the system
# hallucinates on weak evidence; too high and it refuses answerable questions.
# Sweep it against the answerable and unanswerable subsets and pick the knee.

# %%
from emrag.core.types import Principal
from emrag.retrieval import filters as filter_builder
from emrag.retrieval.retriever import RetrievalOptions


async def gate_sweep(thresholds: list[float]) -> None:
    """Report answerable pass rate and unanswerable block rate per threshold."""
    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]
    scores: dict[str, float] = {}

    for item in items:
        principal = Principal(
            subject="eval",
            tenant_id=item.tenant_id,
            groups=["all"],
            clearance="confidential",
            role="admin",
        )
        outcome = await state.retriever.retrieve(
            item.question,
            filter_builder.from_request(principal),
            RetrievalOptions(top_k=8, rewrite="off", parent_expansion=False, gate_tau=0.0),
        )
        scores[item.id] = outcome.gate_score or 0.0

    print(f"{'tau':>6} {'answerable pass':>16} {'unanswerable blocked':>22}")
    for tau in thresholds:
        passed = sum(1 for i in answerable if scores[i.id] >= tau) / max(1, len(answerable))
        blocked = sum(1 for i in unanswerable if scores[i.id] < tau) / max(1, len(unanswerable))
        print(f"{tau:>6.2f} {passed:>16.0%} {blocked:>22.0%}")


await gate_sweep([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])  # type: ignore[top-level-await]

# %% [markdown]
# ## 4. Full evaluation
#
# Only worth running once retrieval is healthy. Set `JUDGE = False` to skip the
# judge calls and keep the free-tier quota for interactive use.

# %%
JUDGE = True

full_report = await evaluate(  # type: ignore[top-level-await]
    state, items, mode="full", top_k=8, judge=JUDGE
)
for name, value in full_report["metrics"].items():
    print(f"{name:28} {value}")
print("latency", full_report["latency"])

# %% [markdown]
# ## 5. Check the guarantee
#
# `citation_accuracy` must be exactly 1.0. It is enforced by construction, so
# anything less means the verifier was bypassed and that is a bug, not a tuning
# problem.

# %%
accuracy = full_report["metrics"].get("citation_accuracy")
print("citation_accuracy:", accuracy)
assert accuracy is None or accuracy == 1.0, "unverified citation reached a response"

wrong_refusals = [
    row
    for row in full_report["per_item"]
    if by_id[row["id"]].answerable and row.get("status") in ("not_found", "refused")
]
over_answers = [
    row
    for row in full_report["per_item"]
    if not by_id[row["id"]].answerable and row.get("status") in ("answered", "partial")
]
print(f"refused but answerable: {len(wrong_refusals)}")
print(f"answered but unanswerable: {len(over_answers)}")
for row in over_answers:
    print(f"  {row['id']}: {by_id[row['id']].question}")

# %% [markdown]
# ## 6. Save the report

# %%
import json

out = Path("var/eval-report.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps({"retrieval": retrieval_report, "full": full_report}, indent=2), encoding="utf-8"
)
print("wrote", out)

await state.aclose()  # type: ignore[top-level-await]
