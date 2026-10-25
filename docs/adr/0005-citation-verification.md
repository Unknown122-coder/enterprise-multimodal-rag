# 0005 Verify quotes deterministically, do not trust the model

**Status** accepted · **Date** 2026-10-25

## Context

The product promise is that every claim is traceable. A model asked to cite its
sources will usually comply and will sometimes produce a citation that looks
right and is not: the correct document, a plausible page, a quote that does not
appear anywhere. That failure is invisible to the user, which makes it worse than
a refusal.

Nothing in a prompt makes this impossible. Only a check does.

## Decision

Four layers, in order of how much they contribute.

1. **A retrieval gate.** If the best reranked score is below a calibrated
   threshold, return `not_found` and make no model call. The commonest route to
   an ungrounded answer is a model handed weak evidence and trying anyway.
2. **A constrained grammar.** Claims carry an evidence alias plus a verbatim
   quote. Aliases are short labels for the passages actually retrieved, so there
   is no realistic identifier to invent.
3. **Mechanical verification.** The alias must be one shown for this request and
   the quote must appear in that chunk's text after normalisation. A near match
   above threshold is accepted and the quote is **replaced with the real span
   from the source**.
4. **A drop policy.** A claim with no verified citation is removed; if all are
   removed the status becomes `not_found`.

Hashing and verification share one normalisation function, so a quote can never
verify against text that was not indexed. That invariant is a test.

## Alternatives

**Trust the model.** Cheapest, and the failure mode is silent wrong answers.

**Ask a second model to check.** Doubles quota consumption, adds latency, and
replaces a deterministic check with a probabilistic one.

**Require exact substring matches only.** Attractive until OCR output, ligatures,
soft hyphens and curly quotes are involved. Fuzzy matching above a high threshold
plus span replacement gets both correctness and robustness.

## Consequences

- Post-filter citation accuracy is 1.0 by construction, and asserted in tests.
- An injected instruction in a document cannot produce a shown claim unless that
  claim is quotable from the corpus. This is the strongest injection defence here.
- A legitimate paraphrase with no quotable span is dropped. The system prefers
  silence to an unverifiable sentence, which occasionally loses a correct answer.
- One repair round is allowed, in which the model may re-cite the failed claims.
  After that, no second chance.
