# Runbooks

| runbook | when |
|---------|------|
| [deploy.md](deploy.md) | standing the system up for the first time |
| [ingest-run.md](ingest-run.md) | processing a batch of documents |
| [rebuild-vector-store.md](rebuild-vector-store.md) | the cluster was suspended, deleted, or counts disagree |
| [migrate-store.md](migrate-store.md) | moving between Qdrant and Pinecone |
| [rotate-keys.md](rotate-keys.md) | a credential leaked or is due for rotation |
| [quota-exhausted.md](quota-exhausted.md) | queries return 503 QUOTA_EXHAUSTED |
| [bad-answers.md](bad-answers.md) | answers are wrong, missing, or over-refused |

Each one should be runnable by someone who did not write the system. If a step
needs judgement, the runbook says what to look at.
