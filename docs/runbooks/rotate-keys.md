# Runbook: rotate credentials

**When** a credential leaked, someone left, or rotation is due.

Only Argon2 hashes of API keys are stored, so a leaked **tenant configuration**
is not a leaked key. A leaked key itself must be revoked.

## Revoke a tenant API key

1. Issue the replacement first, so the tenant is never without one:

   ```bash
   python scripts/make_api_key.py --tenant acme --role writer --group finance
   ```

2. Add the new entry to the tenant's `keys` list and **remove the compromised
   entry**. Locally that is `configs/tenants.yaml`; in production it is the
   `TENANTS_JSON` secret.

3. Restart the service. Tenant configuration is read at startup.

4. Give the plaintext key to the tenant over a channel that is not the one that
   leaked.

5. Confirm the old key is dead:

   ```bash
   curl -sS -o /dev/null -w '%{http_code}\n' https://<host>/v1/documents \
     -H "Authorization: Bearer $OLD_KEY"   # expect 401
   ```

## Rotate a provider key

Groq, Gemini, Qdrant and Hugging Face all allow a second active key, so there is
no outage window.

1. Create the new key in the provider console.
2. Update the secret in the host (Space settings, or Render env group) and in the
   GitHub secrets used by the workflows.
3. Restart. For Gemini and Groq, confirm with `GET /v1/usage` that calls are
   being accepted.
4. Delete the old key in the provider console.

Rotating `HF_TOKEN` also affects the Colab notebook: update the Colab secret.

## Rotate the JWT signing secret

Rotating `JWT_HS256_SECRET` invalidates every live token immediately. Prefer
RS256 with a JWKS URL, where the identity provider can publish two keys and
rotate without invalidating anything.

## After any leak

- Check `audit` log events for the window the credential was exposed: every
  mutating call and every admin bypass is recorded with the principal.
- If an ingestion key leaked, list recently created documents and confirm each
  one is expected.

## Hygiene

- `gitleaks` runs in pre-commit and in the security workflow.
- The logging processor redacts anything whose key matches `key`, `token`,
  `secret`, `password` or `authorization`, recursively. Do not defeat it by
  renaming a field.
- Query text is not logged unless `LOG_QUERY_TEXT=true`.
