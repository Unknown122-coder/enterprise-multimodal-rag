"""Check every external account the system depends on.

Run it after adding each key to ``.env``. Each check makes the cheapest possible
call (a model listing, a health probe, a whoami) so nothing here spends quota.
Secrets are never printed; only whether each one is set and whether it works.

    python scripts/check_setup.py

Exit code is 0 when everything that is configured works, 1 when a configured
service fails, and 2 when nothing is configured at all.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

import httpx

from emrag.settings import get_settings

TIMEOUT = httpx.Timeout(15.0, connect=5.0)


@dataclass
class Result:
    """Outcome of one check."""

    name: str
    status: str  # ok | failed | not_set
    detail: str


async def check_groq(api_key: str | None) -> Result:
    """List models, which is free and proves the key is accepted."""
    if not api_key:
        return Result("Groq", "not_set", "GROQ_API_KEY is empty")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as exc:
        return Result("Groq", "failed", f"network error: {exc}")
    if response.status_code == 401:
        return Result("Groq", "failed", "key rejected (401). Re-copy it from console.groq.com")
    if response.status_code != 200:
        return Result("Groq", "failed", f"HTTP {response.status_code}: {response.text[:120]}")
    names = {m.get("id") for m in response.json().get("data", [])}
    wanted = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
    ]
    present = [w for w in wanted if w in names]
    missing = [w for w in wanted if w not in names]
    detail = (
        f"{len(names)} models available; catalogue models present: {len(present)}/{len(wanted)}"
    )
    if missing:
        detail += f"; missing from your account: {missing} (edit configs/providers.yaml)"
    return Result("Groq", "ok", detail)


async def check_gemini(api_key: str | None) -> Result:
    """List models via the Gemini API."""
    if not api_key:
        return Result("Gemini", "not_set", "GEMINI_API_KEY is empty")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": api_key, "pageSize": 100},
            )
    except httpx.HTTPError as exc:
        return Result("Gemini", "failed", f"network error: {exc}")
    if response.status_code in (400, 403):
        return Result(
            "Gemini",
            "failed",
            f"key rejected ({response.status_code}). Re-copy it from aistudio.google.com/apikey",
        )
    if response.status_code != 200:
        return Result("Gemini", "failed", f"HTTP {response.status_code}: {response.text[:120]}")
    names = {m.get("name", "").removeprefix("models/") for m in response.json().get("models", [])}
    wanted = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    present = [w for w in wanted if w in names]
    missing = [w for w in wanted if w not in names]
    detail = (
        f"{len(names)} models available; catalogue models present: {len(present)}/{len(wanted)}"
    )
    if missing:
        detail += f"; missing: {missing} (edit configs/providers.yaml)"
    return Result("Gemini", "ok", detail)


async def check_qdrant(url: str, api_key: str | None, collection: str) -> Result:
    """Probe the vector store through the real adapter."""
    from emrag.vectorstore.qdrant_store import QdrantStore

    if url.startswith(("local://", ":memory:")):
        return Result("Qdrant", "ok", f"local mode ({url}); no cloud account needed for this")
    store = QdrantStore(url=url, collection=collection, api_key=api_key, timeout=10)
    try:
        health = await store.health()
    finally:
        await store.aclose()
    if not health.reachable:
        hint = (
            "is `docker compose up -d qdrant` running?"
            if "localhost" in url
            else "check QDRANT_URL and QDRANT_API_KEY"
        )
        return Result("Qdrant", "failed", f"unreachable at {url}: {health.detail}. {hint}")
    if health.points is None:
        return Result(
            "Qdrant",
            "ok",
            f"reachable at {url}; collection {collection} not created yet, "
            "run scripts/bootstrap_store.py",
        )
    return Result(
        "Qdrant", "ok", f"reachable at {url}; collection {collection} holds {health.points} points"
    )


async def check_hf(token: str | None, repo: str | None, backend: str) -> Result:
    """Confirm the token works and the dataset repo is reachable."""
    if backend != "hf" and not token:
        return Result(
            "Hugging Face", "not_set", "ARTIFACT_BACKEND=local; HF not needed until you deploy"
        )
    if not token:
        return Result("Hugging Face", "failed", "ARTIFACT_BACKEND=hf but HF_TOKEN is empty")
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        who = await asyncio.to_thread(api.whoami)
    except Exception as exc:  # noqa: BLE001 - any failure means the token is unusable
        return Result("Hugging Face", "failed", f"token rejected: {str(exc)[:120]}")
    user = who.get("name", "?")
    if not repo:
        return Result(
            "Hugging Face", "ok", f"token works (user {user}); HF_DATASET_REPO not set yet"
        )
    try:
        info = await asyncio.to_thread(api.repo_info, repo, repo_type="dataset")
        private = "private" if info.private else "PUBLIC - make it private"
        return Result(
            "Hugging Face",
            "ok",
            f"token works (user {user}); dataset {repo} exists and is {private}",
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            "Hugging Face",
            "failed",
            f"token works (user {user}) but dataset {repo} is not reachable: {str(exc)[:100]}",
        )


def check_tenants(definitions: list[dict[str, object]], anon: bool, env: str) -> Result:
    """Report how callers will authenticate."""
    usable = sum(
        1
        for t in definitions
        for k in t.get("keys", [])  # type: ignore[union-attr]
        if "REPLACE_ME" not in str(k.get("argon2_hash", ""))  # type: ignore[union-attr]
    )
    if env == "prod" and not usable:
        return Result(
            "Tenants",
            "failed",
            "APP_ENV=prod needs at least one real API key; run scripts/make_api_key.py",
        )
    if anon:
        return Result(
            "Tenants",
            "ok",
            f"dev anonymous access on (tenant demo); {usable} real API key(s) configured",
        )
    if not usable:
        return Result(
            "Tenants",
            "failed",
            "anonymous access is off and no API keys exist; run scripts/make_api_key.py",
        )
    return Result("Tenants", "ok", f"{usable} API key(s) across {len(definitions)} tenant(s)")


async def main() -> int:
    """Run every check and print a table."""
    cfg = get_settings()
    results = [
        await check_groq(cfg.groq_api_key),
        await check_gemini(cfg.gemini_api_key),
        await check_qdrant(cfg.qdrant_url, cfg.qdrant_api_key, cfg.collection_name),
        await check_hf(cfg.hf_token, cfg.hf_dataset_repo, cfg.artifact_backend),
        check_tenants(cfg.tenant_definitions, cfg.allow_dev_anon, cfg.app_env),
    ]

    icon = {"ok": "OK     ", "failed": "FAILED ", "not_set": "not set"}
    sys.stdout.write(
        f"\nAPP_ENV={cfg.app_env}  embed={cfg.embed_backend}  store={cfg.vector_backend}  artifacts={cfg.artifact_backend}\n\n"
    )
    for r in results:
        sys.stdout.write(f"  {icon[r.status]}  {r.name:14} {r.detail}\n")

    llm_ok = any(r.status == "ok" for r in results if r.name in ("Groq", "Gemini"))
    sys.stdout.write("\n")
    if not llm_ok:
        sys.stdout.write(
            "  Answers need at least one of Groq or Gemini. Retrieval works without either.\n"
        )
    failed = [r for r in results if r.status == "failed"]
    configured = [r for r in results if r.status != "not_set"]
    if failed:
        sys.stdout.write(f"  {len(failed)} check(s) failed.\n\n")
        return 1
    if not configured:
        return 2
    sys.stdout.write("  Everything configured is working.\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
