"""Keep free-tier resources from idling out.

Hugging Face Spaces sleep after prolonged inactivity and a Qdrant free cluster can
be suspended, so a scheduled workflow pings both. Exits non-zero if either is
unreachable, which surfaces as a failed workflow run.

    python scripts/keepalive.py --url https://org-emrag.hf.space
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from emrag.settings import get_settings
from emrag.vectorstore.factory import build_vector_store


async def ping_api(url: str, timeout_s: float) -> bool:
    """Hit the readiness endpoint, tolerating a cold start."""
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
        for attempt in range(3):
            try:
                # /readyz, not /healthz: Google's front end answers /healthz on
                # Cloud Run with its own 404 and never reaches the container.
                response = await client.get(f"{url.rstrip('/')}/readyz")
                if response.status_code == 200:
                    sys.stdout.write(f"api ok after {attempt + 1} attempt(s)\n")
                    return True
            except httpx.HTTPError as exc:
                sys.stdout.write(f"api attempt {attempt + 1} failed: {str(exc)[:120]}\n")
            await asyncio.sleep(20)
    return False


async def ping_store() -> bool:
    """Issue a trivial count so the vector cluster registers activity."""
    store = build_vector_store(get_settings())
    try:
        health = await store.health()
        sys.stdout.write(f"store reachable={health.reachable} points={health.points}\n")
        return health.reachable
    finally:
        await store.aclose()


async def main() -> int:
    """Ping whichever targets were requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="public API base URL")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--skip-store", action="store_true")
    args = parser.parse_args()

    results: list[bool] = []
    if args.url:
        results.append(await ping_api(args.url, args.timeout))
    if not args.skip_store:
        results.append(await ping_store())
    return 0 if all(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
