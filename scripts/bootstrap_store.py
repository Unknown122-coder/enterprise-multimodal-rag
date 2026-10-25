"""Create the vector collection and its payload indexes. Idempotent.

python scripts/bootstrap_store.py
"""

from __future__ import annotations

import asyncio
import sys

from emrag.observability.logging import configure_logging
from emrag.settings import get_settings
from emrag.vectorstore.factory import build_vector_store


async def main() -> int:
    """Ensure the schema exists and report the current point count."""
    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)
    store = build_vector_store(settings)
    try:
        await store.ensure_schema(settings.embed_dim, sparse=True)
        health = await store.health()
        sys.stdout.write(
            f"backend={health.backend} collection={health.collection} "
            f"reachable={health.reachable} points={health.points}\n"
        )
        return 0 if health.reachable else 1
    finally:
        await store.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
