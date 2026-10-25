"""Rebuild a vector store from artifacts.

Artifacts are the source of truth, so this is the recovery tool: it replays
stored chunks and their already-computed vectors into the configured backend. No
parsing, no embedding, no LLM calls, which makes a full rebuild of a
30,000-chunk corpus a matter of minutes and costs nothing.

Use it when a free-tier cluster is suspended or deleted, after bumping the
collection name for a schema change, or whenever the vector count disagrees with
the registry. See ``docs/runbooks/rebuild-vector-store.md``.

    python scripts/upsert_artifacts.py --tenant all --verify
    python scripts/upsert_artifacts.py --tenant acme --doc d_3fa9c2e7b1d04a55
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from emrag.ingestion.artifacts import ArtifactReader
from emrag.observability.logging import configure_logging
from emrag.settings import get_settings
from emrag.storage.artifact_store import build_artifact_store
from emrag.storage.registry import Registry
from emrag.vectorstore.factory import build_vector_store
from emrag.vectorstore.migrate import replay_artifacts


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default="all", help="tenant id, or 'all'")
    parser.add_argument("--doc", default=None, help="restrict to one document id")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--verify", action="store_true", help="compare point counts after each document"
    )
    parser.add_argument(
        "--refresh-manifest", action="store_true", help="also rewrite manifest records"
    )
    return parser


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    artifacts = build_artifact_store(settings)
    store = build_vector_store(settings)

    try:
        await store.ensure_schema(settings.embed_dim, sparse=True)
        report = await replay_artifacts(
            artifacts,
            store,
            tenant_id=None if args.tenant == "all" else args.tenant,
            doc_id=args.doc,
            batch_size=args.batch_size,
            verify=args.verify,
        )
        if not report.documents:
            sys.stderr.write("no documents with chunk artifacts were found\n")
            return 2

        registry = Registry(artifacts) if args.refresh_manifest else None
        reader = ArtifactReader(artifacts) if args.refresh_manifest else None

        for entry in report.documents:
            line = f"{entry.doc_id}: {entry.upserted}/{entry.chunks} points"
            if entry.counted is not None:
                line += f", store reports {entry.counted}"
            if entry.error:
                line += f"  FAIL {entry.error}"
            if registry is not None and reader is not None and entry.ok:
                record = await reader.read_document(entry.tenant_id, entry.doc_id)
                if record is not None:
                    await registry.put_document(record)
                    line += ", manifest refreshed"
            sys.stdout.write(line + "\n")

        sys.stdout.write(
            f"\ndocuments={len(report.documents)} chunks={report.chunks} "
            f"upserted={report.upserted}\n"
        )
        for entry in report.failures:
            sys.stderr.write(f"WARN {entry.doc_id}: {entry.error or 'count mismatch'}\n")
        return 1 if report.failures else 0
    finally:
        await store.aclose()


def main() -> int:
    """Entry point."""
    return asyncio.run(_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
