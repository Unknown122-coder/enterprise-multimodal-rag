"""Migrate between vector backends.

A thin CLI over :mod:`emrag.vectorstore.migrate`. Migration never re-parses or
re-embeds: the chunk artifacts already hold the vectors, so switching Qdrant to
Pinecone (or back) is a replay plus a verification pass.

The verification is the point. Replaying points fails loudly; a filter-syntax
difference between backends fails silently, as worse retrieval. So the script
measures recall on both backends with the same golden set and refuses to declare
success if the target is meaningfully worse.

    python scripts/migrate_store.py --to pinecone --verify-golden tests/golden/v1.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from emrag.observability.logging import configure_logging
from emrag.settings import Settings, get_settings
from emrag.storage.artifact_store import build_artifact_store
from emrag.vectorstore.base import VectorStore
from emrag.vectorstore.factory import build_vector_store
from emrag.vectorstore.migrate import compare_recall, replay_artifacts


def _store_for(backend: str, settings: Settings) -> VectorStore:
    """Build a store for an explicit backend, overriding the environment."""
    return build_vector_store(settings.model_copy(update={"vector_backend": backend}))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, choices=["qdrant", "pinecone"])
    parser.add_argument("--from", dest="source", default=None, choices=["qdrant", "pinecone"])
    parser.add_argument("--tenant", default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--verify-golden", default=None, help="golden set for a recall comparison")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-recall-delta", type=float, default=0.02)
    return parser


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    artifacts = build_artifact_store(settings)
    target = _store_for(args.to, settings)
    source = _store_for(args.source, settings) if args.source else None

    try:
        await target.ensure_schema(settings.embed_dim, sparse=True)
        report = await replay_artifacts(
            artifacts,
            target,
            tenant_id=None if args.tenant == "all" else args.tenant,
            batch_size=args.batch_size,
            verify=True,
        )
        if not report.documents:
            sys.stderr.write("no chunk artifacts found; nothing to migrate\n")
            return 2

        for entry in report.documents:
            status = "ok" if entry.ok else f"FAIL {entry.error or 'count mismatch'}"
            sys.stdout.write(
                f"{entry.doc_id}: {entry.upserted}/{entry.chunks} points"
                f"{f', target holds {entry.counted}' if entry.counted is not None else ''}"
                f"  {status}\n"
            )
        sys.stdout.write(
            f"\ndocuments={len(report.documents)} chunks={report.chunks} "
            f"upserted={report.upserted}\n"
        )

        failed = not report.ok
        if args.verify_golden:
            comparison = await compare_recall(
                target,
                settings,
                args.verify_golden,
                source=source,
                top_k=args.top_k,
                max_delta=args.max_recall_delta,
            )
            sys.stdout.write(f"{args.to} recall@{args.top_k} = {comparison.target:.3f}\n")
            if comparison.source is not None and comparison.delta is not None:
                sys.stdout.write(
                    f"{args.source} recall@{args.top_k} = {comparison.source:.3f} "
                    f"(target is {comparison.delta:+.3f})\n"
                )
            if not comparison.ok:
                sys.stderr.write(
                    f"FAIL recall dropped by {comparison.delta:.3f}, above the allowed "
                    f"{args.max_recall_delta:.3f}\n"
                )
                failed = True

        if failed:
            for entry in report.failures:
                sys.stderr.write(f"FAIL {entry.doc_id}: {entry.error or 'count mismatch'}\n")
            return 1

        sys.stdout.write(
            f"\nmigration verified. Set VECTOR_BACKEND={args.to} and restart the API.\n"
        )
        return 0
    finally:
        await target.aclose()
        if source is not None:
            await source.aclose()


def main() -> int:
    """Entry point."""
    return asyncio.run(_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
