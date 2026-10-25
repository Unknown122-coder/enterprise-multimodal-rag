"""Ingestion worker CLI.

This is what the Colab notebook actually calls, so the notebook holds no logic
and the pipeline version in use is a git tag rather than a copy-paste. It can
also ingest local files directly, which is how the integration tests and local
development drive the full pipeline.

Usage::

    python -m emrag.ingestion.cli jobs --tenant acme          # drain the inbox
    python -m emrag.ingestion.cli file report.pdf --tenant acme
    python -m emrag.ingestion.cli status --tenant acme
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
from pathlib import Path

from emrag.api.deps import AppState, build_state
from emrag.core.errors import NotFound
from emrag.core.types import Classification
from emrag.ingestion.pipeline import IngestRequest
from emrag.observability.logging import configure_logging, get_logger
from emrag.settings import get_settings
from emrag.storage.registry import inbox_key

logger = get_logger(__name__)


def worker_name() -> str:
    """Identify this worker in job records, so a double claim is visible."""
    return f"{socket.gethostname()}:{os.getpid()}"


async def drain_jobs(state: AppState, tenant: str | None, limit: int) -> int:
    """Claim and run every queued ingestion job. Returns the number processed."""
    jobs = await state.registry.list_queued_jobs(tenant)
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        sys.stdout.write("no queued jobs\n")
        return 0

    processed = 0
    for job in jobs:
        record = await state.registry.find_document(job.tenant_id, job.doc_id)
        if record is not None and record.status in ("ready", "ready_degraded"):
            await state.registry.update_job(job.job_id, status="ready")
            sys.stdout.write(f"skip {job.doc_id}: already {record.status}\n")
            continue
        try:
            await state.registry.claim_job(job.job_id, worker_name())
        except NotFound:
            sys.stdout.write(f"skip {job.job_id}: claimed by another worker\n")
            continue

        data = await _load_source(state, job.tenant_id, job.doc_id)
        if data is None:
            await state.registry.update_job(
                job.job_id, status="failed", error="source bytes not found in inbox"
            )
            sys.stderr.write(f"fail {job.doc_id}: source not found\n")
            continue

        request = IngestRequest(
            tenant_id=job.tenant_id,
            filename=record.filename if record else f"{job.doc_id}.pdf",
            data=data,
            doc_id=job.doc_id,
            title=record.title if record else None,
            acl_groups=record.acl_groups if record else ["all"],
            classification=record.classification if record else "internal",
            tags=record.tags if record else [],
            pii_mode=state.authenticator.pii_mode_for(job.tenant_id),  # type: ignore[arg-type]
            job_id=job.job_id,
            profile="full",
        )
        result = await state.pipeline.run(request)
        processed += 1
        sys.stdout.write(
            f"done {job.doc_id}: {result.chunks} chunks, {result.upserted} vectors, "
            f"{result.duration_s:.1f}s{' (degraded)' if result.degraded else ''}\n"
        )
    return processed


async def _load_source(state: AppState, tenant_id: str, doc_id: str) -> bytes | None:
    """Fetch a document's bytes from the inbox or from its artifacts."""
    for extension in ("pdf", "png", "jpg", "jpeg", "tif", "tiff"):
        try:
            return await state.artifacts.get_bytes(inbox_key(tenant_id, doc_id, extension))
        except Exception:  # noqa: BLE001 - try the next extension
            continue
    try:
        return await state.artifacts.get_bytes(f"artifacts/{tenant_id}/{doc_id}/source.pdf")
    except Exception:  # noqa: BLE001
        return None


async def ingest_file(
    state: AppState,
    path: Path,
    *,
    tenant: str,
    title: str | None,
    tags: list[str],
    classification: Classification,
    acl_groups: list[str],
) -> int:
    """Ingest a local file through the full pipeline."""
    if not await asyncio.to_thread(path.exists):
        sys.stderr.write(f"file not found: {path}\n")
        return 2
    data = await asyncio.to_thread(path.read_bytes)
    request = IngestRequest(
        tenant_id=tenant,
        filename=path.name,
        data=data,
        title=title,
        tags=tags,
        classification=classification,
        acl_groups=acl_groups or ["all"],
        pii_mode=state.authenticator.pii_mode_for(tenant),  # type: ignore[arg-type]
        profile="full",
    )
    result = await state.pipeline.run(request)
    sys.stdout.write(
        f"{result.record.doc_id}: {result.chunks} chunks, {result.parents} parents, "
        f"{result.upserted} vectors, {result.duration_s:.1f}s\n"
    )
    if result.degraded:
        sys.stdout.write(
            f"  degraded: {'; '.join(result.record.quality_report.degraded_reasons[:5])}\n"
        )
    return 0


async def show_status(state: AppState, tenant: str | None) -> int:
    """Print the registry contents and the vector count."""
    await state.registry.refresh()
    stats = await state.registry.stats()
    sys.stdout.write(f"documents: {stats['documents']} by_status={stats['by_status']}\n")
    try:
        count = await state.store.count(tenant_id=tenant)
        sys.stdout.write(f"vectors: {count}\n")
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(f"vectors: unavailable ({str(exc)[:120]})\n")
    jobs = await state.registry.list_queued_jobs(tenant)
    sys.stdout.write(f"queued jobs: {len(jobs)}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="emrag-ingest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    jobs = sub.add_parser("jobs", help="claim and run queued ingestion jobs")
    jobs.add_argument("--tenant", default=None)
    jobs.add_argument("--limit", type=int, default=0)

    single = sub.add_parser("file", help="ingest a local file")
    single.add_argument("path", type=Path)
    single.add_argument("--tenant", default="demo")
    single.add_argument("--title", default=None)
    single.add_argument("--tag", action="append", default=[])
    single.add_argument("--acl-group", action="append", default=[])
    single.add_argument(
        "--classification", choices=["public", "internal", "confidential"], default="internal"
    )

    status = sub.add_parser("status", help="show registry and vector counts")
    status.add_argument("--tenant", default=None)
    return parser


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)
    state = build_state(settings)
    try:
        await state.store.ensure_schema(settings.embed_dim, sparse=True)
        if args.command == "jobs":
            await drain_jobs(state, args.tenant, args.limit)
            return 0
        if args.command == "file":
            return await ingest_file(
                state,
                args.path,
                tenant=args.tenant,
                title=args.title,
                tags=list(args.tag),
                classification=args.classification,
                acl_groups=list(args.acl_group),
            )
        return await show_status(state, args.tenant)
    finally:
        await state.aclose()


def main() -> int:
    """Entry point for the ``emrag-ingest`` console script."""
    return asyncio.run(_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
