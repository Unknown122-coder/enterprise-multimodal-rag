"""Document and job registry.

Backed entirely by the artifact store so the free tier needs no database. An
in-process cache keeps list and get operations off the network; it is refreshed on
startup, after local writes, and when the Hugging Face webhook fires.

Concurrency note: writes are last-writer-wins. That is acceptable because a
document is only ever written by one ingestion worker at a time, and the claim
protocol (:meth:`claim_job`) makes double-processing visible rather than silent.
"""

from __future__ import annotations

import asyncio
from typing import Any

from emrag.core.errors import NotFound
from emrag.core.ids import new_job_id
from emrag.core.types import DocumentRecord, Job, JobStatus, utcnow
from emrag.observability.logging import get_logger
from emrag.storage.artifact_store import ArtifactStore

logger = get_logger(__name__)


def manifest_key(tenant_id: str, doc_id: str) -> str:
    """Key of a document's manifest record."""
    return f"manifest/{tenant_id}/{doc_id}.json"


def job_key(job_id: str) -> str:
    """Key of a job record."""
    return f"jobs/{job_id}.json"


def inbox_key(tenant_id: str, doc_id: str, ext: str = "pdf") -> str:
    """Key of an uploaded source file awaiting ingestion."""
    return f"inbox/{tenant_id}/{doc_id}.{ext}"


def artifacts_prefix(tenant_id: str, doc_id: str) -> str:
    """Prefix holding a document's derived artifacts."""
    return f"artifacts/{tenant_id}/{doc_id}"


class Registry:
    """Read-through cache over the artifact store for documents and jobs."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._docs: dict[tuple[str, str], DocumentRecord] = {}
        self._jobs: dict[str, Job] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    @property
    def store(self) -> ArtifactStore:
        """The underlying artifact store."""
        return self._store

    # ---- sync ---------------------------------------------------------------
    async def refresh(self) -> int:
        """Reload every manifest from the artifact store. Returns the count."""
        async with self._lock:
            keys = await self._store.list_prefix("manifest")
            docs: dict[tuple[str, str], DocumentRecord] = {}
            for key in keys:
                if not key.endswith(".json"):
                    continue
                payload = await self._store.get_json_or_none(key)
                if payload is None:
                    continue
                try:
                    record = DocumentRecord.model_validate(payload)
                except Exception as exc:  # noqa: BLE001 - one bad file must not block startup
                    logger.warning("registry.manifest_invalid", key=key, error=str(exc)[:200])
                    continue
                docs[(record.tenant_id, record.doc_id)] = record
            self._docs = docs
            self._loaded = True
            logger.info("registry.refreshed", documents=len(docs))
            return len(docs)

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self.refresh()

    # ---- documents ----------------------------------------------------------
    async def put_document(self, record: DocumentRecord) -> DocumentRecord:
        """Persist a document record and update the cache."""
        record.updated_at = utcnow()
        if not record.artifacts_prefix:
            record.artifacts_prefix = artifacts_prefix(record.tenant_id, record.doc_id)
        await self._store.put_json(
            manifest_key(record.tenant_id, record.doc_id), record.model_dump(mode="json")
        )
        self._docs[(record.tenant_id, record.doc_id)] = record
        return record

    async def get_document(self, tenant_id: str, doc_id: str) -> DocumentRecord:
        """Return one document record, raising :class:`NotFound` if absent."""
        await self._ensure_loaded()
        cached = self._docs.get((tenant_id, doc_id))
        if cached is not None:
            return cached
        payload = await self._store.get_json_or_none(manifest_key(tenant_id, doc_id))
        if payload is None:
            raise NotFound(f"document {doc_id} not found")
        record = DocumentRecord.model_validate(payload)
        self._docs[(tenant_id, doc_id)] = record
        return record

    async def find_document(self, tenant_id: str, doc_id: str) -> DocumentRecord | None:
        """Return one document record or ``None``."""
        try:
            return await self.get_document(tenant_id, doc_id)
        except NotFound:
            return None

    async def list_documents(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        name_contains: str | None = None,
    ) -> tuple[list[DocumentRecord], str | None]:
        """List a tenant's documents, newest first, with cursor pagination."""
        await self._ensure_loaded()
        records = [r for (t, _), r in self._docs.items() if t == tenant_id]
        if status:
            records = [r for r in records if r.status == status]
        if tag:
            records = [r for r in records if tag in r.tags]
        if name_contains:
            needle = name_contains.lower()
            records = [r for r in records if needle in r.filename.lower()]
        records.sort(key=lambda r: (r.created_at, r.doc_id), reverse=True)

        start = 0
        if cursor:
            ids = [r.doc_id for r in records]
            start = ids.index(cursor) + 1 if cursor in ids else 0
        page = records[start : start + limit]
        next_cursor = page[-1].doc_id if len(records) > start + limit and page else None
        return page, next_cursor

    async def delete_document(self, tenant_id: str, doc_id: str) -> None:
        """Remove a document's manifest, artifacts and inbox copy."""
        await self._store.delete(manifest_key(tenant_id, doc_id))
        await self._store.delete_prefix(artifacts_prefix(tenant_id, doc_id))
        for ext in ("pdf", "png", "jpg", "jpeg", "tif", "tiff"):
            await self._store.delete(inbox_key(tenant_id, doc_id, ext))
        self._docs.pop((tenant_id, doc_id), None)

    # ---- jobs ---------------------------------------------------------------
    async def create_job(self, *, kind: str, doc_id: str, tenant_id: str, pages: int = 0) -> Job:
        """Create and persist a queued job."""
        job = Job(job_id=new_job_id(), kind=kind, doc_id=doc_id, tenant_id=tenant_id)  # type: ignore[arg-type]
        job.progress.pages_total = pages
        await self.put_job(job)
        return job

    async def put_job(self, job: Job) -> Job:
        """Persist a job record and update the cache."""
        job.updated_at = utcnow()
        await self._store.put_json(job_key(job.job_id), job.model_dump(mode="json"))
        self._jobs[job.job_id] = job
        return job

    async def get_job(self, job_id: str) -> Job:
        """Return one job, raising :class:`NotFound` if absent."""
        cached = self._jobs.get(job_id)
        if cached is not None:
            return cached
        payload = await self._store.get_json_or_none(job_key(job_id))
        if payload is None:
            raise NotFound(f"job {job_id} not found")
        job = Job.model_validate(payload)
        self._jobs[job_id] = job
        return job

    async def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        pages_done: int | None = None,
        error: str | None = None,
        worker: str | None = None,
    ) -> Job:
        """Patch a job record in place."""
        job = await self.get_job(job_id)
        if status is not None:
            job.status = status
        if pages_done is not None:
            job.progress.pages_done = pages_done
        if error is not None:
            job.error = error[:2000]
        if worker is not None:
            job.worker = worker
        return await self.put_job(job)

    async def list_queued_jobs(self, tenant_id: str | None = None) -> list[Job]:
        """List queued ingestion jobs, oldest first (used by the Colab worker)."""
        keys = await self._store.list_prefix("jobs")
        jobs: list[Job] = []
        for key in keys:
            payload = await self._store.get_json_or_none(key)
            if payload is None:
                continue
            try:
                job = Job.model_validate(payload)
            except Exception:  # noqa: BLE001
                continue
            if job.status != "queued" or job.kind != "ingest":
                continue
            if tenant_id and job.tenant_id != tenant_id:
                continue
            jobs.append(job)
        jobs.sort(key=lambda j: j.created_at)
        return jobs

    async def claim_job(self, job_id: str, worker: str) -> Job:
        """Mark a queued job as processing by ``worker``.

        Raises if the job was already claimed, so two notebook sessions cannot
        silently process the same document.
        """
        job = await self.get_job(job_id)
        if job.status != "queued":
            raise NotFound(f"job {job_id} is not claimable (status={job.status})")
        job.status = "processing"
        job.worker = worker
        return await self.put_job(job)

    def invalidate(self) -> None:
        """Drop the caches so the next read goes to the artifact store."""
        self._docs.clear()
        self._jobs.clear()
        self._loaded = False

    async def stats(self) -> dict[str, Any]:
        """Summary counts for health and metrics endpoints."""
        await self._ensure_loaded()
        by_status: dict[str, int] = {}
        for record in self._docs.values():
            by_status[record.status] = by_status.get(record.status, 0) + 1
        return {"documents": len(self._docs), "by_status": by_status, "loaded": self._loaded}
