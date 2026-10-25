"""Job status endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from emrag.api.deps import PrincipalDep, StateDep
from emrag.api.schemas.documents import JobResponse
from emrag.core.errors import Forbidden

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse, summary="Get job status")
async def get_job(job_id: str, state: StateDep, principal: PrincipalDep) -> JobResponse:
    """Return one ingestion or deletion job, scoped to the caller's tenant."""
    job = await state.registry.get_job(job_id)
    if job.tenant_id != principal.tenant_id:
        raise Forbidden("job belongs to another tenant")
    return JobResponse(
        job_id=job.job_id,
        kind=job.kind,
        document_id=job.doc_id,
        status=job.status,
        progress=job.progress,
        worker=job.worker,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
