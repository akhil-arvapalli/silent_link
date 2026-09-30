"""Synthesis routes: enqueue and query text->sign synthesis jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from .. import deps
from ..db import SynthesisJob
from ..schemas import SynthesisCreate, SynthesisResponse
from ..services.synthesis import synthesize

router = APIRouter(prefix="/synthesis", tags=["synthesis"])


@router.post("", response_model=SynthesisResponse, status_code=status.HTTP_202_ACCEPTED)
def create_synthesis(
    body: SynthesisCreate, db: deps.DbDep, user: deps.UserDep
) -> SynthesisResponse:
    job = SynthesisJob(
        id=db.next_id("syn"),
        owner_id=user.id,
        text=body.text,
        glosses=[],
        status="queued",
    )
    db.synthesis[job.id] = job
    try:
        result = synthesize(body.text)
        record = db.synthesis[job.id]
        record.status = "succeeded"
        record.glosses = result["glosses"]
        record.result = result
    except Exception as exc:  # noqa: BLE001 - surface any synthesis failure
        record = db.synthesis[job.id]
        record.status = "failed"
        record.error = str(exc)
    return SynthesisResponse(**vars(db.synthesis[job.id]))


@router.get("/{job_id}", response_model=SynthesisResponse)
def get_synthesis(job_id: str, db: deps.DbDep, user: deps.UserDep) -> SynthesisResponse:
    job = db.synthesis.get(job_id)
    if job is None or job.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return SynthesisResponse(**vars(job))
