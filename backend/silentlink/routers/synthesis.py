"""Synthesis routes: run text->sign synthesis and query past jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from .. import deps
from ..db import SynthesisJob
from ..schemas import SynthesisCreate, SynthesisResponse
from ..services.synthesis import synthesize

router = APIRouter(prefix="/synthesis", tags=["synthesis"])


@router.post("", response_model=SynthesisResponse)
def create_synthesis(
    body: SynthesisCreate, db: deps.DbDep, user: deps.UserDep
) -> SynthesisResponse:
    """Synthesize immediately and return the terminal record.

    This used to advertise 202/``queued`` while running the work inline on the
    request thread and returning ``succeeded``, so the status code, the status
    field and the client contract all disagreed. The on-device path consumes
    ``glosses`` straight off this response to cross-check its own synthesis, so
    the work stays synchronous and the response is honestly 200. `GET
    /synthesis/{job_id}` remains available for replaying a stored result.
    """
    job = SynthesisJob(
        id=db.next_id("syn"),
        owner_id=user.id,
        text=body.text,
        glosses=[],
        status="running",
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
