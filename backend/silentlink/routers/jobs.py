"""Train-job routes: create, list, and inspect training jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from .. import deps
from ..db import MemoryDB, TrainJob
from ..schemas import TrainJobCreate, TrainJobResponse
from ..services.training import TrainingQueue


def _queue(db: MemoryDB) -> TrainingQueue:
    queue = getattr(db, "_train_queue", None)
    if queue is None:
        queue = TrainingQueue(db)
        db._train_queue = queue
    return queue


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=TrainJobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(body: TrainJobCreate, db: deps.DbDep, user: deps.UserDep) -> TrainJobResponse:
    if body.dataset_id not in db.datasets:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    job = TrainJob(
        id=db.next_id("job"),
        owner_id=user.id,
        dataset_id=body.dataset_id,
        model_type=body.model_type,
        status="queued",
        params=body.model_dump(exclude={"dataset_id", "model_type"}),
    )
    db.jobs[job.id] = job
    _queue(db).start(job)
    return TrainJobResponse(**vars(job))


@router.get("", response_model=list[TrainJobResponse])
def list_jobs(db: deps.DbDep, user: deps.UserDep) -> list[TrainJobResponse]:
    jobs = [j for j in db.jobs.values() if j.owner_id == user.id]
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [TrainJobResponse(**vars(j)) for j in jobs]


@router.get("/{job_id}", response_model=TrainJobResponse)
def get_job(job_id: str, db: deps.DbDep, user: deps.UserDep) -> TrainJobResponse:
    job = db.jobs.get(job_id)
    if job is None or job.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return TrainJobResponse(**vars(job))
