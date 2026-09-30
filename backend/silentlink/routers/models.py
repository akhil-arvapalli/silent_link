"""Model registry routes: register and list trained models."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from .. import deps
from ..db import ModelRecord
from ..schemas import ModelRegisterRequest, ModelResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def register_model(
    body: ModelRegisterRequest, db: deps.DbDep, user: deps.UserDep
) -> ModelResponse:
    if body.job_id and body.job_id not in db.jobs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    record = ModelRecord(
        id=db.next_id("model"),
        owner_id=user.id,
        name=body.name,
        model_type=body.model_type,
        job_id=body.job_id,
        accuracy=body.accuracy,
        size_bytes=body.size_bytes,
    )
    db.models[record.id] = record
    return ModelResponse(**vars(record))


@router.get("", response_model=list[ModelResponse])
def list_models(db: deps.DbDep, user: deps.UserDep) -> list[ModelResponse]:
    records = [m for m in db.models.values() if m.owner_id == user.id]
    records.sort(key=lambda m: m.created_at, reverse=True)
    return [ModelResponse(**vars(m)) for m in records]
