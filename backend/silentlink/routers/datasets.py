"""Dataset routes: list captured datasets and refresh from disk."""

from __future__ import annotations

from fastapi import APIRouter

from .. import deps
from ..schemas import DatasetResponse
from ..services.datasets import sync_datasets

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetResponse])
def list_datasets(db: deps.DbDep) -> list[DatasetResponse]:
    sync_datasets(db)
    return [DatasetResponse(**vars(d)) for d in db.datasets.values()]


@router.post("/refresh", response_model=list[DatasetResponse])
def refresh_datasets(db: deps.DbDep) -> list[DatasetResponse]:
    sync_datasets(db)
    return [DatasetResponse(**vars(d)) for d in db.datasets.values()]
