"""Dataset routes: list captured datasets and refresh from disk."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import deps
from ..schemas import DatasetResponse
from ..services.datasets import sync_datasets, upsert_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _root(request: Request):
    return getattr(request.app.state, "processed_root", None)


@router.get("", response_model=list[DatasetResponse])
def list_datasets(request: Request, db: deps.DbDep) -> list[DatasetResponse]:
    # Upsert, not sync: a GET must not destroy datasets registered elsewhere.
    upsert_dataset(db, _root(request))
    return [DatasetResponse(**vars(d)) for d in db.datasets.values()]


@router.post("/refresh", response_model=list[DatasetResponse])
def refresh_datasets(request: Request, db: deps.DbDep) -> list[DatasetResponse]:
    sync_datasets(db, _root(request))
    return [DatasetResponse(**vars(d)) for d in db.datasets.values()]
