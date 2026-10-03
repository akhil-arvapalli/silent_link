"""FastAPI application factory for Silent Link backend.

Provides a create_app(settings) seam so tests can build an isolated app with a
fresh in-memory DB, while the default entrypoint runs the real service.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from .db import MemoryDB
from .routers import auth, datasets, jobs, models, synthesis


def create_app(
    db: MemoryDB | None = None, processed_root: Path | None = None
) -> FastAPI:
    app = FastAPI(title="Silent Link Backend", version="0.1.0")
    app.state.db = db or MemoryDB()
    app.state.processed_root = processed_root
    app.include_router(auth.router)
    app.include_router(datasets.router)
    app.include_router(jobs.router)
    app.include_router(models.router)
    app.include_router(synthesis.router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "glosses": len(app.state.db.datasets)}

    return app


app = create_app()
