"""FastAPI dependencies: shared DB + authenticated user."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from .db import MemoryDB, User
from .security import verify_token

SECRET = os.environ.get("SILENTLINK_SECRET", "dev-secret-change-me")


def get_db(request: Request) -> MemoryDB:
    db = getattr(request.app.state, "db", None)
    if db is None:
        db = MemoryDB()
        request.app.state.db = db
    return db


def _extract_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return auth[len("bearer ") :].strip()


def get_current_user(request: Request) -> User:
    token = _extract_token(request)
    payload = verify_token(token, SECRET)
    if payload is None or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    db = get_db(request)
    user = db.get_user(payload["sub"])
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


DbDep = Annotated[MemoryDB, Depends(get_db)]
UserDep = Annotated[User, Depends(get_current_user)]
