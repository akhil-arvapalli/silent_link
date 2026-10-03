"""FastAPI dependencies: shared DB + authenticated user."""

from __future__ import annotations

import logging
import os
import secrets
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request, status

from .config import REPO_ROOT
from .db import MemoryDB, User
from .security import verify_token

log = logging.getLogger(__name__)

load_dotenv(REPO_ROOT / ".env")

# Where a generated key is cached so that every worker and every restart agrees.
# `backend/.silentlink_secret` is gitignored.
SECRET_FILE = REPO_ROOT / "backend" / ".silentlink_secret"


def _resolve_secret() -> str:
    """Return the token-signing key, preferring the environment.

    Falling back to an in-memory random key silently breaks multi-worker
    deployments: `uvicorn --workers N` imports this module once per worker, each
    generating a *different* key, so a token minted by one worker fails
    verification in the next and the client sees intermittent 401s. Persisting
    the generated key fixes both that and "tokens die on every restart".
    """
    configured = os.environ.get("SILENTLINK_SECRET")
    if configured:
        return configured
    try:
        cached = SECRET_FILE.read_text(encoding="utf-8").strip()
        if cached:
            return cached
    except OSError:
        pass
    generated = secrets.token_hex(32)
    try:
        SECRET_FILE.write_text(generated, encoding="utf-8")
        log.warning(
            "SILENTLINK_SECRET is unset; generated a key at %s. "
            "Set SILENTLINK_SECRET in .env to control it.",
            SECRET_FILE,
        )
    except OSError:
        log.warning(
            "SILENTLINK_SECRET is unset and %s is not writable; using an "
            "ephemeral key. Tokens will break between workers and on restart.",
            SECRET_FILE,
        )
    return generated


SECRET = _resolve_secret()


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
