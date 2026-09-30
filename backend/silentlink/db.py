"""In-memory persistence for the backend.

A deliberately simple repository layer so the service is testable with zero
external services. Each store owns its id generation and mutation. Swap these
for real DB-backed repositories without changing the routers.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class User:
    id: str
    username: str
    password_hash: str
    created_at: str = field(default_factory=_utcnow)


@dataclass
class Dataset:
    id: str
    name: str
    glosses: list[str]
    sequence_count: int
    frames: int
    created_at: str = field(default_factory=_utcnow)


@dataclass
class TrainJob:
    id: str
    owner_id: str
    dataset_id: str
    model_type: str
    status: str  # queued | running | succeeded | failed
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utcnow)


@dataclass
class ModelRecord:
    id: str
    owner_id: str
    name: str
    model_type: str
    job_id: str | None
    accuracy: float | None
    size_bytes: int | None
    created_at: str = field(default_factory=_utcnow)


@dataclass
class SynthesisJob:
    id: str
    owner_id: str
    text: str
    glosses: list[str]
    status: str  # queued | succeeded | failed
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utcnow)


class MemoryDB:
    """Thread-safe collection of in-memory stores."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self.users: dict[str, User] = {}
        self.users_by_username: dict[str, str] = {}
        self.datasets: dict[str, Dataset] = {}
        self.jobs: dict[str, TrainJob] = {}
        self.models: dict[str, ModelRecord] = {}
        self.synthesis: dict[str, SynthesisJob] = {}

    def next_id(self, prefix: str) -> str:
        with self._lock:
            return f"{prefix}_{next(self._ids)}"

    def upsert_user(self, user: User) -> None:
        with self._lock:
            self.users[user.id] = user
            self.users_by_username[user.username] = user.id

    def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        uid = self.users_by_username.get(username)
        return self.users.get(uid) if uid else None
