"""Pydantic request/response schemas for the Silent Link backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SignupRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    username: str


class DatasetResponse(BaseModel):
    id: str
    name: str
    glosses: list[str]
    sequence_count: int
    frames: int
    created_at: str


class TrainJobCreate(BaseModel):
    dataset_id: str
    model_type: str = Field(default="stgcn", pattern="^(stgcn|baseline)$")
    epochs: int = Field(default=60, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=1e-3, gt=0)


class TrainJobResponse(BaseModel):
    id: str
    dataset_id: str
    model_type: str
    status: str
    params: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    created_at: str


class ModelRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    model_type: str = Field(default="stgcn", pattern="^(stgcn|baseline)$")
    job_id: str | None = None
    accuracy: float | None = Field(default=None, ge=0, le=1)
    size_bytes: int | None = Field(default=None, ge=0)


class ModelResponse(BaseModel):
    id: str
    name: str
    model_type: str
    job_id: str | None
    accuracy: float | None
    size_bytes: int | None
    created_at: str


class SynthesisCreate(BaseModel):
    text: str = Field(min_length=1, max_length=512)


class SynthesisResponse(BaseModel):
    id: str
    text: str
    glosses: list[str]
    status: str
    result: dict[str, Any] | None
    error: str | None
    created_at: str
