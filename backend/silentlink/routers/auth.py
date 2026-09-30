"""Auth routes: signup, login."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from .. import deps
from ..db import User
from ..schemas import LoginRequest, SignupRequest, TokenResponse, UserResponse
from ..security import hash_password, sign_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, db: deps.DbDep) -> TokenResponse:
    if db.get_user_by_username(body.username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")
    user = User(
        id=db.next_id("user"),
        username=body.username,
        password_hash=hash_password(body.password),
    )
    db.upsert_user(user)
    token = sign_token(user.id, deps.SECRET)
    return TokenResponse(token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: deps.DbDep) -> TokenResponse:
    user = db.get_user_by_username(body.username)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = sign_token(user.id, deps.SECRET)
    return TokenResponse(token=token)


@router.get("/me", response_model=UserResponse)
def me(user: deps.UserDep) -> UserResponse:
    return UserResponse(id=user.id, username=user.username)
