"""Auth primitives: password hashing + HMAC-signed bearer tokens.

Deliberately dependency-light: PBKDF2 for password hashing and an HMAC-SHA256
signed token (user id + expiry) for sessions, so the service runs with zero
extra runtime deps beyond the stdlib.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

_ITERATIONS = 120_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return a salted PBKDF2 hash string `pbkdf2$iterations$salt$hash`."""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2${_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt_b64, hash_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(dk, expected)
    except (ValueError, TypeError):
        return False


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign_token(user_id: str, secret: str, ttl_seconds: int = 60 * 60 * 24) -> str:
    """Issue an HMAC-SHA256 signed token: `payload.signature`."""
    payload = {"sub": user_id, "exp": int(time.time()) + ttl_seconds}
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64url(sig)}"


def verify_token(token: str, secret: str) -> dict | None:
    """Validate a token and return its payload, or None if invalid/expired."""
    try:
        body, sig_b64 = token.split(".")
        expected = hmac.new(
            secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256
        ).digest()
        actual = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
        if not hmac.compare_digest(actual, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
