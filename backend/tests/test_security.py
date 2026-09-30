from silentlink.security import (
    hash_password,
    sign_token,
    verify_password,
    verify_token,
)


def test_password_roundtrip():
    h = hash_password("hunter2")
    assert h.startswith("pbkdf2$")
    assert verify_password("hunter2", h)
    assert not verify_password("wrong", h)


def test_hash_is_salted():
    a = hash_password("pw")
    b = hash_password("pw")
    assert a != b  # different salts even for same password


def test_token_roundtrip():
    token = sign_token("user_1", "secret")
    payload = verify_token(token, "secret")
    assert payload is not None
    assert payload["sub"] == "user_1"


def test_token_wrong_secret_rejected():
    token = sign_token("user_1", "secret-a")
    assert verify_token(token, "secret-b") is None


def test_token_tampered_rejected():
    token = sign_token("user_1", "secret")
    tampered = token[:-2] + ("AA" if not token.endswith("AA") else "BB")
    assert verify_token(tampered, "secret") is None
