import pytest
from fastapi.testclient import TestClient

from silentlink.db import MemoryDB
from silentlink.main import create_app


@pytest.fixture
def client():
    db = MemoryDB()
    app = create_app(db)
    return TestClient(app)


@pytest.fixture
def authed_client(client):
    res = client.post(
        "/auth/signup", json={"username": "alice", "password": "secret123"}
    )
    token = res.json()["token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client
