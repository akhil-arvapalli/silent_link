import numpy as np
import pytest
from fastapi.testclient import TestClient

from silentlink.db import MemoryDB
from silentlink.main import create_app


@pytest.fixture
def processed_root(tmp_path):
    root = tmp_path / "processed"
    for gloss, count in (("hello", 3), ("thanks", 2)):
        gloss_dir = root / gloss
        gloss_dir.mkdir(parents=True)
        for i in range(count):
            np.save(gloss_dir / f"seq_{i:03d}.npy", np.zeros((40, 42, 3), dtype=np.float32))
    return root


@pytest.fixture
def client(processed_root):
    db = MemoryDB()
    app = create_app(db, processed_root=processed_root)
    return TestClient(app)


@pytest.fixture
def authed_client(client):
    res = client.post(
        "/auth/signup", json={"username": "alice", "password": "secret123"}
    )
    token = res.json()["token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client
