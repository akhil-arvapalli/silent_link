

def test_datasets_listed(authed_client):
    r = authed_client.get("/datasets")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    # The repo has processed capture data (synthetic), so expect at least one dataset.
    assert len(data) >= 1
    assert data[0]["glosses"]


def test_models_register_and_list(authed_client):
    r = authed_client.post(
        "/models",
        json={
            "name": "stgcn-v1",
            "model_type": "stgcn",
            "accuracy": 0.97,
            "size_bytes": 1500000,
        },
    )
    assert r.status_code == 201
    model_id = r.json()["id"]

    listed = authed_client.get("/models").json()
    assert any(m["id"] == model_id for m in listed)


def test_jobs_require_auth(client):
    assert client.post("/jobs", json={}).status_code == 401


def test_synthesis_maps_text_to_glosses(authed_client):
    r = authed_client.post("/synthesis", json={"text": "good morning"})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "succeeded"
    assert "good_morning" in body["glosses"]
    assert body["result"]["frames"] > 0
