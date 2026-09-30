def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_signup_and_login(client):
    r = client.post("/auth/signup", json={"username": "bob", "password": "secret123"})
    assert r.status_code == 201
    token = r.json()["token"]
    assert token

    # duplicate username -> conflict
    dup = client.post("/auth/signup", json={"username": "bob", "password": "secret123"})
    assert dup.status_code == 409

    # login with token
    login = client.post("/auth/login", json={"username": "bob", "password": "secret123"})
    assert login.status_code == 200
    assert login.json()["token"]


def test_login_bad_password(client):
    r = client.post("/auth/login", json={"username": "nobody", "password": "nope"})
    assert r.status_code == 401


def test_me_requires_auth(client):
    assert client.get("/auth/me").status_code == 401
