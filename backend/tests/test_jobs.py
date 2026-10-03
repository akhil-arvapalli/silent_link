"""Training-job lifecycle.

`TrainingQueue` and `run_job` never executed under the old suite: the only
`/jobs` assertion checked 401 *before* the handler ran. These tests inject a fake
runner through the `db._train_queue` seam that `routers.jobs._queue` honours, so
no subprocess or GPU work is needed.
"""

import threading

import pytest

from silentlink.db import TrainJob
from silentlink.services.training import TrainingQueue


@pytest.fixture
def db(client):
    return client.app.state.db


def _install_queue(db, runner):
    queue = TrainingQueue(db, runner=runner)
    db._train_queue = queue
    return queue


def _await_terminal(client, job_id, timeout=5.0):
    """Poll until the job leaves queued/running, mirroring what a client does."""
    deadline = threading.Event()
    for _ in range(int(timeout * 100)):
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        deadline.wait(0.01)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def test_create_job_requires_a_known_dataset(authed_client):
    assert authed_client.post("/jobs", json={"dataset_id": "nope"}).status_code == 404


def test_job_succeeds_and_records_metrics(authed_client, db):
    authed_client.get("/datasets")  # registers `default`
    _install_queue(db, lambda job: {"best_val_acc": 0.97, "best_epoch": 3})

    created = authed_client.post("/jobs", json={"dataset_id": "default", "epochs": 1})
    assert created.status_code == 202

    body = _await_terminal(authed_client, created.json()["id"])
    assert body["status"] == "succeeded"
    assert body["result"] == {"best_val_acc": 0.97, "best_epoch": 3}
    assert body["error"] is None


def test_job_failure_is_recorded_not_raised(authed_client, db):
    authed_client.get("/datasets")

    def boom(job):
        raise RuntimeError("training exploded")

    _install_queue(db, boom)
    created = authed_client.post("/jobs", json={"dataset_id": "default"})
    body = _await_terminal(authed_client, created.json()["id"])
    assert body["status"] == "failed"
    assert "training exploded" in body["error"]


def test_queue_runs_one_job_at_a_time(authed_client, db):
    """The worker claims to serialise; verify it actually does.

    Each job is a subprocess of the real trainer, so concurrent jobs would spawn
    concurrent trainers. A semaphore in `TrainingQueue.start` is what makes the
    docstring true.
    """
    authed_client.get("/datasets")
    concurrent = []
    active = {"now": 0, "peak": 0}
    gate = threading.Event()

    def runner(job):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        concurrent.append(job.id)
        gate.wait(2.0)
        active["now"] -= 1
        return {"best_val_acc": 1.0}

    _install_queue(db, runner)

    first = authed_client.post("/jobs", json={"dataset_id": "default"}).json()["id"]
    second = authed_client.post("/jobs", json={"dataset_id": "default"}).json()["id"]

    for _ in range(200):
        if len(concurrent) >= 1 and active["now"] >= 1:
            break
        gate.wait(0.01)
    # Give the second job a chance to (incorrectly) start concurrently.
    gate.wait(0.2)
    assert active["peak"] == 1, "two training jobs ran at once"
    gate.set()

    assert _await_terminal(authed_client, first)["status"] == "succeeded"
    assert _await_terminal(authed_client, second)["status"] == "succeeded"
    assert sorted(concurrent) == sorted([first, second])


def test_list_jobs_is_scoped_to_the_owner(client):
    def signup(name):
        res = client.post("/auth/signup", json={"username": name, "password": "secret123"})
        client.headers["Authorization"] = f"Bearer {res.json()['token']}"

    signup("alice")
    alice_job = client.post("/jobs", json={"dataset_id": "missing"}).status_code
    signup("bob")
    bob_jobs = client.get("/jobs").json()

    assert alice_job == 404
    assert bob_jobs == []


def test_get_job_hides_other_users_jobs(client):
    """A job id must not be a usable capability across accounts."""

    def signup(name: str) -> str:
        res = client.post("/auth/signup", json={"username": name, "password": "secret123"})
        return res.json()["token"]

    client.headers["Authorization"] = f"Bearer {signup('alice')}"
    db = client.app.state.db
    job = TrainJob(
        id="job_secret", owner_id="u_alice", dataset_id="d", model_type="stgcn", status="queued"
    )
    db.jobs[job.id] = job

    client.headers["Authorization"] = f"Bearer {signup('bob')}"
    assert client.get("/jobs/job_secret").status_code == 404
    assert client.get("/jobs").json() == []


def test_queue_handles_a_runner_returning_nothing(db):
    """A runner returning None must not corrupt the record."""
    job = TrainJob(id="job_1", owner_id="u", dataset_id="d", model_type="stgcn", status="queued")
    db.jobs[job.id] = job
    tick = threading.Event()

    queue = TrainingQueue(db, runner=lambda j: None)
    queue.start(job)
    for _ in range(200):
        if db.jobs[job.id].status in {"succeeded", "failed"}:
            break
        tick.wait(0.01)
    assert db.jobs[job.id].status == "succeeded"
    assert db.jobs[job.id].result is None
