"""Train-job orchestration.

A queued job runs the repo's real training pipeline (scripts/train_model.py)
in a background subprocess and records the outcome. `run_job` is the seam used
by the worker; tests inject a fake runner to avoid GPU/CPU training.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from ..config import REPO_ROOT
from ..db import MemoryDB, TrainJob

TrainRunner = Callable[[TrainJob], dict]


def run_job(job: TrainJob, out_dir: Path) -> dict:
    """Run real training for `job` into `out_dir`, returning result metrics."""
    subprocess.run(
        [
            "python",
            str(REPO_ROOT / "scripts" / "train_model.py"),
            "--model",
            job.model_type,
            "--data",
            str(REPO_ROOT / "data" / "processed"),
            "--out",
            str(out_dir),
            "--epochs",
            str(job.params.get("epochs", 60)),
            "--batch-size",
            str(job.params.get("batch_size", 32)),
            "--lr",
            str(job.params.get("lr", 1e-3)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "best_val_acc": metrics.get("best_val_acc"),
            "best_epoch": metrics.get("best_epoch"),
        }
    return {"detail": "training finished but produced no metrics.json"}


def _default_runner() -> TrainRunner:
    def runner(job: TrainJob) -> dict:
        from .training import run_job as _run

        out_dir = REPO_ROOT / "model" / "runs" / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        return _run(job, out_dir)

    return runner


class TrainingQueue:
    """A background worker that runs at most one training job at a time.

    Each job is a `subprocess.run` of the real pipeline, so N concurrent jobs
    would spawn N CPU/GPU trainers. Jobs beyond the first block on the
    semaphore and stay honestly `queued` until a slot frees.
    """

    def __init__(self, db: MemoryDB, runner: TrainRunner | None = None) -> None:
        self.db = db
        self.runner = runner or _default_runner()
        self._slot = threading.Semaphore(1)

    def start(self, job: TrainJob) -> None:
        def work() -> None:
            self._slot.acquire()
            try:
                self.db.jobs[job.id].status = "running"
                result = self.runner(job)
                record = self.db.jobs[job.id]
                record.status = "succeeded"
                record.result = result
            except Exception as exc:  # noqa: BLE001 - report any failure to the caller
                record = self.db.jobs[job.id]
                record.status = "failed"
                record.error = str(exc)
            finally:
                self._slot.release()

        threading.Thread(target=work, daemon=True).start()
