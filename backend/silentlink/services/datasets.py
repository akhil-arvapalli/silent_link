"""Dataset management: discover on-disk capture data and register it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import REPO_ROOT
from ..db import Dataset, MemoryDB

DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"


def scan_processed(root: Path = DEFAULT_PROCESSED) -> list[dict]:
    """Return per-gloss sequence counts under data/processed/<gloss>/<id>.npy."""
    glosses: list[tuple[str, int]] = []
    if not root.exists():
        return []
    for gloss_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        npy = [p for p in gloss_dir.glob("*.npy")]
        if npy:
            glosses.append((gloss_dir.name, len(npy)))
    return [{"gloss": g, "count": c} for g, c in glosses]


def frames_for(root: Path = DEFAULT_PROCESSED) -> int:
    """Infer the frame count from the first available sequence."""
    if not root.exists():
        return 0
    for gloss_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in gloss_dir.glob("*.npy"):
            arr = np.load(p, allow_pickle=False)
            return int(arr.shape[0])
    return 0


def sync_datasets(db: MemoryDB) -> None:
    """Refresh DB dataset records from the on-disk processed folder."""
    rows = scan_processed()
    db.datasets.clear()
    total = sum(r["count"] for r in rows)
    db.datasets["default"] = Dataset(
        id="default",
        name="processed-capture",
        glosses=[r["gloss"] for r in rows],
        sequence_count=total,
        frames=frames_for(),
    )
