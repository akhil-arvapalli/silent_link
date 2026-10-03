"""Dataset management: discover on-disk capture data and register it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import REPO_ROOT
from ..db import Dataset, MemoryDB

DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"

# root -> (stamp, snapshot). The scan walks every sequence file and reads one
# array header, which is far too much to repeat on every request.
_CACHE: dict[str, tuple[tuple, Dataset]] = {}


def _stamp(root: Path) -> tuple:
    """A cheap fingerprint of the processed tree, for cache invalidation."""
    if not root.exists():
        return ("missing",)
    parts: list[tuple] = []
    for gloss_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in gloss_dir.glob("*.npy")
        )
        parts.append((gloss_dir.name, tuple(files)))
    return tuple(parts)


def scan_processed(root: Path | None = None) -> list[dict]:
    """Return per-gloss sequence counts under data/processed/<gloss>/<id>.npy."""
    root = root or DEFAULT_PROCESSED
    glosses: list[tuple[str, int]] = []
    if not root.exists():
        return []
    for gloss_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        npy = [p for p in gloss_dir.glob("*.npy")]
        if npy:
            glosses.append((gloss_dir.name, len(npy)))
    return [{"gloss": g, "count": c} for g, c in glosses]


def frames_for(root: Path | None = None) -> int:
    """Infer the frame count from the first available sequence."""
    root = root or DEFAULT_PROCESSED
    if not root.exists():
        return 0
    for gloss_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in gloss_dir.glob("*.npy"):
            # mmap so the frame count is read from the header without paging
            # the whole sequence into memory.
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            return int(arr.shape[0])
    return 0


def snapshot(root: Path | None = None) -> Dataset:
    """Build the dataset record for `root`, reusing the cached scan if valid."""
    root = root or DEFAULT_PROCESSED
    key = str(root)
    stamp = _stamp(root)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    rows = scan_processed(root)
    record = Dataset(
        id="default",
        name="processed-capture",
        glosses=[r["gloss"] for r in rows],
        sequence_count=sum(r["count"] for r in rows),
        frames=frames_for(root),
    )
    _CACHE[key] = (stamp, record)
    return record


def upsert_dataset(db: MemoryDB, root: Path | None = None) -> Dataset:
    """Refresh the on-disk dataset record in place, preserving other datasets.

    Idempotent, so it is safe on a read route.
    """
    record = snapshot(root)
    db.datasets[record.id] = record
    return record


def sync_datasets(db: MemoryDB, root: Path | None = None) -> None:
    """Rebuild DB dataset records from the on-disk processed folder.

    Mutating, so it belongs on the explicit refresh route only. A previous
    version called this from `GET /datasets`, which made a read wipe every
    registered dataset.
    """
    record = snapshot(root)
    db.datasets.clear()
    db.datasets[record.id] = record
