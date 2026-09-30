"""Dataset read/write utilities.

Layout under data/processed:
    data/processed/<gloss>/<sequence_id>.npy   -> (T, 42, 3) float32 normalized
    data/processed/meta.json                    -> sequence -> gloss mapping
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEQUENCE_LENGTH = 40


def save_sequence(seq: np.ndarray, gloss: str, sequence_id: int, root: Path) -> Path:
    """Write one normalized sequence to data/processed/<gloss>/<id>.npy.

    Args:
        seq: (T, 42, 3) normalized input-ready array
        gloss: canonical gloss key (must exist in glosses.json)
        sequence_id: integer index
        root: data/processed directory
    Returns:
        Path to the written file
    """
    if seq.shape[0] != SEQUENCE_LENGTH:
        raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, got {seq.shape[0]}")
    gloss_dir = root / gloss
    gloss_dir.mkdir(parents=True, exist_ok=True)
    path = gloss_dir / f"{sequence_id}.npy"
    np.save(path, seq.astype(np.float32))
    return path


def load_sequence(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def write_meta(root: Path, records: list[dict]) -> None:
    (root / "meta.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def read_meta(root: Path) -> list[dict]:
    meta = root / "meta.json"
    if not meta.exists():
        return []
    return json.loads(meta.read_text(encoding="utf-8"))


def discover_sequences(root: Path) -> dict[str, list[Path]]:
    """Return {gloss: [sorted sequence paths]} from data/processed."""
    result: dict[str, list[Path]] = {}
    if not root.exists():
        return result
    for gloss_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        npy = sorted(p for p in gloss_dir.glob("*.npy"))
        if npy:
            result[gloss_dir.name] = npy
    return result


def build_dataset(root: Path, augment_fn=None, rng=None) -> tuple[np.ndarray, list[str]]:
    """Load all sequences into (X, gloss_labels).

    X shape: (N, T, 42, 3). Optionally apply an augmentation callable per
    sequence (e.g. for train-time on-the-fly augmentation).
    """
    seqs: list[np.ndarray] = []
    labels: list[str] = []
    for gloss, paths in discover_sequences(root).items():
        for p in paths:
            seq = load_sequence(p)
            if augment_fn is not None:
                seq = augment_fn(seq)
            seqs.append(seq)
            labels.append(gloss)
    if not seqs:
        return np.empty((0, SEQUENCE_LENGTH, 42, 3), dtype=np.float32), []
    return np.stack(seqs).astype(np.float32), labels
