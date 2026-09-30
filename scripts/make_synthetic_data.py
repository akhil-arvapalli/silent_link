#!/usr/bin/env python
"""Generate synthetic landmark sequences for pipeline smoke-testing.

The data/processed folder is empty until real webcam capture fills it. To
validate the training pipeline end-to-end (train/val split, early stopping,
ONNX export) without real data, this script fabricates (T, 42, 3) sequences
with a per-gloss distinguishing temporal pattern.

Usage:
    python scripts/make_synthetic_data.py --per-gloss 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model" / "src"))

from data.dataset import SEQUENCE_LENGTH, save_sequence, write_meta  # noqa: E402
from data.normalize import NUM_LANDMARKS  # noqa: E402


def make_sequence(seed: int, freq: float, T: int, V: int, C: int) -> np.ndarray:
    """Create a smooth, class-discriminative landmark sequence.

    Each joint follows a sinusoid whose phase is seeded per (joint, class)
    so classes are separable, plus mild noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, T)[:, None, None]
    phase = rng.uniform(0, 2 * np.pi, size=(V, C))
    base = np.sin(freq * t + phase) * 0.5
    noise = rng.normal(0, 0.02, size=(T, V, C))
    return (base + noise).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic landmark sequences")
    parser.add_argument("--per-gloss", type=int, default=40, help="Sequences per gloss (default 40)")
    parser.add_argument(
        "--glosses-json",
        type=Path,
        default=REPO_ROOT / "model" / "configs" / "glosses.json",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed")
    args = parser.parse_args()

    glosses = sorted(json.loads(args.glosses_json.read_text(encoding="utf-8"))["glosses"].keys())
    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i, gloss in enumerate(glosses):
        freq = 0.5 + 0.3 * i  # per-gloss frequency signature
        for seq_id in range(args.per_gloss):
            seq = make_sequence(seed=1000 * i + seq_id, freq=freq, T=SEQUENCE_LENGTH, V=42, C=3)
            save_sequence(seq, gloss, seq_id, args.out)
            records.append({"gloss": gloss, "id": seq_id})
    write_meta(args.out, records)
    print(f"Wrote {len(records)} synthetic sequences for {len(glosses)} glosses to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
