"""Landmark-space data augmentation (numpy-only).

Augmentations operate directly on normalized (T, V=42, C=3) landmark
sequences. Because coordinates are already wrist-relative + scale-normalized,
we apply small perturbations that preserve hand shape semantics:

- random 3D rotation (small)
- random per-hand scale jitter
- left/right hand mirror (swap hands) for handedness generalization
- temporal jitter / frame dropout
- noise injection
"""

from __future__ import annotations

import numpy as np

from .normalize import NUM_LANDMARKS


def _rotation_matrix(angles_deg: np.ndarray) -> np.ndarray:
    ax, ay, az = np.radians(angles_deg)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotate_sequence(
    seq: np.ndarray,
    max_deg: float = 12.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    angles = rng.uniform(-max_deg, max_deg, size=3)
    R = _rotation_matrix(angles)
    out = seq.copy()
    for t in range(out.shape[0]):
        out[t] = out[t] @ R.T
    return out


def scale_hand(
    seq: np.ndarray,
    scale_std: float = 0.08,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = seq.copy()
    for hand in (0, 1):
        idx = slice(hand * NUM_LANDMARKS, (hand + 1) * NUM_LANDMARKS)
        s = 1.0 + rng.normal(0.0, scale_std)
        out[:, idx] *= s
    return out


def mirror_hands(seq: np.ndarray) -> np.ndarray:
    """Swap left/right hands (for handedness generalization)."""
    out = seq.copy()
    left = out[:, :NUM_LANDMARKS].copy()
    out[:, :NUM_LANDMARKS] = out[:, NUM_LANDMARKS:]
    out[:, NUM_LANDMARKS:] = left
    return out


def temporal_jitter(
    seq: np.ndarray,
    drop_prob: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    original_len = seq.shape[0]
    keep = rng.random(original_len) > drop_prob
    if not keep.any():
        keep[0] = True
    seq = seq[keep]
    # resample back to original length with linear interpolation per (joint, channel)
    n = seq.shape[0]
    idx_old = np.linspace(0, original_len - 1, n)
    idx_new = np.linspace(0, original_len - 1, original_len)
    out = np.empty((original_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            out[:, j, c] = np.interp(idx_new, idx_old, seq[:, j, c])
    return out


def add_noise(
    seq: np.ndarray,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return seq + rng.normal(0.0, noise_std, size=seq.shape).astype(np.float32)


def augment(
    seq: np.ndarray,
    rng: np.random.Generator | None = None,
    p_mirror: float = 0.5,
) -> np.ndarray:
    """Apply a randomized composition of augmentations."""
    rng = rng or np.random.default_rng()
    out = seq.copy()
    out = rotate_sequence(out, rng=rng)
    out = scale_hand(out, rng=rng)
    out = temporal_jitter(out, rng=rng)
    out = add_noise(out, rng=rng)
    if rng.random() < p_mirror:
        out = mirror_hands(out)
    return out
