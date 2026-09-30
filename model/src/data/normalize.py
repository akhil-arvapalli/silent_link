"""Landmark normalization utilities (numpy-only).

Converts raw MediaPipe hand landmarks into normalized, hand-relative
coordinates matching the training contract in model/configs/glosses.json:
input shape (T, V=42, C=3) where V = 2 hands * 21 landmarks.

All functions are pure numpy so they are unit-testable without mediapipe/cv2.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NUM_LANDMARKS = 21
NUM_HANDS = 2
NUM_CHANNELS = 3
V = NUM_HANDS * NUM_LANDMARKS  # 42


def landmark_to_array(landmarks: Sequence[object]) -> np.ndarray:
    """Convert a sequence of MediaPipe NormalizedLandmark objects to (21, 3)."""
    out = np.zeros((NUM_LANDMARKS, NUM_CHANNELS), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        out[i] = (lm.x, lm.y, lm.z)
    return out


def pack_hands(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray:
    """Pack left/right hand arrays into one (42, 3) frame.

    Args:
        left: (21, 3) or None
        right: (21, 3) or None
    Returns:
        (42, 3) where [0:21]=left, [21:42]=right. Missing hands are zeros.
    """
    frame = np.zeros((V, NUM_CHANNELS), dtype=np.float32)
    if left is not None:
        frame[0:NUM_LANDMARKS] = left
    if right is not None:
        frame[NUM_LANDMARKS:] = right
    return frame


def normalize_hand_relative(sequence: np.ndarray) -> np.ndarray:
    """Normalize each hand to wrist-relative, scale-invariant coordinates.

    Args:
        sequence: (T, 42, 3) or (42, 3)
    Returns:
        Same shape, normalized. Wrist (landmark 0) becomes origin for each
        hand; coordinates scaled by the mean of hand-bone lengths to be
        scale invariant.
    """
    was_3d = sequence.ndim == 2
    if was_3d:
        sequence = sequence[np.newaxis, ...]

    out = sequence.copy()
    T = sequence.shape[0]
    for t in range(T):
        for hand in (0, 1):
            idx = slice(hand * NUM_LANDMARKS, (hand + 1) * NUM_LANDMARKS)
            hand_pts = out[t, idx]
            wrist = hand_pts[0].copy()
            hand_pts -= wrist  # wrist-relative
            bone_len = np.mean(np.linalg.norm(hand_pts[1:] - hand_pts[:-1], axis=1))
            if bone_len > 1e-6:
                hand_pts /= bone_len
    return out[0] if was_3d else out


def sequence_to_input(sequence: np.ndarray) -> np.ndarray:
    """Validate and shape a captured sequence into training input format.

    Args:
        sequence: (T, 42, 3) raw packed frames
    Returns:
        (T, 42, 3) float32 normalized.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[1] != V or seq.shape[2] != NUM_CHANNELS:
        raise ValueError(f"Expected (T, {V}, {NUM_CHANNELS}), got {seq.shape}")
    return normalize_hand_relative(seq)
