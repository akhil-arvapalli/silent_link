"""Per-gloss landmark motion library (Phase 2).

A gloss's *motion* is a canonical (T, 42, 3) landmark trajectory that
MediaPipe's Hand Landmarker would observe while the sign is performed.
This module:

  - defines the MotionLibrary container (gloss -> motion template + duration),
  - generates deterministic *synthetic* motion templates so the synthesis
    pipeline can be built and tested before real webcam capture exists.

Real motion capture (scripts/capture_data.py) will populate the library with
authentic trajectories; the stitching pipeline consumes the same interface.

The synthetic generator models a hand as finger chains that curl/uncurl in a
coordinated way over a whole-hand sway envelope, so the Skia skeleton reads as
a gesture rather than independent per-landmark noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.normalize import NUM_HANDS, NUM_LANDMARKS, normalize_hand_relative

_FINGER_TIPS = {4, 8, 12, 16, 20}
_T = 40
_V = NUM_HANDS * NUM_LANDMARKS
_C = 3


def make_motion_template(seed: int, T: int = _T, V: int = _V, C: int = _C) -> np.ndarray:
    """Create a deterministic synthetic (T, V, C) motion template.

    Each landmark follows a smooth sinusoid whose frequency/phase is seeded by
    landmark index, producing a coherent, normalized hand gesture over T frames.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, T)[:, None, None]
    freq = rng.uniform(0.6, 1.4, size=(V, C))
    phase = rng.uniform(0, 2 * np.pi, size=(V, C))
    amp = np.ones((V, C))
    for tip in _FINGER_TIPS:
        amp[tip] = 1.2
    seq = (amp * np.sin(freq * t + phase)).astype(np.float32)
    return normalize_hand_relative(seq)


def load_glosses(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data["glosses"].keys())


class MotionLibrary:
    """Holds one canonical motion template per gloss."""

    def __init__(self, motions: dict[str, np.ndarray], frames: int) -> None:
        self.motions = motions
        self.frames = frames

    def __contains__(self, gloss: str) -> bool:
        return gloss in self.motions

    def get(self, gloss: str) -> np.ndarray:
        """Return the (T, V, C) motion template for a gloss (zeros if missing)."""
        if gloss not in self.motions:
            return np.zeros((self.frames, _V, _C), dtype=np.float32)
        return self.motions[gloss]


def build_motion_library(glosses_path: Path, seed: int = 0) -> MotionLibrary:
    """Build a synthetic motion library for every canonical gloss."""
    glosses = load_glosses(glosses_path)
    motions: dict[str, np.ndarray] = {}
    for i, gloss in enumerate(glosses):
        motions[gloss] = make_motion_template(seed=seed + i)
    return MotionLibrary(motions, frames=_T)
