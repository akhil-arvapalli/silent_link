"""Per-gloss landmark motion library (Phase 2).

A gloss's *motion* is a canonical (T, 42, 3) landmark trajectory that the
renderer animates. Templates come from `choreography.py`: an anatomically
structured hand performing an authored ISL performance, not noise.

Real signer capture (scripts/capture_data.py) can replace these without
changing the interface — the stitching pipeline consumes the same
(T, V, C) templates either way. See ADR-007.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.normalize import NUM_HANDS, NUM_LANDMARKS

from .choreography import CHOREOGRAPHY, render_choreography
from .handmodel import REST_REACH, HandPose, build_hands

_FRAMES = 40
_V = NUM_HANDS * NUM_LANDMARKS
_C = 3


def load_glosses(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data["glosses"].keys())


def make_motion_template(seed: int = 0, T: int = _FRAMES) -> np.ndarray:
    """Build a deterministic (T, V, C) template for an un-authored gloss.

    Used for vocabulary entries that have no hand-authored performance yet. It
    is still a real hand: an open palm swaying through a seeded phase, so it
    degrades to "neutral gesture" rather than to noise.
    """
    phase = (seed % 32) * (np.pi / 8.0)
    out = np.zeros((T, _V, _C), dtype=np.float32)
    for i in range(T):
        t = i / max(1, T - 1)
        angle = 2.0 * np.pi * t
        pose = HandPose(
            curl=(0.15, 0.10, 0.10, 0.12, 0.18),
            spread=-0.25 + 0.05 * float(np.sin(angle + phase)),
            roll=0.35 * float(np.sin(angle + phase)),
            offset=(0.25 * float(np.cos(angle + phase)), 0.10 * float(np.sin(2 * angle)), 0.05),
        )
        hands = build_hands(pose)
        out[i, :NUM_LANDMARKS] = hands[0] / REST_REACH
        out[i, NUM_LANDMARKS:] = hands[1] / REST_REACH
    return out


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


def build_motion_library(glosses_path: Path, T: int = _FRAMES) -> MotionLibrary:
    """Build the motion library for every canonical gloss.

    Glosses with an authored performance use it; the rest fall back to a
    neutral swaying hand so a vocabulary addition never breaks synthesis.
    """
    glosses = load_glosses(glosses_path)
    motions: dict[str, np.ndarray] = {}
    for i, gloss in enumerate(glosses):
        if gloss in CHOREOGRAPHY:
            motions[gloss] = render_choreography(CHOREOGRAPHY[gloss], T)
        else:
            motions[gloss] = make_motion_template(seed=i, T=T)
    return MotionLibrary(motions, frames=T)
