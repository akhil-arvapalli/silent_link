"""Concatenative landmark motion stitching (Phase 2, ADR-004).

Takes a gloss sequence (e.g. ["hello", "thanks"]) and produces a continuous
(T_total, 42, 3) normalized landmark trajectory by concatenating each gloss's
motion template with a coarticulation crossfade at the boundaries.

Coarticulation: instead of hard-cutting between signs (which produces a
visible snap), we overlap the tail of the current sign and the head of the
next sign and blend them with a raised-cosine window. This is the same
temporal interpolation trick the training pipeline uses for resampling.
"""

from __future__ import annotations

import numpy as np

from data.normalize import NUM_CHANNELS, V

from .motion import MotionLibrary


def _raised_cosine_blend(window: int) -> np.ndarray:
    """A 0->1 raised-cosine ramp of length `window`."""
    if window <= 0:
        return np.empty(0, dtype=np.float32)
    x = np.linspace(0, np.pi, window)
    return (0.5 - 0.5 * np.cos(x)).astype(np.float32)


def stitch_glosses(
    glosses: list[str],
    library: MotionLibrary,
    blend_frames: int = 8,
) -> np.ndarray:
    """Stitch a gloss sequence into one (T, V, C) landmark trajectory.

    Args:
        glosses: canonical gloss keys, in order.
        library: MotionLibrary holding per-gloss (T, V, C) templates.
        blend_frames: number of frames to crossfade at each sign boundary.
    Returns:
        (T_total, V, C) float32 normalized sequence (concatenated with blends).
        Empty (0, V, C) if no glosses.
    """
    if not glosses:
        return np.zeros((0, V, NUM_CHANNELS), dtype=np.float32)

    blocks: list[np.ndarray] = []
    for gloss in glosses:
        blocks.append(np.asarray(library.get(gloss), dtype=np.float32))

    if len(blocks) == 1:
        return blocks[0]

    parts = [blocks[0]]
    for i in range(1, len(blocks)):
        nxt = blocks[i]
        # Blend the actual accumulated tail with the next sign's head.
        w = min(blend_frames, parts[-1].shape[0], nxt.shape[0])
        if w <= 0:
            parts.append(nxt)
            continue
        ramp = _raised_cosine_blend(w)[:, None, None]
        blended = parts[-1][-w:] * (1.0 - ramp) + nxt[:w] * ramp
        parts[-1] = parts[-1][:-w]
        parts.append(blended)
        parts.append(nxt[w:])

    return np.concatenate(parts, axis=0).astype(np.float32)
