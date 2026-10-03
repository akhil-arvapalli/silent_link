"""Per-gloss ISL choreography for the Text->Sign motion library.

Each canonical gloss is a short keyframed performance: the handshape(s) the
sign is made in, plus the wrist path that gives it movement. Keyframes are
interpolated with a smoothstep so the rendered landmark trajectory is smooth
enough to survive the concatenative stitcher.

These are *authored* performances, not captured ones. They reproduce the
handshape and movement convention of each sign at a coarse level; they are not
a substitute for real signer capture (ADR-007). See scripts/capture_data.py for
the capture path that would replace them.
"""

from __future__ import annotations

import math

import numpy as np

from .handmodel import REST_REACH, HandPose, build_hands

# --- reusable handshapes -------------------------------------------------


def _open(spread: float = -0.25, **kw: object) -> HandPose:
    """Flat open palm, fingers together."""
    return HandPose(curl=(0.12, 0.04, 0.04, 0.05, 0.10), spread=spread, opposition=0.15, **kw)


def _fist(**kw: object) -> HandPose:
    """Closed fist."""
    return HandPose(curl=(0.75, 1.0, 1.0, 1.0, 1.0), spread=0.0, **kw)


def _flat(**kw: object) -> HandPose:
    """Flat hand, fingers together and extended (the 'please'/'thanks' shape)."""
    return HandPose(curl=(0.14, 0.10, 0.10, 0.12, 0.18), spread=-0.45, opposition=0.2, **kw)


def _ily(**kw: object) -> HandPose:
    """Thumb, index and pinky extended; middle and ring curled."""
    return HandPose(curl=(0.04, 0.04, 1.0, 1.0, 0.04), spread=0.15, opposition=0.35, **kw)


def _point(**kw: object) -> HandPose:
    """Index extended, everything else curled."""
    return HandPose(curl=(0.70, 0.04, 1.0, 1.0, 1.0), spread=-0.05, opposition=0.5, **kw)


def _v_shape(**kw: object) -> HandPose:
    """Index and thumb extended, remaining fingers closed."""
    return HandPose(curl=(0.05, 0.04, 1.0, 1.0, 1.0), spread=-0.1, opposition=0.55, **kw)


def _circle(cx: float, cy: float, r: float, n: int = 5, **kw: object) -> list[tuple]:
    """Keyframes tracing a full circle through (cx, cy) of radius r."""
    out = []
    for i in range(n + 1):
        angle = 2 * math.pi * i / n
        out.append(
            (
                i / n,
                _flat(offset=(cx + r * math.cos(angle), cy + r * math.sin(angle), 0.05), **kw),
                None,
            )
        )
    return out


# --- the canonical vocabulary --------------------------------------------

Key = tuple[float, HandPose, HandPose | None]

CHOREOGRAPHY: dict[str, list[Key]] = {
    "hello": [
        (0.0, _flat(offset=(0.0, 0.18, 0.10), pitch=-0.35), None),
        (1.0, _flat(offset=(0.30, -0.05, 0.28), pitch=-0.10), None),
    ],
    "thanks": [
        (0.0, _flat(offset=(0.0, 0.14, 0.10)), None),
        (1.0, _flat(offset=(0.26, -0.22, 0.30)), None),
    ],
    "iloveyou": [
        (0.0, _ily(offset=(0.0, 0.02, 0.06)), None),
        (0.5, _ily(offset=(0.16, 0.12, 0.20), pitch=-0.25), None),
        (1.0, _ily(offset=(0.0, 0.02, 0.06)), None),
    ],
    "yes": [
        (0.0, _fist(offset=(0.0, 0.16, 0.10)), None),
        (0.5, _fist(offset=(0.0, -0.12, 0.18)), None),
        (1.0, _fist(offset=(0.0, 0.16, 0.10)), None),
    ],
    "no": [
        (0.0, _v_shape(offset=(-0.22, 0.06, 0.10), roll=0.45), None),
        (0.5, _v_shape(offset=(0.22, 0.06, 0.10), roll=-0.45), None),
        (1.0, _v_shape(offset=(-0.10, 0.06, 0.10), roll=0.20), None),
    ],
    "help": [
        (0.0, _fist(offset=(0.0, -0.06, 0.02), pitch=math.pi / 2),
         _fist(offset=(0.0, 0.06, 0.12))),
        (1.0, _fist(offset=(0.0, 0.22, 0.06), pitch=math.pi / 2),
         _fist(offset=(0.0, 0.34, 0.16))),
    ],
    "please": _circle(0.10, 0.0, 0.14, roll=0.35),
    "sorry": [
        (
            t,
            _fist(
                offset=(
                    0.10 + 0.13 * math.cos(2 * math.pi * t),
                    0.13 * math.sin(2 * math.pi * t),
                    0.08,
                )
            ),
            None,
        )
        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    ],
    "good_morning": [
        (0.0, _open(offset=(0.0, -0.26, 0.02), roll=0.35), None),
        (1.0, _open(offset=(0.0, 0.26, 0.14), roll=-0.35), None),
    ],
    "good_night": [
        (0.0, _open(offset=(0.0, 0.26, 0.14), roll=-0.55), None),
        (1.0, _open(offset=(0.0, -0.26, 0.02), roll=0.55), None),
    ],
    "whats_your_name": [
        (0.0, _flat(offset=(0.0, 0.10, 0.14)), None),
        (1.0, _point(offset=(0.24, 0.16, 0.10), roll=0.30),
         _point(offset=(-0.24, 0.16, 0.10), roll=-0.30)),
    ],
}


def _smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def pose_at(keyframes: list[Key], t: float) -> tuple[HandPose, HandPose | None]:
    """Interpolated (left, right) pose at normalized time t in [0, 1]."""
    if t <= keyframes[0][0]:
        _, first_l, first_r = keyframes[0]
        return first_l, first_r
    if t >= keyframes[-1][0]:
        _, last_l, last_r = keyframes[-1]
        return last_l, last_r
    for (t0, l0, r0), (t1, l1, r1) in zip(keyframes, keyframes[1:], strict=False):
        if t0 <= t <= t1:
            span = t1 - t0
            u = _smoothstep((t - t0) / span) if span > 0 else 0.0
            return l0.blend(l1, u), (r0.blend(r1, u) if r0 and r1 else None)
    return keyframes[-1][1], keyframes[-1][2]


def render_choreography(keyframes: list[Key], frames: int) -> np.ndarray:
    """Render a keyframed performance into a (frames, 42, 3) landmark track.

    Coordinates are divided by `REST_REACH` — a fixed constant, not a per-frame
    measurement — so an open hand is 1.0 from wrist to middle fingertip and a
    closed one is visibly smaller. The wrist is deliberately *not* moved to the
    origin: translation is the whole point of a sign like "good morning", and
    the Skia renderer draws from absolute coordinates.
    """
    out = np.zeros((frames, 42, 3), dtype=np.float32)
    for i in range(frames):
        t = i / max(1, frames - 1)
        left, right = pose_at(keyframes, t)
        hands = build_hands(left, right)
        out[i, :21] = hands[0] / REST_REACH
        out[i, 21:] = hands[1] / REST_REACH
    return out
