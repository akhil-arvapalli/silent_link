"""Anatomical hand model for authoring ISL landmark motion.

Builds MediaPipe-shaped 21-point hands from a joint *configuration* rather
than from noise, so a gloss template is a real hand performing a real
movement. The output is the same (V=42, C=3) normalized landmark space the
training pipeline and the Skia renderer already use.

Local hand frame used throughout:
    +x  radial (towards the thumb)
    +y  distal (along the fingers)
    +z  dorsal (out of the palm)

A pose is a set of per-finger curl angles plus a wrist placement and
orientation; forward kinematics chains them out into 21 landmarks.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# Landmark chain per finger, root joint first, tip last.
FINGERS: dict[str, tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# Segment lengths, from the wrist outward along each chain. Units are chosen
# so a fully extended middle finger reaches ~1.0 from the wrist, which is the
# scale normalize_hand_relative() later divides away anyway.
BONE_LENGTHS: dict[str, tuple[float, ...]] = {
    "thumb": (0.22, 0.14, 0.10, 0.09),
    "index": (0.30, 0.17, 0.10, 0.07),
    "middle": (0.33, 0.18, 0.10, 0.08),
    "ring": (0.31, 0.17, 0.09, 0.07),
    "pinky": (0.27, 0.14, 0.08, 0.06),
}

# Rest direction of each finger from the wrist in the local frame.
REST_DIR: dict[str, tuple[float, float, float]] = {
    "thumb": (0.66, 0.60, 0.14),
    "index": (0.30, 0.95, 0.02),
    "middle": (0.05, 0.99, 0.00),
    "ring": (-0.18, 0.97, 0.00),
    "pinky": (-0.42, 0.90, 0.04),
}

# Total rotation from straight to fully closed at each finger joint.
CURL_TOTAL = np.deg2rad(105.0)
SPREAD_MAX = np.deg2rad(20.0)
OPPOSITION_MAX = np.deg2rad(60.0)

# Per-finger share of the spread, so `spread` fans the fingers apart or draws
# them together. A single rigid rotation would only spin the whole hand and
# leave the fan width unchanged.
FAN_WEIGHT = {
    "thumb": -0.50,
    "index": -0.55,
    "middle": 0.0,
    "ring": 0.50,
    "pinky": 1.0,
}

_PALM_NORMAL = np.array([0.0, 0.0, 1.0])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of the vector `v` about `axis` by `angle` radians."""
    k = _unit(np.asarray(axis, dtype=np.float64))
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about `axis` by `angle` radians."""
    k = _unit(np.asarray(axis, dtype=np.float64))
    c, s = np.cos(angle), np.sin(angle)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)


class HandPose:
    """A hand's joint configuration.

    Args:
        curl: per-finger curl in [0, 1], ordered as FINGER_ORDER.
            0 is fully extended, 1 is a closed fist.
        spread: finger splay in [-1, 1] (negative closes the fingers together).
        opposition: thumb swing out of the palm plane in [0, 1].
        roll: wrist rotation about the palm normal, radians.
        pitch: wrist nod about the radial axis, radians.
        offset: wrist translation in hand-local units.
    """

    __slots__ = ("curl", "spread", "opposition", "roll", "pitch", "offset")

    def __init__(
        self,
        curl: tuple[float, float, float, float, float] = (0.0,) * 5,
        spread: float = 0.0,
        opposition: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.curl = tuple(float(min(1.0, max(0.0, c))) for c in curl)
        self.spread = float(spread)
        self.opposition = float(opposition)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.offset = tuple(float(v) for v in offset)

    def blend(self, other: HandPose, t: float) -> HandPose:
        """Linearly interpolate towards `other` by t in [0, 1]."""
        u = min(1.0, max(0.0, t))
        return HandPose(
            curl=tuple(a + (b - a) * u for a, b in zip(self.curl, other.curl, strict=True)),
            spread=self.spread + (other.spread - self.spread) * u,
            opposition=self.opposition + (other.opposition - self.opposition) * u,
            roll=self.roll + (other.roll - self.roll) * u,
            pitch=self.pitch + (other.pitch - self.pitch) * u,
            offset=tuple(a + (b - a) * u for a, b in zip(self.offset, other.offset, strict=True)),
        )


def build_hand(pose: HandPose) -> np.ndarray:
    """Forward-kinematics a single (21, 3) hand from a pose."""
    pts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
    for f, name in enumerate(FINGER_ORDER):
        chain = FINGERS[name]
        lengths = BONE_LENGTHS[name]
        d = _rotate(
            np.array(REST_DIR[name], dtype=np.float64),
            _PALM_NORMAL,
            pose.spread * SPREAD_MAX * FAN_WEIGHT[name],
        )
        if name == "thumb":
            d = _rotate(d, np.array([0.0, 1.0, 0.0]), -pose.opposition * OPPOSITION_MAX)
        pts[chain[0]] = _unit(d) * lengths[0]
        step = pose.curl[f] * CURL_TOTAL / (len(chain) - 1)
        for i in range(1, len(chain)):
            axis = np.cross(_PALM_NORMAL, d)
            d = _unit(_rotate(d, axis, step))
            pts[chain[i]] = pts[chain[i - 1]] + d * lengths[i]

    rot = _rotation_matrix(_PALM_NORMAL, pose.roll) @ _rotation_matrix(
        np.array([0.0, 1.0, 0.0]), pose.pitch
    )
    return pts @ rot.T + np.array(pose.offset, dtype=np.float64)


def build_hands(pose: HandPose, right_pose: HandPose | None = None) -> np.ndarray:
    """Build the (2, 21, 3) pair; the right hand mirrors the left's rest shape.

    Two-handed signs pass `right_pose`; everything else signs one-handed and
    the non-dominant hand is only a passive mirror.
    """
    out = np.zeros((2, NUM_LANDMARKS, 3), dtype=np.float64)
    out[0] = build_hand(pose)
    out[1] = build_hand(right_pose if right_pose is not None else pose)
    out[1] *= np.array([-1.0, 1.0, 1.0])
    return out


# Wrist-to-middle-tip length of the flat open hand. Dividing authored motion by
# this gives a unit where an open hand is exactly 1.0 long from wrist to middle
# fingertip, which is what the Skia renderer's pixel scale is calibrated
# against. Normalizing by mean *bone* length instead would put the hand at 4.5
# units across, far outside the render canvas.
REST_REACH = float(np.linalg.norm(build_hand(HandPose(curl=(0.0,) * 5))[12]))
