"""Serialize the motion library to a compact JSON asset for the RN app.

The on-device Text->Sign renderer needs a per-gloss motion library. Real
templates come from webcam capture; until then synthetic templates stand in.
This module writes app/src/synthesis/motionLibrary.json with a flat
{gloss: [T*V*C numbers]} layout (rounded) to keep the bundle small.
"""

from __future__ import annotations

import json
from pathlib import Path

from .motion import MotionLibrary


def export_motion_library(library: MotionLibrary, path: Path, decimals: int = 4) -> Path:
    """Write the motion library as a compact JSON asset.

    Layout:
        {"frames": T, "version": "1.0.0", "motions": {gloss: [flat float array]}}
    """
    payload: dict[str, object] = {
        "frames": library.frames,
        "version": "1.0.0",
        "motions": {
            gloss: [round(float(v), decimals) for v in template.reshape(-1).tolist()]
            for gloss, template in library.motions.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path
