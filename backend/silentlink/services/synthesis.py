"""Text->Sign synthesis service.

Reuses the model package's synthesis engine (model/src/synthesis) to map input
text to a stitched landmark trajectory. The heavy work is off the device — this
is exactly the "cloud synthesis" the backend is meant to provide.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from ..config import REPO_ROOT

_MODEL_SRC = str(REPO_ROOT / "model" / "src")
if _MODEL_SRC not in sys.path:
    sys.path.insert(0, _MODEL_SRC)

from synthesis.motion import MotionLibrary, build_motion_library  # noqa: E402
from synthesis.normalize import GlossNormalizer, load_gloss_specs  # noqa: E402
from synthesis.stitch import stitch_glosses  # noqa: E402

_GLOSSES_JSON = REPO_ROOT / "model" / "configs" / "glosses.json"


@lru_cache(maxsize=1)
def _normalizer() -> GlossNormalizer:
    """Cached: parsing glosses.json and rebuilding the phrase table per request
    is pure overhead."""
    return GlossNormalizer(load_gloss_specs(_GLOSSES_JSON))


@lru_cache(maxsize=1)
def _library() -> MotionLibrary:
    """Cached: building the motion library forward-kinematics 11 glosses x 40
    frames on every request, when the result only changes when the choreography
    source does."""
    return build_motion_library(_GLOSSES_JSON)


def synthesize(text: str) -> dict:
    """Map text to glosses, stitch a trajectory, and return a compact result."""
    glosses = _normalizer().normalize(text)
    trajectory = stitch_glosses(glosses, _library())

    frames = int(trajectory.shape[0])
    return {
        "text": text,
        "glosses": glosses,
        "frames": frames,
        "channels": int(trajectory.shape[2]),
        "samples": min(int(trajectory.reshape(-1).size), 2000),  # preview only
    }
