"""Text->Sign synthesis service.

Reuses the model package's synthesis engine (model/src/synthesis) to map input
text to a stitched landmark trajectory. The heavy work is off the device — this
is exactly the "cloud synthesis" the backend is meant to provide.
"""

from __future__ import annotations

import sys

from ..config import REPO_ROOT

_MODEL_SRC = str(REPO_ROOT / "model" / "src")
if _MODEL_SRC not in sys.path:
    sys.path.insert(0, _MODEL_SRC)

from synthesis.motion import build_motion_library  # noqa: E402
from synthesis.normalize import GlossNormalizer, load_gloss_specs  # noqa: E402
from synthesis.stitch import stitch_glosses  # noqa: E402

_GLOSSES_JSON = REPO_ROOT / "model" / "configs" / "glosses.json"


def synthesize(text: str) -> dict:
    """Map text to glosses, stitch a trajectory, and return a compact result."""
    specs = load_gloss_specs(_GLOSSES_JSON)
    normalizer = GlossNormalizer(specs)
    glosses = normalizer.normalize(text)

    library = build_motion_library(_GLOSSES_JSON)
    trajectory = stitch_glosses(glosses, library)

    frames = int(trajectory.shape[0])
    return {
        "text": text,
        "glosses": glosses,
        "frames": frames,
        "channels": int(trajectory.shape[2]),
        "samples": min(int(trajectory.reshape(-1).size), 2000),  # preview only
    }
