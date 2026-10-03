"""Text-to-Sign synthesis engine (Phase 2).

Pipeline: sentence -> gloss normalization -> concatenative landmark motion
stitching -> (T, 42, 3) sequence for the on-device Skia renderer.

Motion templates come from `choreography.py`, which drives `handmodel.py`'s
forward-kinematic hand. They are authored performances, not captured ones
(ADR-007); scripts/capture_data.py replaces them without changing any
interface here.
"""

from .choreography import CHOREOGRAPHY, render_choreography
from .handmodel import HandPose, build_hand, build_hands
from .motion import MotionLibrary, build_motion_library, make_motion_template
from .normalize import GlossNormalizer, load_gloss_specs
from .stitch import stitch_glosses

__all__ = [
    "CHOREOGRAPHY",
    "GlossNormalizer",
    "HandPose",
    "MotionLibrary",
    "build_hand",
    "build_hands",
    "build_motion_library",
    "load_gloss_specs",
    "make_motion_template",
    "render_choreography",
    "stitch_glosses",
]
