"""Text-to-Sign synthesis engine (Phase 2).

Pipeline: sentence -> gloss normalization -> concatenative landmark motion
stitching -> (T, 42, 3) normalized sequence for the on-device Skia renderer.
"""

from .motion import MotionLibrary, build_motion_library, make_motion_template
from .normalize import GlossNormalizer, load_gloss_specs
from .stitch import stitch_glosses

__all__ = [
    "GlossNormalizer",
    "MotionLibrary",
    "build_motion_library",
    "load_gloss_specs",
    "make_motion_template",
    "stitch_glosses",
]
