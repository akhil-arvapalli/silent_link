import json

import numpy as np

from synthesis.motion import MotionLibrary, make_motion_template
from synthesis.normalize import GlossNormalizer
from synthesis.stitch import stitch_glosses

_GLOSSES = {
    "hello": {"words": ["hello", "hi"], "type": "phrase"},
    "thanks": {"words": ["thank", "thanks", "thank you"], "type": "phrase"},
    "good_morning": {"words": ["good morning"], "type": "phrase"},
}


def _specs() -> dict[str, dict[str, object]]:
    return json.loads(json.dumps(_GLOSSES))


def _library() -> MotionLibrary:
    motions = {g: make_motion_template(seed=i) for i, g in enumerate(_GLOSSES)}
    return MotionLibrary(motions, frames=40)


def test_normalizer_phrase_match():
    n = GlossNormalizer(_specs())
    assert n.normalize("Hello") == ["hello"]
    assert n.normalize("thank you very much") == ["thanks"]
    assert n.normalize("good morning") == ["good_morning"]


def test_normalizer_longest_phrase_wins():
    n = GlossNormalizer(_specs())
    # "thank you" (2 words) should match as one phrase before "thank"/"thanks"
    assert n.normalize("thank you") == ["thanks"]


def test_normalizer_multiple_and_collapse():
    n = GlossNormalizer(_specs())
    assert n.normalize("hello hello") == ["hello"]
    assert n.normalize("hello thanks") == ["hello", "thanks"]


def test_normalizer_empty_and_unknown():
    n = GlossNormalizer(_specs())
    assert n.normalize("") == []
    assert n.normalize("zzz zzz") == []


def test_motion_template_shape_and_normalized():
    m = make_motion_template(seed=1)
    assert m.shape == (40, 42, 3)
    assert m.dtype == np.float32


def test_motion_library_build_and_get():
    lib = _library()
    assert "hello" in lib
    m = lib.get("hello")
    assert m.shape == (40, 42, 3)
    missing = lib.get("nope")
    assert not missing.any()


def test_stitch_single_is_identity():
    lib = _library()
    single = stitch_glosses(["hello"], lib)
    np.testing.assert_allclose(single, lib.get("hello"))


def test_stitch_empty():
    lib = _library()
    out = stitch_glosses([], lib)
    assert out.shape == (0, 42, 3)


def test_stitch_multiple_length():
    lib = _library()
    out = stitch_glosses(["hello", "thanks"], lib)
    # 40 + (40 - blend) with blend=8 => 72 frames
    assert out.shape == (72, 42, 3)


def test_stitch_seam_continuous():
    lib = _library()
    out = stitch_glosses(["hello", "thanks"], lib, blend_frames=8)
    # No abrupt discontinuity across the seam (finite differences stay small).
    d = np.abs(np.diff(out, axis=0)).max()
    assert d < 5.0


def test_stitch_three_glosses_length_and_continuity():
    lib = _library()
    out = stitch_glosses(["hello", "thanks", "good_morning"], lib, blend_frames=8)
    # T + (n-1)*(T-w) = 40 + 2*32
    assert out.shape == (104, 42, 3)
    d = np.abs(np.diff(out, axis=0)).max()
    assert d < 5.0
