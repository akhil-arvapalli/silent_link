import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from synthesis.choreography import CHOREOGRAPHY, pose_at, render_choreography
from synthesis.handmodel import FINGERS, HandPose, build_hand, build_hands
from synthesis.motion import build_motion_library

_GLOSSES = Path(__file__).resolve().parents[1] / "configs" / "glosses.json"


def _canonical() -> list[str]:
    return sorted(json.loads(_GLOSSES.read_text(encoding="utf-8"))["glosses"])


def test_build_hand_shape_and_wrist():
    hand = build_hand(HandPose())
    assert hand.shape == (21, 3)
    assert np.isfinite(hand).all()
    np.testing.assert_allclose(hand[0], np.zeros(3), atol=1e-12)


def test_rest_bone_lengths_respected():
    hand = build_hand(HandPose(curl=(0.0,) * 5))
    np.testing.assert_allclose(np.linalg.norm(hand[5]), 0.30, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(hand[9]), 0.33, atol=1e-6)


def test_curl_folds_fingertips_into_the_palm():
    open_hand = build_hand(HandPose(curl=(0.0,) * 5))
    fist = build_hand(HandPose(curl=(1.0,) * 5))
    for name in ("index", "middle", "ring", "pinky"):
        tip = list(FINGERS[name])[-1]
        # +z is out of the palm, so a closed fist must sit clearly below it.
        assert fist[tip][2] < open_hand[tip][2] - 0.1, name


def test_spread_splays_fingertips_apart():
    closed = build_hand(HandPose(spread=-1.0))
    splayed = build_hand(HandPose(spread=1.0))
    width_closed = closed[8][0] - closed[20][0]
    width_splayed = splayed[8][0] - splayed[20][0]
    assert width_splayed > width_closed


def test_build_hands_mirrors_the_right_hand():
    pose = HandPose(curl=(0.3, 0.2, 0.2, 0.2, 0.2), spread=0.3)
    hands = build_hands(pose)
    assert hands.shape == (2, 21, 3)
    np.testing.assert_allclose(hands[1][:, 0], -hands[0][:, 0], atol=1e-12)
    np.testing.assert_allclose(hands[1][:, 1:], hands[0][:, 1:], atol=1e-12)


def test_two_handed_choreography_differs_between_hands():
    left = HandPose(curl=(0.5,) * 5)
    right = HandPose(curl=(0.0,) * 5)
    hands = build_hands(left, right)
    assert not np.allclose(hands[0], hands[1])


def test_every_canonical_gloss_has_choreography():
    missing = [g for g in _canonical() if g not in CHOREOGRAPHY]
    assert missing == []


def test_choreography_keyframes_are_ordered_and_in_range():
    for gloss, keys in CHOREOGRAPHY.items():
        times = [t for t, _, _ in keys]
        assert times == sorted(times), gloss
        assert all(0.0 <= t <= 1.0 for t in times), gloss
        assert times[0] == 0.0 and times[-1] == 1.0, gloss


def test_render_choreography_shape_and_smoothness():
    track = render_choreography(CHOREOGRAPHY["hello"], 40)
    assert track.shape == (40, 42, 3)
    assert track.dtype == np.float32
    assert np.isfinite(track).all()
    # No frame-to-frame jump large enough to look like a pop in the renderer.
    assert np.abs(np.diff(track, axis=0)).max() < 2.0


def test_rendered_templates_keep_the_wrist_off_origin():
    """Translation carries the sign, so normalization must not recentre it."""
    track = render_choreography(CHOREOGRAPHY["good_morning"], 40)
    assert np.abs(track[:, 0, 1]).max() > 0.2


def test_library_covers_the_vocabulary_at_consistent_scale():
    lib = build_motion_library(_GLOSSES)
    for gloss in _canonical():
        template = lib.get(gloss)
        assert template.shape == (40, 42, 3), gloss
        assert np.isfinite(template).all(), gloss


def test_library_templates_are_mutually_distinct():
    """No two glosses may render as the same gesture.

    `hello` and `thanks` are the closest pair (both a flat hand leaving the
    face), which is faithful to ISL; the floor just has to reject templates
    that are actually interchangeable.
    """
    lib = build_motion_library(_GLOSSES)
    names = _canonical()
    for a, b in itertools.combinations(names, 2):
        diff = float(np.abs(lib.get(a) - lib.get(b)).mean())
        assert diff > 0.05, f"{a} and {b} are too similar ({diff:.3f})"


def test_library_fits_the_render_canvas():
    """TextToSignScreen draws at scale=100 into a 300x280 stage (140px half-height).

    Anything beyond ~1.4 units would be clipped, so pin the authored motion to
    stay comfortably inside that.
    """
    lib = build_motion_library(_GLOSSES)
    for gloss in _canonical():
        template = lib.get(gloss)
        assert np.abs(template).max() <= 1.4, f"{gloss} overflows the render canvas"


def test_open_hand_is_one_unit_long():
    """REST_REACH normalization: an open hand spans 1.0 wrist-to-middle-tip."""
    template = render_choreography(CHOREOGRAPHY["good_morning"], 4)
    reach = np.linalg.norm(template[0, 12] - template[0, 0])
    assert reach == pytest.approx(1.0, abs=0.15)


def test_missing_gloss_returns_zero_template():
    lib = build_motion_library(_GLOSSES)
    assert not lib.get("not_a_gloss").any()


def test_pose_at_hits_the_endpoints():
    keys = CHOREOGRAPHY["yes"]
    start, _ = pose_at(keys, 0.0)
    end, _ = pose_at(keys, 1.0)
    np.testing.assert_allclose(start.offset, keys[0][1].offset)
    np.testing.assert_allclose(end.offset, keys[-1][1].offset)


def test_synthesis_round_trip_is_reproducible():
    a = build_motion_library(_GLOSSES).get("thanks")
    b = build_motion_library(_GLOSSES).get("thanks")
    np.testing.assert_array_equal(a, b)
