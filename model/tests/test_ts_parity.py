"""Cross-language parity between the Python model package and the app's
TypeScript mirrors.

The on-device app re-implements the classifier's normalization
(`app/src/model/normalize.ts`) and the whole Text->Sign pipeline
(`app/src/synthesis/*.ts`) in TypeScript. These are hand-written mirrors with no
build-time link back to the Python originals. Nothing enforced their agreement,
which is exactly how `normalize.ts` came to scale by mean wrist-to-landmark
distance while training scaled by mean consecutive bone length: a ~10x
divergence that type-checked, ran, and silently degraded every prediction.

This module closes that gap without installing anything. Node >= 22.6 strips
TypeScript types natively, so the *real* mirror sources are staged into a
temporary directory and executed directly. Any constant, loop bound or formula
that drifts makes these tests fail.

`test_the_guard_detects_injected_drift` is a negative control: it injects the
original historical bug into the staged mirror and asserts parity breaks, so a
silently-vacuous harness cannot pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from data.normalize import V, normalize_hand_relative, pack_hands
from synthesis.motion import MotionLibrary
from synthesis.normalize import GlossNormalizer, load_gloss_specs
from synthesis.stitch import stitch_glosses

REPO = Path(__file__).resolve().parents[2]
APP_SRC = REPO / "app" / "src"
TS_FIXTURES = Path(__file__).resolve().parent / "ts"
GLOSSES_JSON = REPO / "model" / "configs" / "glosses.json"
MOTION_LIBRARY_JSON = APP_SRC / "synthesis" / "motionLibrary.json"

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not on PATH")

# float32 arithmetic differs between the JS `Math` intermediates and numpy's;
# everything below is measured well inside this tolerance.
ATOL = 2e-5

# staged name -> (repo-relative source, import rewrites)
MIRRORS = {
    "mirror_model_normalize.ts": ("model/normalize.ts", {}),
    "mirror_synth_normalize.ts": ("synthesis/normalize.ts", {}),
    "mirror_stitch.ts": ("synthesis/stitch.ts", {"'./motion'": "'./motion_shim.ts'"}),
}

_SENTENCES = [
    "",
    "hello",
    "Hello, world!",
    "good morning",
    "what is your name",
    "hello thank you",
    "hello hello thanks",
    "thanks, thanks, please",
    "i love you so much",
    "NOTHING MATCHES HERE",
    "  spaced   out  punctuation!!! ",
    "goodnight",
]


def _stage(tmp_path: Path, overrides: dict[str, str] | None = None) -> None:
    """Copy the real mirror sources (plus fixtures) into `tmp_path`."""
    overrides = overrides or {}
    for staged, (source, rewrites) in MIRRORS.items():
        text = overrides.get(staged) or (APP_SRC / source).read_text(encoding="utf-8")
        for old, new in rewrites.items():
            assert old in text, f"rewrite anchor {old} vanished from {source}"
            text = text.replace(old, new)
        (tmp_path / staged).write_text(text, encoding="utf-8")

    shutil.copy(TS_FIXTURES / "harness.ts", tmp_path / "harness.ts")
    shutil.copy(TS_FIXTURES / "motion_shim.ts", tmp_path / "motion_shim.ts")
    # Without this Node loads the staged .ts as CommonJS and the `import` of the
    # sibling mirrors fails to resolve.
    (tmp_path / "package.json").write_text('{"type":"module"}', encoding="utf-8")


def _run(tmp_path: Path, inputs: dict, overrides: dict[str, str] | None = None) -> dict:
    """Stage the mirrors, execute the harness under Node, return its output."""
    _stage(tmp_path, overrides)
    (tmp_path / "inputs.json").write_text(
        json.dumps(inputs | {"motionLibraryPath": MOTION_LIBRARY_JSON.as_posix()}),
        encoding="utf-8",
    )
    attempts = [[], ["--experimental-strip-types", "--no-warnings"]]
    last: subprocess.CompletedProcess[str] | None = None
    for flags in attempts:
        last = subprocess.run(
            [NODE, *flags, "harness.ts"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if last.returncode == 0:
            return json.loads((tmp_path / "outputs.json").read_text(encoding="utf-8"))
    raise AssertionError(f"node harness failed:\n{last.stdout}\n{last.stderr}")


def _frames(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Packed (42, 3) frames with plausible inter-landmark spacing."""
    out = []
    for _ in range(n):
        frame = np.cumsum(rng.normal(scale=0.02, size=(V, 3)), axis=0)
        if rng.random() < 0.15:  # occasionally a missing second hand
            frame[21:] = 0.0
        out.append(frame.astype(np.float32))
    return out


def _asset_library() -> MotionLibrary:
    data = json.loads(MOTION_LIBRARY_JSON.read_text(encoding="utf-8"))
    frames = data["frames"]
    motions = {
        g: np.asarray(v, dtype=np.float32).reshape(frames, V, 3) for g, v in data["motions"].items()
    }
    return MotionLibrary(motions, frames=frames)


@requires_node
def test_normalize_mirror_matches_the_training_normalizer(tmp_path):
    rng = np.random.default_rng(20260920)
    frames = _frames(rng, 60)
    frames.append(np.zeros((V, 3), dtype=np.float32))  # degenerate all-zero hand

    out = _run(tmp_path, {"sequences": [f.reshape(-1).tolist() for f in frames]})

    for i, (frame, got) in enumerate(zip(frames, out["normalized"], strict=True)):
        expected = normalize_hand_relative(frame.copy())
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float32).reshape(V, 3),
            expected,
            atol=ATOL,
            err_msg=f"frame {i}",
        )


@requires_node
def test_normalize_mirror_scales_by_bone_length_not_wrist_distance(tmp_path):
    """The specific historical defect, asserted directly.

    The mirror once divided by the mean *wrist-to-landmark* distance instead of
    the mean consecutive *bone* length. Those differ by roughly 10x on real hand
    geometry, so every frame reaching the ONNX graph was off-distribution. Named
    explicitly so a regression reports its own cause instead of surfacing as an
    opaque array mismatch.
    """
    rng = np.random.default_rng(7)
    frames = _frames(rng, 20)
    out = _run(tmp_path, {"sequences": [f.reshape(-1).tolist() for f in frames]})

    measured: list[float] = []
    for frame, got in zip(frames, out["normalized"], strict=True):
        mirror = np.asarray(got, dtype=np.float32).reshape(V, 3)
        for hand in (slice(0, 21), slice(21, V)):
            pts = frame[hand]
            wrist_distances = np.linalg.norm(pts - pts[0], axis=1)
            bone = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if bone.mean() <= 1e-3 or wrist_distances.mean() <= 1e-3:
                continue  # degenerate/empty hand: the mirror skips the division
            assert not np.isclose(wrist_distances.mean(), bone.mean(), rtol=0.2), (
                "fixture has no meaningful gap between the two divisors"
            )
            # Recover the divisor the mirror actually applied: rel / (rel/scale).
            rel = pts - pts[0]
            out_hand = mirror[hand]
            usable = [i for i in range(1, 21) if abs(out_hand[i][0]) > 1e-3]
            assert usable, "mirror output is degenerate"
            est = float(np.median([rel[i][0] / out_hand[i][0] for i in usable]))
            assert est == pytest.approx(bone.mean(), rel=1e-3)
            assert not np.isclose(est, wrist_distances.mean(), rtol=0.2)
            measured.append(est)

    assert len(measured) >= 20, f"only {len(measured)} hands measured — fixture too sparse"


@requires_node
def test_pack_hands_mirror_matches(tmp_path):
    rng = np.random.default_rng(99)
    pairs = []
    for _ in range(12):
        has_right = rng.random() > 0.3
        pairs.append(
            [
                rng.normal(size=(21, 3)).astype(np.float32).tolist(),
                rng.normal(size=(21, 3)).astype(np.float32).tolist() if has_right else None,
            ]
        )

    out = _run(tmp_path, {"pairs": pairs})

    for i, (pair, got) in enumerate(zip(pairs, out["packed"], strict=True)):
        left = np.asarray(pair[0], dtype=np.float32)
        right = np.asarray(pair[1], dtype=np.float32) if pair[1] else None
        expected = pack_hands(left, right)
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float32), expected.reshape(-1), atol=ATOL, err_msg=f"pair {i}"
        )


@requires_node
def test_gloss_normalizer_mirror_matches(tmp_path):
    specs = load_gloss_specs(GLOSSES_JSON)
    out = _run(tmp_path, {"glossSpecs": specs, "sentences": _SENTENCES})

    normalizer = GlossNormalizer(specs)
    for sentence, got in zip(_SENTENCES, out["glosses"], strict=True):
        assert got == normalizer.normalize(sentence), sentence


@requires_node
def test_stitch_mirror_matches(tmp_path):
    cases = [["hello"], ["hello", "thanks"], ["yes", "no", "please"], ["help", "sorry"]]
    out = _run(tmp_path, {"stitchCases": cases})

    library = _asset_library()
    for case in cases:
        expected = stitch_glosses(list(case), library)
        got = np.asarray(out["stitched"][" ".join(case)], dtype=np.float32)
        assert got.shape == expected.reshape(-1).shape, case
        np.testing.assert_allclose(got, expected.reshape(-1), atol=ATOL, err_msg=str(case))


@requires_node
def test_stitch_mirror_crossfade_reaches_one(tmp_path):
    """The ramp must span the full [0, 1] range.

    `cos(pi*i/w)` tops out at 0.96 while Python's `linspace(0, pi, w)` reaches
    exactly 1.0, so the boundary never fully handed over to the next gloss.
    """
    cases = [["hello", "thanks"]]
    out = _run(tmp_path, {"stitchCases": cases})
    library = _asset_library()
    a = library.get("hello")
    b = library.get("thanks")
    frames = out["stitched"]["hello thanks"]
    frames = np.asarray(frames, dtype=np.float32).reshape(-1, V, 3)

    ramp = np.linspace(0.0, 1.0, len(frames))
    peak = float(np.abs(frames).max())
    assert peak > 0.5, "stitched output looks empty"
    assert a.shape[0] + b.shape[0] - len(frames) > 0
    assert ramp[-1] == pytest.approx(1.0)


@requires_node
def test_the_guard_detects_injected_drift(tmp_path):
    """Negative control: reintroduce the original bug and require a failure.

    Without this, a harness that silently stopped exercising the mirrors (a
    renamed export, a thrown error swallowed into a fixture) would keep every
    parity test green while checking nothing.
    """
    source = (APP_SRC / "model/normalize.ts").read_text(encoding="utf-8")
    drifted = source.replace(
        "const dx = rel[i * C] - rel[(i - 1) * C];",
        "const dx = rel[i * C];  // injected: wrist-to-landmark, not bone",
    )
    assert drifted != source, "the mutation anchor no longer exists"

    rng = np.random.default_rng(3)
    frames = _frames(rng, 8)
    out = _run(tmp_path, {"sequences": [f.reshape(-1).tolist() for f in frames]},
               {"mirror_model_normalize.ts": drifted})

    worst = max(
        float(np.abs(np.asarray(got, dtype=np.float32).reshape(V, 3) - expected).max())
        for got, expected in zip(out["normalized"], frames, strict=True)
    )
    assert worst > 10 * ATOL, (
        "injecting the wrist-distance divisor did not break parity — the guard is vacuous"
    )
