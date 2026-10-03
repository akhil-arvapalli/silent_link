"""Contracts between the Python model and the app's TypeScript declarations.

`app/src/**` cannot be type-checked in CI (no `node_modules`, no `tsc`), and the
model artifacts are gitignored, so nothing tied the app's hardcoded tensor
names, shapes and vocabulary to the graph and data that actually ship. These
tests read the real ONNX graph, the real `glosses.json` and the real exported
motion library, then assert the app's source declares the same things.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from data.normalize import NUM_CHANNELS, NUM_HANDS, NUM_LANDMARKS, V
from models.stgcn import _HAND_EDGES
from synthesis.motion import build_motion_library

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
APP_SRC = APP / "src"
ONNX = APP / "assets" / "model.onnx"
GLOSSES_JSON = REPO / "model" / "configs" / "glosses.json"
MOTION_LIBRARY_JSON = APP_SRC / "synthesis" / "motionLibrary.json"

CLASSIFIER_TS = (APP_SRC / "model" / "classifier.ts").read_text(encoding="utf-8")
BONES_TS = (APP_SRC / "model" / "bones.ts").read_text(encoding="utf-8")
NORMALIZE_TS = (APP_SRC / "model" / "normalize.ts").read_text(encoding="utf-8")


def _const(source: str, name: str) -> int:
    """Read a `const NAME = <int>;` declaration out of a TS source file."""
    match = re.search(rf"\bconst\s+{re.escape(name)}\s*=\s*(-?\d+)\s*;", source)
    assert match, f"{name} is no longer a plain integer constant in the TS source"
    return int(match.group(1))


def _session():
    onnxruntime = pytest.importorskip("onnxruntime")
    return onnxruntime.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])


def test_app_glosses_json_is_byte_identical_to_the_model_copy():
    app_copy = APP_SRC / "config" / "glosses.json"
    assert app_copy.read_bytes() == GLOSSES_JSON.read_bytes(), (
        "the app's vocabulary copy has drifted from model/configs/glosses.json; "
        "run scripts/sync_app_assets.py"
    )


def test_training_contract_is_internally_consistent():
    contract = json.loads(GLOSSES_JSON.read_text(encoding="utf-8"))["training_contract"]
    assert contract["hand_landmarks_per_hand"] == NUM_LANDMARKS
    assert contract["hands"] == NUM_HANDS
    assert contract["hand_channels"] == NUM_CHANNELS
    assert contract["sequence_length_frames"] * V * NUM_CHANNELS == 40 * V * NUM_CHANNELS
    assert contract["input_shape"] == [
        contract["sequence_length_frames"],
        NUM_HANDS * NUM_LANDMARKS,
        NUM_CHANNELS,
    ]


def test_classifier_ts_shape_matches_the_exported_graph():
    session = _session()
    graph_input = session.get_inputs()[0]
    assert graph_input.name == "landmarks"
    # [batch, T, V, C]
    assert list(graph_input.shape) == [1, _const(CLASSIFIER_TS, "SEQUENCE_LENGTH"), V, NUM_CHANNELS]
    assert graph_input.type == "tensor(float)"


def test_classifier_ts_tensor_names_match_the_exported_graph():
    session = _session()
    assert re.search(r"INPUT_NAME\s*=\s*'landmarks'", CLASSIFIER_TS)
    assert re.search(r"OUTPUT_NAME\s*=\s*'logits'", CLASSIFIER_TS)
    assert session.get_outputs()[0].name == "logits"


def test_classifier_window_length_matches_training_contract():
    contract = json.loads(GLOSSES_JSON.read_text(encoding="utf-8"))["training_contract"]
    assert _const(CLASSIFIER_TS, "SEQUENCE_LENGTH") == contract["sequence_length_frames"]


def test_model_output_width_matches_the_gloss_vocabulary():
    """argmax over the logits indexes `GLOSSES` (sorted gloss keys)."""
    session = _session()
    graph_output = session.get_outputs()[0]
    assert len(graph_output.shape) == 2 and graph_output.shape[0] == 1

    glosses = json.loads(GLOSSES_JSON.read_text(encoding="utf-8"))["glosses"]
    assert int(graph_output.shape[1]) == len(glosses)


def test_app_shape_constants_match_the_normalizer():
    assert _const(NORMALIZE_TS, "V") == V
    assert _const(NORMALIZE_TS, "C") == NUM_CHANNELS


def test_app_has_no_hardcoded_shape_literals_left():
    """`21`/`42`/`3`/`40` must be imported, not retyped.

    The SkeletonAvatar drew `hand * 21` and GestureScreen showed `progress/40`
    while the same numbers lived in five other files. Each was a silent-drift
    site if the model ever changed.
    """
    files = {
        "components/SkeletonAvatar.tsx": (APP_SRC / "components" / "SkeletonAvatar.tsx"),
        "screens/GestureScreen.tsx": (APP_SRC / "screens" / "GestureScreen.tsx"),
    }
    imports = {
        "components/SkeletonAvatar.tsx": "../model/normalize",
        "screens/GestureScreen.tsx": "../model/classifier",
    }
    for label, path in files.items():
        source = path.read_text(encoding="utf-8")
        assert imports[label] in source, f"{label} must import its shape constants"
        body = re.sub(r"^import[\s\S]*?from\s+'[^']+';$", "", source, flags=re.MULTILINE)
        # Drop the comment block and font sizes, which legitimately use digits.
        code = "\n".join(
            line for line in body.splitlines() if not line.strip().startswith(("*", "/*", "//"))
        )
        code = re.sub(r"fontSize:\s*\d+", "fontSize: <n>", code)
        for literal in ("21", "42", "40"):
            assert literal not in code, f"{label} still hardcodes {literal}"


def test_bones_ts_edge_table_matches_the_model_adjacency():
    """The SkeletonAvatar draws `bones.ts`; the ST-GCN learns `stgcn._HAND_EDGES`.

    A divergence would render a skeleton the model was never trained on while
    still producing confident predictions.
    """
    pairs = re.findall(r"\[(\d+),\s*(\d+)\]", BONES_TS.split("export const HAND_EDGES")[1])
    assert [(int(a), int(b)) for a, b in pairs] == [tuple(e) for e in _HAND_EDGES]


def test_motion_library_asset_is_fresh():
    """`app/src/synthesis/motionLibrary.json` must match a rebuild.

    The choreography is authored Python; the app renders the exported JSON. If
    the export is stale the app signs something other than what the model
    package describes, and every Python-side test still passes.
    """
    shipped = json.loads(MOTION_LIBRARY_JSON.read_text(encoding="utf-8"))
    rebuilt = build_motion_library(GLOSSES_JSON, T=shipped["frames"])

    assert set(shipped["motions"]) == set(rebuilt.motions), "gloss coverage differs"
    for gloss, flat in shipped["motions"].items():
        expected = np.asarray(rebuilt.get(gloss), dtype=np.float32)
        assert len(flat) == expected.size, f"{gloss}: {len(flat)} vs {expected.size}"
        got = np.asarray(flat, dtype=np.float32).reshape(expected.shape)
        # The asset is rounded to 4 decimals on export.
        np.testing.assert_allclose(got, expected, atol=5e-5, err_msg=f"{gloss} is stale")


def test_motion_library_asset_frame_count_matches_the_stitcher():
    shipped = json.loads(MOTION_LIBRARY_JSON.read_text(encoding="utf-8"))
    rebuilt = build_motion_library(GLOSSES_JSON, T=shipped["frames"])
    assert shipped["frames"] == rebuilt.frames
    template = rebuilt.get("hello")
    assert template.shape == (shipped["frames"], V, NUM_CHANNELS)
