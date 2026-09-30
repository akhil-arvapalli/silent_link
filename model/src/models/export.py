"""ONNX export for gesture classification models.

Exports a trained model to an ONNX graph with input (N, T=40, V=42, C=3),
the target for the on-device (TFLite / ONNX runtime) inference stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
import torch

from .baseline import Conv1D_BiLSTM
from .stgcn import STGCN


def _ensure_utf8_streams() -> None:
    # torch.onnx prints progress glyphs (e.g. "✅") that crash on cp1252
    # Windows consoles. Reconfigure streams to UTF-8 so export never fails
    # on encoding.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def load_model_state(model_name: str, n_classes: int, weights: Path, **kwargs) -> torch.nn.Module:
    model = _build(model_name, n_classes, **kwargs)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def _build(model_name: str, n_classes: int, **kwargs):
    if model_name == "baseline":
        return Conv1D_BiLSTM(n_classes=n_classes, **kwargs)
    if model_name == "stgcn":
        return STGCN(n_classes=n_classes, **kwargs)
    raise ValueError(f"Unknown model '{model_name}' (choose 'baseline' or 'stgcn')")


def export_onnx(
    model_name: str,
    n_classes: int,
    weights: Path,
    out_path: Path,
    T: int = 40,
    V: int = 42,
    C: int = 3,
    **kwargs,
) -> Path:
    model = load_model_state(model_name, n_classes, weights, T=T, V=V, C=C, **kwargs)
    _ensure_utf8_streams()
    # On-device inference processes one sequence at a time, so the exported
    # graph uses a fixed batch of 1 (also required for clean TFLite
    # conversion with the reshape-based graph conv).
    dummy = torch.randn(1, T, V, C)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["landmarks"],
        output_names=["logits"],
        opset_version=18,
    )
    # torch.onnx externalizes large weights into a sibling ".data" file.
    # Re-save as a single self-contained protobuf so the model can be bundled
    # as one app asset (the mobile inference stack expects a single file).
    _embed_weights(out_path)
    return out_path


def _embed_weights(path: Path) -> None:
    model_proto = onnx.load(str(path))
    onnx.save_model(model_proto, str(path), save_as_external_data=False)
    data_file = path.with_name(path.name + ".data")
    if data_file.exists():
        data_file.unlink()
