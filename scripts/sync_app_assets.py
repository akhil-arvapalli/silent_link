#!/usr/bin/env python
"""Sync trained model artifacts + canonical vocab from the model package into the RN app.

Keeps app assets in sync with the latest trained model without hardcoding vocab
(AGENTS.md: read from glosses.json). Copies:

    model/configs/glosses.json            -> app/src/config/glosses.json
    model/assets/hand_landmarker.task     -> app/assets/hand_landmarker.task
    <run>/model.onnx                      -> app/assets/model.onnx

Usage:
    python scripts/sync_app_assets.py [--run model/runs/default]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model" / "src"))

from synthesis.export import export_motion_library  # noqa: E402
from synthesis.motion import build_motion_library  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync model artifacts into the RN app")
    parser.add_argument("--run", type=Path, default=REPO_ROOT / "model" / "runs" / "default")
    parser.add_argument(
        "--no-motions", action="store_true", help="Skip generating/exporting the motion library"
    )
    args = parser.parse_args()

    app = REPO_ROOT / "app"
    copies = [
        (REPO_ROOT / "model" / "configs" / "glosses.json", app / "src" / "config" / "glosses.json"),
        (REPO_ROOT / "model" / "assets" / "hand_landmarker.task", app / "assets" / "hand_landmarker.task"),
        (args.run / "model.onnx", app / "assets" / "model.onnx"),
    ]

    for src, dst in copies:
        if not src.exists():
            print(f"SKIP (missing source): {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src.name} -> {dst.relative_to(REPO_ROOT)}")

    if not args.no_motions:
        glosses_path = REPO_ROOT / "model" / "configs" / "glosses.json"
        library = build_motion_library(glosses_path)
        dst = app / "src" / "synthesis" / "motionLibrary.json"
        export_motion_library(library, dst)
        print(f"generated motion library -> {dst.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
