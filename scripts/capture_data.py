#!/usr/bin/env python
"""Interactive ISL gesture data capture.

Records landmark sequences from the webcam for the glosses defined in
model/configs/glosses.json and saves them to data/processed.

Usage:
    python scripts/capture_data.py                 # capture for all glosses
    python scripts/capture_data.py --gloss hello   # capture only 'hello'
    python scripts/capture_data.py --per-gloss 100 --frames 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model" / "src"))

from data.capture import HandCapture, capture_sequence  # noqa: E402
from data.dataset import SEQUENCE_LENGTH, save_sequence  # noqa: E402


def load_glosses(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data["glosses"].keys())


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture ISL hand landmark data")
    parser.add_argument("--gloss", help="Capture only this gloss")
    parser.add_argument("--per-gloss", type=int, default=100, help="Sequences per gloss (default 100)")
    parser.add_argument("--frames", type=int, default=SEQUENCE_LENGTH, help="Frames per sequence")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--glosses-json",
        type=Path,
        default=REPO_ROOT / "model" / "configs" / "glosses.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
    )
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "model" / "assets" / "hand_landmarker.task")
    args = parser.parse_args()

    import cv2

    glosses = [args.gloss] if args.gloss else load_glosses(args.glosses_json)
    args.out.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(args.camera)
    cap = HandCapture(model_path=args.model)
    try:
        for gloss in glosses:
            for seq_id in range(args.per_gloss):
                print(f"\n>>> {gloss} — sequence {seq_id}/{args.per_gloss}. Strike any key to record.")
                cv2.waitKey(0)
                seq = capture_sequence(cap, video, length=args.frames)
                path = save_sequence(seq, gloss, seq_id, args.out)
                print(f"   saved {path}")
    finally:
        video.release()
        cap.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
