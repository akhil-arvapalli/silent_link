"""Hand landmark capture using the MediaPipe Tasks API (HandLandmarker).

On-device/host capture path. Produces normalized (T, 42, 3) sequences that
match the training contract. Requires: mediapipe, opencv-python.

Model file: model/assets/hand_landmarker.task (downloaded during setup).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dataset import SEQUENCE_LENGTH, save_sequence
from .normalize import landmark_to_array, pack_hands, sequence_to_input

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"
DEFAULT_MODEL = ASSET_DIR / "hand_landmarker.task"


class HandCapture:
    """Wraps a MediaPipe HandLandmarker for frame-by-frame capture."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        min_detection: float = 0.5,
        min_tracking: float = 0.5,
    ):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {model_path}. "
                "Download hand_landmarker.task into model/assets/ (see AGENTS.md)."
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return (42, 3) packed landmarks for one BGR frame (zeros if no hands)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp = __import__("mediapipe")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
        self._frame_ts += 1

        left = right = None
        if result.hand_landmarks:
            # MediaPipe returns hands in order; identify by handedness if available.
            for i, hand in enumerate(result.hand_landmarks):
                arr = landmark_to_array(hand)
                if result.handedness and result.handedness[i][0].category_name == "Left":
                    left = arr
                elif result.handedness and result.handedness[i][0].category_name == "Right":
                    right = arr
                else:
                    # Fallback ordering: first is right, second is left (typical).
                    if i == 0:
                        right = arr
                    else:
                        left = arr
        return pack_hands(left, right)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def capture_sequence(
    capture: HandCapture,
    video: cv2.VideoCapture,
    length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Capture `length` frames of landmarks from the camera into one sequence."""
    frames: list[np.ndarray] = []
    while len(frames) < length:
        ok, frame = video.read()
        if not ok:
            break
        frames.append(capture.process_frame(frame))
        cv2.waitKey(10)
    if not frames:
        raise RuntimeError("No frames captured from camera")
    return sequence_to_input(np.stack(frames))


def capture_and_save(
    gloss: str,
    sequence_id: int,
    output_root: Path,
    num_frames: int = SEQUENCE_LENGTH,
    model_path: Path = DEFAULT_MODEL,
    camera_index: int = 0,
) -> Path:
    """Capture one sequence for a gloss and save to output_root/<gloss>/<id>.npy."""
    cap = HandCapture(model_path=model_path)
    video = cv2.VideoCapture(camera_index)
    try:
        seq = capture_sequence(cap, video, length=num_frames)
    finally:
        video.release()
        cap.close()
    return save_sequence(seq, gloss, sequence_id, output_root)
