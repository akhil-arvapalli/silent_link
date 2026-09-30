# Architecture Decision Records (ADR)

Statuses: `proposed` | `accepted` | `superseded`. Append new ADRs; do not rewrite history. Number sequentially.

---

## ADR-001 — Training Framework: Keras → PyTorch

- **Status:** accepted
- **Date:** 2026-09-19
- **Context:** Legacy uses Keras LSTM, poor for modern cloud tooling/quantization/ONNX.
- **Decision:** Migrate training to PyTorch + PyTorch Lightning. Export via ONNX → TFLite for mobile.
- **Consequences:** Better cloud/MLflow/W&B support, first-class ONNX export, larger ecosystem for ST-GCN/transformer research.

---

## ADR-002 — Model Architecture: ST-GCN primary, Conv1D+BiLSTM baseline

- **Status:** accepted
- **Date:** 2026-09-19
- **Context:** Raw hand landmarks are graph-structured point clouds over time. A plain ViT (image model) is the wrong tool; LSTM underperforms on skeleton action recognition.
- **Decision:** Primary = Spatial-Temporal Graph Conv Network (ST-GCN) over hand graphs. Baseline = Conv1D + BiLSTM to beat. Optional second track = Transformer encoder over frame tokens.
- **Consequences:** Hand-only input `(T=40, V=42, C=3)`, no face/pose compute, fewer params, better spatial priors.

---

## ADR-003 — Target Language: ISL only

- **Status:** accepted
- **Date:** 2026-09-19
- **Context:** User explicitly scoped the project to ISL. Legacy GIFs are ISL-ish; WLASL/MSASL are ASL.
- **Decision:** ISL only. Use INCLUDE dataset; ASL datasets only for transfer/pretraining, never silently mixed.
- **Consequences:** Slower data growth than ASL; requires curation discipline.

---

## ADR-004 — Text→Sign: Concatenative landmark synthesis (not GIF lookup)

- **Status:** accepted
- **Date:** 2026-09-19
- **Context:** Legacy is a hardcoded phrase → GIF dictionary with zero generalization.
- **Decision:** Build a per-gloss **hand-landmark motion library** + coarticulation-blending stitcher, rendered as a Skia animated skeleton. Neural motion synthesis (motion diffusion) is a stretch research track, not the ship path.
- **Consequences:** Fully on-device, deterministic, infinitely scalable to any gloss; removes the 90-GIF hard cap.

---

## ADR-005 — On-device MediaPipe: Hand Landmarker (not Holistic)

- **Status:** accepted
- **Date:** 2026-09-19
- **Context:** MediaPipe Holistic (face+pose) doesn't run well on React Native.
- **Decision:** Use the MediaPipe **Hand Landmarker** task API on-device; design model input hand-only (aligns with ADR-002).
- **Consequences:** Dropping face/pose features from the input contract.

---

## ADR-006 — Data Pipeline: MediaPipe Tasks API + hand-only normalized sequences

- **Status:** accepted
- **Date:** 2026-09-19
- **Context:** Installed mediapipe 1.0.1 removed `mp.solutions` (the legacy capture API). Capture must use the Tasks API.
- **Decision:** Build the data pipeline (`model/src/data/`) on the **Tasks API** `HandLandmarker`, outputting wrist-relative, scale-normalized `(T=40, V=42, C=3)` sequences. Storage: `data/processed/<gloss>/<id>.npy`.
- **Consequences:** Consistent on-device/host representation; enables landmark-space augmentation; matches the mobile inference contract.

---

## ADR-007 — Text→Sign Synthesis: on-device TS mirror of the Python stitcher

- **Status:** accepted
- **Date:** 2026-09-20
- **Context:** ADR-004 chose concatenative landmark synthesis. The synthesis engine (normalizer, motion library, stitcher) lives in Python for authoring/testing; the renderer must run fully on-device in React Native.
- **Decision:** Keep the canonical synthesis logic in `model/src/synthesis/` (Python, unit-tested) and **mirror it in TypeScript** under `app/src/synthesis/`. The per-gloss motion library is exported to a bundled JSON asset (`motionLibrary.json`) by `scripts/sync_app_assets.py`; the on-device stitcher produces the same trajectory, rendered by `react-native-skia` (`SkeletonAvatar`).
- **Consequences:** Deterministic, fully offline Text→Sign; single source of truth remains the Python package + `glosses.json`. Risk: TS/Python drift — mitigated by keeping both mirrors identical and the tests as the spec. Synthetic motion templates stand in until real capture exists.

