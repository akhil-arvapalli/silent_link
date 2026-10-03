# AGENTS.md — Persistent Memory for Agents

This file is the primary memory for any agent (human or AI) working on this repo. Read this first. Update it whenever project conventions, decisions, or structure change. Keep it accurate and current — it is the source of truth for how to work here.

## Project Identity

- **Name:** Silent Link — v1-prod (rebuild)
- **Purpose:** Offline-first Indian Sign Language (ISL) communication app: gesture → text, and text → animated sign (replacing the GIF-dictionary approach).
- **Target language:** ISL ONLY. Do not silently mix ASL datasets.
- **Repo root:** `D:\DL_MODEL_SILENT_LINK\silent-link_v1-prod\`

## Guardrails / Do Not Do

- Do **not** modify the legacy folders `ActionDetectionforSignLanguage/` or `silent_link/` (the old toy). This repo is a clean rebuild.
- Do **not** add comments to code unless asked.
- Do **not** commit unless explicitly asked.
- Do **not** introduce the old `epochs=2000` + `test_size=0.05` training anti-pattern. Use proper validation splits + early stopping.
- **Privacy/offline is a selling point**: keep inference on-device; only training/heavy synthesis hits the cloud.
- Data is the bottleneck. Prioritize dataset work over architecture churn.

## Architecture Summary (target)

```
React Native (Expo) app
├─ Gesture→Text : camera → MediaPipe Hand Landmarker → keypoint buffer → ONNX/TFLite model → gloss/text
├─ Text→Sign    : input → sentence→gloss normalization → concatenative landmark synth → Skia animated skeleton
Backend (FastAPI): auth, datasets, train jobs, model registry, synthesis queue
Model (PyTorch): ST-GCN (primary) / Conv1D+BiLSTM (baseline); export via ONNX → TFLite
Cloud training: Colab / Vertex AI / Lambda Labs; Lightning + MLflow/W&B + DVC
```

See `docs/ARCHITECTURE.md` (to be written) and `docs/ADR.md` for decisions.

## Model Input Contract (Phase 1 target)

- **Hand-only landmarks** (NOT MediaPipe Holistic face/pose): `(T=40, V=42 hands, C=3)` normalized.
- Fingerspelling: separate lightweight per-frame hand-shape classifier.
- Export target: < 5 MB, < 15 ms/frame on mid-range phone.

## Canonical Vocabulary

- Single source of truth: `model/configs/glosses.json` (canonical ISL gloss list).
- Consistent sequence counts (target 100 seq/word, 40 frames).
- Do NOT hardcode vocab lists in app code (the old repo's `isl_gif` array was an anti-pattern). Read from `glosses.json`.

## Commands

- Lint: `ruff check .` (Python) — run in `backend/` and `model/`
- Tests: `pytest` — run in `backend/` and `model/` (model uses `pythonpath=src`)
- CI: `.github/workflows/ci.yml` — ruff + pytest for both packages, plus `tsc --noEmit` for `app/`. Nothing runs these locally by default, so push or run them before claiming green.
- Data capture: `python scripts/capture_data.py [--gloss hello] [--per-gloss 100]`
- Synthetic smoke data (no webcam): `python scripts/make_synthetic_data.py [--per-gloss 40]`
- Train: `python scripts/train_model.py [--model stgcn|baseline] [--epochs 100] [--batch-size 32] [--out model/runs/<name>]`
  - Trains with stratified train/val split + early stopping (never the `epochs=2000`/`test_size=0.05` anti-pattern), saves `best_model.pt` + `metrics.json`, exports `model.onnx` (self-contained).
- Sync trained model + vocab into the app: `python scripts/sync_app_assets.py [--run model/runs/<name>]`
- RN (app/): `npm` / `npx expo` — **native dev build required** (not Expo Go). Prebuild + run:
  - `npx expo prebuild --clean` then `npx expo run:android` (needs Android SDK/device). iOS not wired (Android-only MediaPipe plugin).
  - `npx tsc --noEmit` typecheck; `npx expo export --platform android` to verify bundling.

## Data Pipeline (Phase 1a — built)

- `model/src/data/normalize.py` — hand-only landmark → `(T, V=42, C=3)` wrist-relative, scale-invariant.
- `model/src/data/augment.py` — landmark-space augmentation (rotation, scale, mirror, temporal jitter, noise).
- `model/src/data/dataset.py` — save/load sequences to `data/processed/<gloss>/<id>.npy`.
- `model/src/data/capture.py` — MediaPipe **Tasks API** `HandLandmarker` capture (NOT `mp.solutions`, removed in mediapipe 1.0.1).
- Model file: `model/assets/hand_landmarker.task` (downloaded, 7.8 MB).
- Tests: `model/tests/test_data_pipeline.py` (9 tests).

MediaPipe note: mediapipe 1.0.1 dropped `mp.solutions`; use `mediapipe.tasks.python.vision.HandLandmarker`. This also matches the on-device RN stack (ADR-005).

## App (Phase 1c — built, native dev build)

- `@mediapipe/tasks-vision` (web/WASM) **cannot run in React Native** (Hermes lacks DOM/WASM-loading). Use a **native** MediaPipe binding instead (ADR-005).
- On-device stack: `react-native-vision-camera` **v4** (v5 changed to Nitro and breaks the plugin) + `react-native-worklets-core` + `expo-vision-camera-v4-mediapipe` (Android-only Kotlin frame-processor plugin) + `onnxruntime-react-native`.
- Model assets bundled as single files: `model.onnx` is **self-contained** (weights embedded; `export.py` re-saves without external data) and `hand_landmarker.task` is copied to Android assets at prebuild.
- Sync app assets from the model package: `python scripts/sync_app_assets.py [--run model/runs/<name>]`.
- Verify without a device: `npx tsc --noEmit` and `npx expo export --platform android` (metro must treat `.onnx`/`.task` as assets — see `metro.config.js`).
- App UI follows the **"Group 2133" mockup** design system: tokens in `app/src/theme.ts` (soft blue bg, mint/teal chips, coral primary CTA, deep-teal ink). Home → Gesture→Text / Text→Sign via a simple shell in `App.tsx` (no nav lib yet).

## Phase 2 — Text→Sign Synthesis (core built, real-motion pending)

- Canonical logic in `model/src/synthesis/` (Python, unit-tested): `normalize.py` (sentence→gloss), `handmodel.py` (forward-kinematic MediaPipe hand), `choreography.py` (keyframed ISL per gloss), `motion.py` (per-gloss `(T,42,3)` motion library), `stitch.py` (concatenative + coarticulation crossfade), `export.py` (→ bundled JSON).
- On-device TS mirrors in `app/src/synthesis/` (`normalize.ts`, `motion.ts`, `stitch.ts`); renderer is `app/src/components/SkeletonAvatar.tsx` (react-native-skia) with topology in `app/src/model/bones.ts` (mirrors ST-GCN `_HAND_EDGES`).
- Motion templates are **authored, not captured** (ADR-007, ADR-008). `HandPose` drives 21 joints by forward kinematics; `CHOREOGRAPHY` holds one keyframed performance per gloss. Templates are divided by `handmodel.REST_REACH` so an open hand is 1.0 wrist-to-middle-tip — this deliberately keeps wrist translation, which the classifier's own wrist-relative normalization would erase. Replace with real captured landmark motion before production.
- `TextToSignScreen` renders at `scale={100}` in a 300×280 stage; `test_library_fits_the_render_canvas` pins the authored extent to ≤1.4 units so it cannot outgrow the canvas.
- `scripts/sync_app_assets.py` regenerates `app/src/synthesis/motionLibrary.json`.

## Backend (FastAPI — core built, in-memory persistence)

- Auth: `silentlink/security.py` (PBKDF2 hashing + HMAC-signed bearer tokens, stdlib only) + `routers/auth.py`.
- Persistence: `silentlink/db.py` — thread-safe **in-memory** stores; `create_app(db)` factory in `main.py` lets tests inject an isolated DB. Swap for SQLite/Postgres repositories before production.
- Routers: `datasets.py` (scans `data/processed`), `jobs.py` (train-job queue running `scripts/train_model.py`), `models.py` (registry), `synthesis.py` (text→sign, reuses `model/src/synthesis` so on-device trajectories match).
- Run: `uvicorn silentlink.main:app --reload`. Tests: `pytest` in `backend/`. Lint: `ruff check .`.

## File Organization

```
app/            React Native (Expo) — Phase 1+ (native dev build required)
  src/screens/  HomeScreen.tsx, GestureScreen.tsx (Gesture→Text camera), TextToSignScreen.tsx (Text→Sign), FingerspellingScreen.tsx, LoginScreen.tsx
  src/components/ SkeletonAvatar.tsx (Skia animated skeleton), Icon.tsx (Skia line icons)
  src/api/      client.ts (backend fetch client), config/api.ts (base URL)
  src/synthesis/ normalize.ts, motion.ts, stitch.ts (TS mirrors of Python) + motionLibrary.json (bundled)
  src/model/    normalize.ts (mirror of normalize.py), classifier.ts (onnxruntime), bones.ts (hand topology), fingerspell.ts (per-frame letter classifier)
  src/hooks/    useGestureRecognizer.ts (40-frame buffer + inference)
  src/config/   glosses.json + glosses.ts (vocab, read from file, not hardcoded)
  src/theme.ts  design tokens (Group 2133 mockup)
  assets/       hand_landmarker.task, model.onnx (synced by scripts/sync_app_assets.py)
backend/        FastAPI service — Phase 1+ (core built, in-memory persistence)
  silentlink/   main.py (create_app factory), config.py, db.py, security.py, deps.py, schemas.py
    routers/    auth.py, datasets.py, jobs.py, models.py, synthesis.py
    services/   datasets.py, training.py (job runner), synthesis.py (reuses model/src/synthesis)
  tests/        conftest.py, test_security.py, test_auth.py, test_api.py, test_jobs.py
model/          PyTorch training, configs, datasets — Phase 1+
  src/          model source
    data/       normalize, augment, dataset, loader (split/DataLoader)
    models/     baseline (Conv1D+BiLSTM), stgcn, training (early stop), export (ONNX)
    synthesis/  normalize (sentence→gloss), handmodel (FK hand), choreography (authored ISL), motion (library), stitch (coarticulation), export (→ JSON)
  configs/      glosses.json, training configs
  datasets/     dataset loaders
  runs/         training outputs (best_model.pt, metrics.json, model.onnx)
  tests/        test_data_pipeline.py, test_models.py, test_synthesis.py, test_handmodel.py,
                test_ts_parity.py, test_app_contract.py, test_config.py
data/           raw/ + processed/ (keypoints, WebDataset shards)
docs/           ADR.md, ARCHITECTURE.md, etc.
scripts/        one-off / automation scripts
```

## The app is not type-checked locally — two test suites stand in for it

`app/node_modules` is absent, so `npx tsc --noEmit` cannot run here. Two
suites in `model/tests/` cover the app's Python contract from the Python side,
and both need no npm install:

- **`test_ts_parity.py`** stages the *real* `app/src/**.ts` mirrors into a temp
  dir and executes them under Node (>= 22.6, native type stripping), then
  compares the results against the Python originals numerically. This is what
  stops the mirrors drifting again — the `normalize.ts` wrist-distance bug was
  invisible to every other check. Skips cleanly if `node` is absent, so CI
  installs Node to keep it live. `test_the_guard_detects_injected_drift` is a
  negative control: it reinstates the original bug and requires parity to fail,
  so the guard cannot go vacuous.
- **`test_app_contract.py`** reads the shipped `model.onnx` with onnxruntime
  and asserts `classifier.ts`'s tensor names, window length and `V`/`C`, plus
  `bones.ts` vs `stgcn._HAND_EDGES`, the byte-identity of the app's
  `glosses.json`, and that `motionLibrary.json` still matches a rebuild.

When you change any of `app/src/model/*`, `app/src/synthesis/*` or the
choreography, run `pytest` in `model/`. Both suites fail loudly on drift.

## Verified state

- `model`: 64 tests, `backend`: 22 tests, `ruff check .` clean in both.
- The ONNX graph, the app's tensor contract and the gloss vocabulary agree
  (checked against the real artifact, not against the source comments).
- **Still unverified without a device:** the app renders, the camera frame
  processor and MediaPipe plugin run, and the native Android build succeeds.
  `app/patches/onnxruntime-react-native+1.24.3.patch` is still inert (no
  `patch-package`, no `postinstall`), so a fresh `npm ci` + Gradle build will
  hit the unpatched `VersionNumber.parse(REACT_NATIVE_VERSION)` reference.

## Progress & State

See `TODO.md` (current work) and `CHANGELOG.md` (released changes). Phases are executed with explicit user confirmation between phases — do not advance phases unilaterally.
