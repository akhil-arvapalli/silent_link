# TODO — Current & Upcoming Work

> One item `in_progress` at a time. Update statuses as work proceeds. Phases require explicit user confirmation before starting.

## In Progress
- [ ] Backend (FastAPI) service — core built (auth, datasets, train jobs, model registry, synthesis)
  - [x] `silentlink/security.py` — PBKDF2 password hashing + HMAC-signed bearer tokens (zero extra deps)
  - [x] `silentlink/db.py` — thread-safe in-memory stores (users, datasets, jobs, models, synthesis)
  - [x] Routers: auth (signup/login/me), datasets (disk scan+sync), jobs (train queue), models (registry), synthesis (text→sign)
  - [x] `services/synthesis.py` reuses `model/src/synthesis` for real on-device-matching trajectories
  - [x] `services/training.py` runs the real `scripts/train_model.py` in a background job
  - [x] `create_app()` factory + isolated-DB test seam; 22 tests, ruff clean
  - [ ] Persist to a real DB (SQLite/Postgres) — currently in-memory
  - [ ] Auth hardening (refresh tokens, rate limiting), artifact storage for trained models
  - [ ] 9 of 13 backend routes have no UI (`/datasets`, all `/jobs`, `/models`, `GET /synthesis/{id}`, `GET /auth/me`) — app cannot reach them
- [ ] CI + verification gates
  - [x] `.github/workflows/ci.yml` — ruff + pytest for `model/` and `backend/`, `tsc --noEmit` for `app/`
  - [x] `model/tests/test_ts_parity.py` — executes the real `app/src/**.ts` mirrors under Node and compares numerically to Python (no npm install needed)
  - [x] `model/tests/test_app_contract.py` — pins `classifier.ts` tensor names/shape, `bones.ts` edges, `glosses.json` and `motionLibrary.json` to the real artifacts
  - [ ] `npm ci && npx tsc --noEmit` verified green on a machine with network (never run here)
- [ ] App ↔ backend integration + fingerspelling + icon polish
  - [x] `src/api/client.ts` + `src/config/api.ts` — typed fetch client, bearer token, backend base URL
  - [x] `LoginScreen` (signup/login) wired into `App.tsx` auth state + logout
  - [x] `TextToSignScreen` sends the phrase to backend `/synthesis` when signed in (offline-first animation unchanged)
  - [x] `FingerspellingScreen` + `src/model/fingerspell.ts` (per-frame letter classifier, stub until real model)
  - [x] `Icon.tsx` — Skia flat line icons replacing Home emoji
  - [x] tsc clean; model 64 tests + ruff clean
  - [ ] Real per-frame letter model for fingerspelling (needs data/training)
    - [ ] Only 7 of the 26 letters in `glosses.json` → `fingerspell.letters` have reference shapes (`a b c l o v y`); the other 19 can never be produced
- [ ] Phase 2: Text→Sign synthesis engine — core built (authored motion), capture pending
  - [x] `model/src/synthesis/`: sentence→gloss normalizer, motion library, coarticulation stitcher
  - [x] `scripts/sync_app_assets.py` exports bundled `motionLibrary.json` into the app
  - [x] App-side TS mirrors (`normalize.ts`, `motion.ts`, `stitch.ts`)
  - [x] `SkeletonAvatar` (react-native-skia) + `TextToSignScreen` (input → gloss chips → animated sign)
  - [x] App UI restyled to the "Group 2133" mockup (teal/coral on soft blue) — Home, Gesture, Text→Sign
  - [x] Authored motion library: `handmodel.py` (FK hand) + `choreography.py` (keyframed ISL per gloss) replaces random-sinusoid templates (ADR-008)
  - [x] App `normalize.ts` normalizer corrected to match training (was ~10x off); `stitch.ts` ramp matches Python
  - [ ] Replace authored motion templates with real captured landmark motion (needs webcam capture)
  - [ ] Neural motion synthesis (Phase 3 stretch) / sentence→gloss seq2seq
- [ ] Phase 1c: RN app — camera + landmarker + on-device inference + fixed-vocab text out
  - [x] Expo app scaffolded (TS, SDK 57) in `app/` with dev-client
  - [x] Native stack: vision-camera v4 + worklets-core + `expo-vision-camera-v4-mediapipe` (Android) + onnxruntime-react-native
  - [x] `expo prebuild` verified — MediaPipe plugin generates Kotlin plugin + copies `.task` to assets
  - [x] `src/model/normalize.ts` mirrors training normalization; `classifier.ts` runs ST-GCN ONNX
  - [x] Vocab read from bundled `glosses.json` (no hardcoding); class order matches training
  - [x] GestureScreen: live camera + frame processor + 40-frame buffer + top-gloss display
  - [x] ONNX export made self-contained (~1.5 MB) for single-asset bundling
  - [x] `scripts/sync_app_assets.py` keeps app vocab/model in sync with model package
  - [x] `tsc` clean + `expo export --platform android` bundles (1017 modules)
  - [ ] Native build on a real device/emulator (no Android toolchain here) — user runs `expo run:android`
  - [ ] Replace synthetic `model.onnx` with a model trained on real captured ISL data
  - [ ] iOS native module (needs macOS) — deferred

## Phase 1b — Gesture Classifier (model code done; retrain pending)
- [x] `model/src/models/`: Conv1D+BiLSTM baseline, ST-GCN (primary)
- [x] `model/src/data/loader.py`: stratified train/val split, DataLoader glue
- [x] `model/src/models/training.py`: train loop + early stopping (anti-`epochs=2000` guardrail)
- [x] `model/src/models/export.py`: ONNX export (fixed batch=1, self-contained, on-device target)
- [x] `scripts/train_model.py` CLI + `scripts/make_synthetic_data.py` (pipeline smoke-test)
- [x] CUDA torch (2.14+cu130) on RTX 2050; ONNX verified via onnxruntime
- [x] Tests (`test_models.py` 9) + ruff clean; end-to-end verified on synthetic data (ST-GCN ~97% val acc)
- [ ] Retrain on real captured data (data/processed) once captured
- [ ] Cloud training harness (Colab/Vertex/Lambda) + TFLite conversion (Phase 1b stretch)

## Phase 1a — Data Pipeline (complete)
- [x] Hand-keypoint capture (MediaPipe Tasks API HandLandmarker)
- [x] Keypoint extraction + normalization (hand-only, (T,V,C) format)
- [x] Landmark-space augmentation (rotation, scale, mirror, temporal jitter, noise)
- [x] Dataset read/write + `scripts/capture_data.py` CLI
- [x] Tests (9) + ruff clean; end-to-end webcam capture verified

## Phase 2 (in progress) — Text→Sign Synthesis Engine
- [x] Sentence→gloss normalization (rule-based; seq2seq is the stretch path)
- [x] Concatenative landmark synth + coarticulation stitcher
- [x] Skia animated skeleton renderer (replaces GIF lookup)
- [x] Authored per-gloss motion library — FK hand + keyframes, all 11 glosses distinct
- [ ] Real captured per-gloss motion library (replace authored templates)

## Phase 3 (blocked) — Scale & Polish
- [ ] Neural motion synthesis (stretch), TTS, iOS, signer generalization, brutalism polish
