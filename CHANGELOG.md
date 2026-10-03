# Changelog

All notable changes to this project. Format based on [Keep a Changelog](https://keepachangelog.com/). Each phase is a section; record meaningful changes, decisions, and fixes.

## [Unreleased]

## [Hardening] — 2026-10-03 — Verification gates + backend/app defect fixes

The defect sweep below fixed real bugs but left the *reason* they existed
untouched: `app/src/**` is hand-mirrored from Python with nothing checking the
copies, `model.onnx` is gitignored so no artifact was pinned to a contract, and
nothing ran `ruff`/`pytest` automatically. That is what this section closes.

### Added
- **`model/tests/test_ts_parity.py`** — executes the *real* `app/src/model/normalize.ts`,
  `app/src/synthesis/{normalize,stitch}.ts` under Node's native type stripping and
  compares the results numerically against `data/normalize.py` and
  `synthesis/{normalize,stitch}.py`. No npm install, no `node_modules`. Skips
  cleanly when `node` is absent.
  - Includes `test_the_guard_detects_injected_drift`, a **negative control**: it
    reinstates the exact wrist-distance bug in a staged copy and requires parity
    to break, so the guard cannot silently become vacuous.
  - `test_normalize_mirror_scales_by_bone_length_not_wrist_distance` recovers the
  divisor the mirror actually applied and asserts it is the bone length, so a
  regression names its own cause.
- **`model/tests/test_app_contract.py`** — reads the shipped `model.onnx` with
  `onnxruntime` and pins `classifier.ts` (input/output tensor names, 40-frame
  window, `V`/`C`), `bones.ts` vs `stgcn._HAND_EDGES`, the byte-identity of the
  app's `glosses.json`, and that `motionLibrary.json` still matches a rebuild of
  the authored choreography.
- **`backend/tests/test_jobs.py`** — `TrainingQueue` and `run_job` never executed
  under the old suite (the only `/jobs` assertion checked 401 before the handler
  ran). Covers success, runner failure, serialization, owner scoping, and that a
  job id is not a capability across accounts.
- **`.github/workflows/ci.yml`** — `ruff` + `pytest` for both packages and
  `tsc --noEmit` for `app/`. Installs Node so the parity suite runs in CI instead
  of skipping.

### Fixed
- **`classifier.ts` leaked a native tensor per inference.** The input `ort.Tensor`
  was never disposed and the session is a module-level singleton, so every
  40-frame window retained one tensor plus its 5040-float buffer indefinitely.
- **A failed model load was cached forever.** `sessionPromise` was assigned before
  it settled and only cleared by `disposeSession()` (never called), so one
  transient failure left every later `predict()` rejecting instantly — gesture
  recognition silently dead for the rest of the session. The promise is now reset
  on failure.
- **`reset()` was undone by an in-flight inference.** The pending `.then` still
  fired after the user cleared the result card, repopulating the gloss they had
  just dismissed. Added a generation token.
- **Two contradictory `declare global` blocks for `detectHandLandmarks`.**
  `GestureScreen` and `FingerspellingScreen` each declared it with a different
  result type; `declare global` merges duplicates as overloads rather than
  erroring, so the effective type was decided by file ordering. Declared once in
  `useGestureRecognizer.ts`.
- **`classifier.ts` widened a `Tensor` to `any[]`.** `Array.isArray(logits)` can
  never be true — `onnxvalue` types map values as `OnnxValue = Tensor` in
  `onnxruntime-common@1.24.3` — and the branch only compiled because `Array.from`
  on the intersection degraded to `any`.
- **`GET /datasets` was not read-only.** It called `sync_datasets`, which ran
  `db.datasets.clear()` and rebuilt from disk on every request — a read that
  destroyed every registered dataset, walked the whole processed tree twice, and
  `np.load`-ed an entire sequence just to read a frame count. `GET` now upserts
  (idempotent, mtime-cached, `mmap`); only `POST /datasets/refresh` rebuilds.
- **`POST /synthesis` advertised `202 Accepted` + `queued` while running inline.**
  Status code, status field and client contract all disagreed. The on-device path
  reads `glosses` straight off the response, so it is now honestly `200`.
- **A generated signing key broke multi-worker deployments.** `deps.SECRET` was
  computed per import, so under `uvicorn --workers N` each worker minted a
  different key and tokens failed verification at random. The key is now cached in
  a gitignored `backend/.silentlink_secret`, which also stops tokens dying on every
  restart.
- **`TrainingQueue` spawned one unbounded thread per job** while its docstring
  claimed "executes one job at a time" — N concurrent training subprocesses. Now
  bounded by a semaphore, and jobs waiting for a slot stay honestly `queued`.
- **`numpy` was missing from `backend/pyproject.toml`** although
  `services/datasets.py` and `tests/conftest.py` both import it — a fresh clone
  failed at collection.
- **Per-request synthesis rebuilt the whole motion library** (forward kinematics
  over 11 glosses × 40 frames) and re-parsed `glosses.json`. Both are now
  `lru_cache`d.
- **`fingerspell.ts` degenerate feature vector was 8 long, real one is 9.** The
  8-length fallback made `distance()` skip the pinky spread and compare against
  mis-aligned templates.
- **Duplicated shape constants.** `SEQUENCE_LENGTH`/`V`/`C` existed in five files
  and `40` was hardcoded twice in `GestureScreen`. Now defined once in
  `model/normalize.ts` / `model/classifier.ts` and imported.
- `.gitignore` did not cover `*.task` (two 7.8 MB MediaPipe bundles),
  `*.onnx.data` (orphaned sidecars `export.py` already deletes), `.knownPackages`
  or the generated `backend/.silentlink_secret`.

### Verified
- `model` 64 tests, `backend` 22 tests, `ruff check .` clean in both.
- The parity suite confirms the TS mirrors match the Python originals to within
  float32 rounding, on 60 random frames including an all-zero hand.
- The exported graph (`landmarks [1,40,42,3]` → `logits [1,11]`), the app's
  classifier declarations and the 11-gloss vocabulary agree, checked against the
  real artifact.
- `motionLibrary.json` is current: a fresh rebuild matches the shipped asset.

### Known gaps (unchanged, still need a decision or data)
- `app/node_modules` is absent, so the app is still not type-checked or bundled
  locally. `model/tests/` now covers its Python contract, but nothing renders it.
- `app/patches/onnxruntime-react-native+1.24.3.patch` is still inert — no
  `patch-package`, no `postinstall` — so a fresh `npm ci` + Gradle build hits the
  unpatched `VersionNumber.parse(REACT_NATIVE_VERSION)`. Wiring it needs an npm
  install.
- No real ISL data has ever been collected; `data/processed/` does not exist. Every
  ML artifact, including the shipped `model.onnx`, is trained on synthetic data.

## [Fixes] — 2026-10-03 — Defect sweep + authored motion library

### Added
- `model/src/synthesis/handmodel.py` — MediaPipe hand topology with anatomically valid bone lengths, a `HandPose` (per-finger curl, spread, thumb opposition, wrist roll/pitch/offset) and forward kinematics producing real (21, 3) hands.
- `model/src/synthesis/choreography.py` — keyframed ISL performances for all 11 canonical glosses, smoothstep-interpolated.
- `model/tests/test_handmodel.py` — tests covering bone lengths, curl direction, spread, mirroring, choreography coverage, distinctness and render extent.
- ADR-008 — motion templates keep wrist translation.

### Changed
- `model/src/synthesis/motion.py` — motion library now built from choreography instead of per-gloss random sinusoids. Un-authored glosses fall back to a neutral swaying hand rather than noise.
- Motion templates normalize by `REST_REACH` (open hand = 1.0 wrist-to-middle-tip) instead of mean bone length, so wrist travel survives and the skeleton fits the render canvas. `TextToSignScreen` renders at `scale={100}`.

### Fixed
- **`app/src/model/normalize.ts` did not match the training normalizer.** It scaled by mean wrist-to-landmark distance while `model/src/data/normalize.py` scales by mean consecutive bone length — a ~10x divergence that pushed every frame off the ONNX graph's training distribution. Verified to 1.9e-06 over 300 random trials.
- **`app/src/synthesis/stitch.ts` crossfade ramp** used `cos(pi*i/w)` against Python's `linspace(0, pi, w)`; the app ramp never reached 1.0. Now matches exactly.
- **Committed token-signing secret** (`backend/silentlink/deps.py`) removed. `python-dotenv` is now actually imported, so `.env` works; an unset `SILENTLINK_SECRET` generates an ephemeral key with a warning instead of falling back to a published literal.
- **`.env.example` described a fictional config** — ~15 variables no code read, and the one that mattered was missing.
- **`tests/test_api.py::test_datasets_listed` was not hermetic** — asserted on gitignored `data/processed`. Added a `processed_root` seam through `create_app`; the test now builds its own dataset and asserts exact counts.
- **422 errors rendered as `[object Object]`** — FastAPI returns `detail` as an array on validation failure; the API client now unwraps both shapes.
- **The floating back-bar occluded every sub-screen's title.** Added a `TOP_BAR_HEIGHT` token and inset the four sub-screens.
- **Dead code removed:** the Kotlin `HandLandmarkerPlugin` declared `frameCounter`/`lastPosePoints`/`lastFacePoints` and read `pose`/`face` params for inference that never existed; `HandDetectionResult` declared `pose`/`face` fields the plugin never sends. Also an unused `useCameraDevice` subscription in `GestureScreen`.
- `config/api.ts` no longer documents a nonexistent `app.json → expo.extra.backendUrl` mechanism.

### Verified
- `model` 46 passed, `backend` 15 passed, both ruff clean.
- All 11 glosses render mutually distinct templates (closest pair: hello/thanks at 0.074 mean abs difference — both flat-hand-from-the-face gestures, faithful to ISL).
- Backend synthesis end-to-end: `good morning` → 40 frames, `hello thank you` → 72, `yes no please` → 104.

## [App] — 2026-09-20 — Backend integration + fingerspelling + icon polish

### Added
- `app/src/api/client.ts` — minimal typed client for the backend (fetch-based, in-memory bearer token) + `app/src/config/api.ts` (base URL via `EXPO_PUBLIC_BACKEND_URL`, default `http://10.0.2.2:8000` for the Android emulator).
- `app/src/screens/LoginScreen.tsx` — signup/login against the backend (Group 2133 design).
- `app/src/screens/FingerspellingScreen.tsx` — camera + per-frame hand-shape letter recognition, builds a word with hold-to-commit + Space/Clear.
- `app/src/model/fingerspell.ts` — hand-shape feature extraction (per-finger curl + spread) + nearest-neighbour letter classifier (deterministic stub until a real letter model is trained).
- `app/src/components/Icon.tsx` — flat line icons (hand / text / keys) drawn with react-native-skia, replacing emoji in Home feature cards.

### Changed
- `app/App.tsx` — auth state (login/logout) + new `spell` and `login` routes; `HomeScreen` gained a Fingerspelling card and a Sign in/Log out chip.
- `app/src/screens/TextToSignScreen.tsx` — when signed in, also sends the phrase to the backend `/synthesis` and shows whether the cloud mapping matches on-device (offline-first animation unchanged).

### Verified
- `tsc --noEmit` clean (app); model 29 tests + ruff clean (unchanged from prior phase).

### Notes
- Backend integration is best-effort (graceful offline fallback); the phone/emulator must reach the backend IP. `fingerspell.ts` letter recognition is a placeholder until a real per-frame letter model is trained.

## [Backend] — 2026-09-20 — FastAPI Service (core)

### Added
- `silentlink/security.py` — PBKDF2 password hashing + HMAC-SHA256 signed bearer tokens (stdlib only, no extra runtime deps).
- `silentlink/db.py` — thread-safe in-memory repository (users, datasets, train jobs, model registry, synthesis jobs) with a `create_app(db)` factory so tests get an isolated DB.
- Routers (`silentlink/routers/`):
  - `auth.py` — signup / login / me (JWT-style bearer).
  - `datasets.py` — list + refresh captured datasets (scans `data/processed` on disk).
  - `jobs.py` — create/list/get training jobs; runs the real `scripts/train_model.py` in a background thread.
  - `models.py` — model registry (register/list trained models with accuracy/size).
  - `synthesis.py` — enqueue text→sign synthesis; reuses `model/src/synthesis` so on-device trajectories match.
- `silentlink/main.py` — `create_app()` app factory + `/health`, default `app = create_app()`.

### Changed
- `backend/pyproject.toml` — declared `silentlink.routers` + `silentlink.services` packages; added `httpx` to dev deps.

### Tests
- `tests/test_security.py` (5), `tests/test_auth.py` (4), `tests/test_api.py` (4), shared `conftest.py` fixtures. Total 15 passed, ruff clean.

### Notes
- Persistence is **in-memory** (per-process); swap in SQLite/Postgres repositories for durability. Auth is dependency-light but production hardening (refresh tokens, rate limiting) is a follow-up.

## [Phase 2] — 2026-09-20 — Text→Sign Synthesis Engine (core)

### Added
- `model/src/synthesis/` package:
  - `normalize.py` — rule-based sentence→gloss normalizer (longest-phrase-first match against `glosses.json` words; collapses duplicate adjacent glosses).
  - `motion.py` — `MotionLibrary` (gloss → `(T,42,3)` template) + deterministic synthetic template generator so the pipeline builds before real capture.
  - `stitch.py` — concatenative landmark stitcher with raised-cosine coarticulation crossfade at sign boundaries (`parts[-1]`-accumulated, correct for 3+ signs).
  - `export.py` — serializes the motion library to a compact bundled JSON asset.
- App-side TS mirrors: `app/src/synthesis/{normalize,motion,stitch}.ts` (on-device behavior matches training/synthesis).
- `scripts/sync_app_assets.py` now also generates `app/src/synthesis/motionLibrary.json` from the canonical glosses.
- `app/src/model/bones.ts` — hand skeleton topology (mirrors ST-GCN `_HAND_EDGES`) + joint→screen projection.
- `app/src/components/SkeletonAvatar.tsx` — react-native-skia animated two-hand skeleton (reanimated frame callback drives the trajectory; Skia `Path` rebuilt per frame).
- `app/src/screens/TextToSignScreen.tsx` — type a phrase → gloss chips → animated signed skeleton.
- `app/src/screens/HomeScreen.tsx` + `app/App.tsx` shell — Home (Gesture→Text / Text→Sign) + simple top-bar navigation.
- `app/src/theme.ts` — design tokens lifted from the "Group 2133" mockup (soft blue bg, mint/teal chips, coral primary, deep-teal ink).
- GestureScreen restyled to the same design system.

### Changed
- `app/package.json` — added `@shopify/react-native-skia` (via `npx expo install`, SDK 57 compatible).

### Fixed
- `model/src/synthesis/stitch.py` — stitcher now blends against the accumulated tail (`parts[-1]`) instead of the original template, fixing the seam for sequences of 3+ glosses.

### Tests
- `model/tests/test_synthesis.py` — 11 tests (normalizer matching/longest-phrase/collapse, motion shape, stitch single/empty/multi/continuity/three-gloss). Total model suite 29 passed, ruff clean.
- App: `tsc --noEmit` clean; `expo export --platform android` bundles (1284 modules).

### Notes / Blockers
- Motion library is **synthetic**; replace with real captured landmark motion (per-gloss templates) before production.
- On-device skeleton verification still needs the native build (`npx expo run:android`) on the user's machine.

## [Phase 1c] — 2026-09-20 — RN App: Gesture → Text

### Added
- Expo (SDK 57, TypeScript) app scaffolded in `app/` with `expo-dev-client` (native dev build; **not** Expo Go).
- Native inference stack (Android): `react-native-vision-camera` v4 + `react-native-worklets-core` + `expo-vision-camera-v4-mediapipe` + `onnxruntime-react-native`.
- `src/screens/GestureScreen.tsx` — live camera (front/back toggle), frame-processor landmark detection, 40-frame gesture buffer, recognized gloss + confidence display.
- `src/model/normalize.ts` — TS mirror of the Python normalization (left-hand-first packing, wrist-relative, scale-invariant).
- `src/model/classifier.ts` — loads bundled `model.onnx` via expo-asset + onnxruntime, runs ST-GCN.
- `src/hooks/useGestureRecognizer.ts` — buffers/normalizes frames, triggers inference, exposes result/progress/reset.
- `src/config/glosses.ts` — reads bundled `glosses.json`; class order matches training (`sorted(keys)`), no hardcoded vocab.
- `app.json` config plugins (vision-camera permission, mediapipe hand landmarker, onnxruntime); `metro.config.js` treats `.onnx`/`.task` as assets.
- `scripts/sync_app_assets.py` — copies `glosses.json`, `hand_landmarker.task`, and a trained `model.onnx` from `model/` into the app.
- `app/README.md` — build/run steps and architecture.

### Changed
- `model/src/models/export.py` — ONNX now exported **self-contained** (weights embedded, no external `.data` file) so it bundles as a single app asset (~1.5 MB for ST-GCN).
- `model/pyproject.toml` — added `onnx` to deps.

### Verified
- `expo prebuild --clean` succeeds — MediaPipe plugin generates `HandLandmarkerPlugin.kt`, adds `com.google.mediapipe:tasks-vision` to gradle, registers in `MainApplication.kt`, copies `hand_landmarker.task` to Android assets.
- `tsc --noEmit` clean; `expo export --platform android` bundles (1017 modules) with `model.onnx` as an asset.
- Self-contained ONNX re-verified through onnxruntime (`(1,40,42,3) -> (1,11)`).
- Model suite: `ruff` clean, 18 tests pass.

### Notes / Blockers
- Requires a **native dev build** (`npx expo run:android`) — no Android toolchain in this environment, so on-device verification is pending on the user's machine/emulator.
- Bundled `model.onnx` is a **synthetic smoke-test** model; replace with a model trained on real captured ISL data.
- iOS not wired (plugin is Android-only; needs macOS).

## [Phase 1b] — 2026-09-20 — Gesture Classification Models + Training Pipeline

### Added
- `model/src/models/` package:
  - `baseline.py` — Conv1D + BiLSTM baseline.
  - `stgcn.py` — ST-GCN (primary), with block-diagonal two-hand skeleton adjacency (21 landmarks/hand, wrist→finger graph).
  - `training.py` — train/val loop with **early stopping** on validation accuracy (replaces the old `epochs=2000`/`test_size=0.05` anti-pattern); saves `best_model.pt` + `metrics.json`.
  - `export.py` — ONNX export, fixed batch=1 (on-device / TFLite target), UTF-8 stream fix for cp1252 consoles.
- `model/src/data/loader.py` — in-memory `LandmarkDataset` with optional on-the-fly augmentation, stratified per-gloss train/val split, `DataLoader` glue.
- `scripts/train_model.py` — CLI: `--model {stgcn,baseline}`, epochs/lr/batch/val-frac/patience, trains + exports ONNX.
- `scripts/make_synthetic_data.py` — generates synthetic (T,42,3) sequences per gloss for pipeline smoke-testing (real `data/processed` is empty until webcam capture).
- CUDA-enabled PyTorch (2.14.0+cu130) + onnx/onnxruntime/onnxscript deps; verified on RTX 2050 (compute 8.6).

### Changed
- `model/pyproject.toml` deps extended with `torch`, `onnx`, `onnxruntime`, `onnxscript`.

### Tests
- `model/tests/test_models.py` — 9 new tests: forward shapes, graph adjacency structure, backprop, stratified split coverage, checkpointing, ONNX→onnxruntime inference. Total model suite now 18 tests.
- Verified: `ruff check .` clean (model + backend), `pytest` 18 passed (model) + 2 passed (backend).
- End-to-end verified on synthetic data on GPU: ST-GCN best val acc ~97.7%, baseline exports and runs through onnxruntime.

### Notes
- `data/processed/*` currently holds **synthetic** smoke data. Overwrite with real captured data before final training.

## [Phase 1a] — 2026-09-19 — Data Pipeline (capture + extraction + augmentation)

### Added
- `model/src/data/` package: `normalize.py`, `augment.py`, `dataset.py`, `capture.py`.
- Hand-only landmark pipeline producing `(T=40, V=42, C=3)` normalized sequences per the training contract.
- Landmark-space augmentation: 3D rotation, per-hand scale, left/right mirror, temporal jitter, noise.
- MediaPipe **Tasks API** (`HandLandmarker`) capture module — matches ADR-005 on-device stack.
- Dataset read/write: `save_sequence` / `discover_sequences` / `build_dataset` to `data/processed/<gloss>/<id>.npy`.
- Interactive CLI: `scripts/capture_data.py` (gloss-driven from `glosses.json`).
- Model package `pyproject.toml` (ruff + pytest, `pythonpath=src`).
- Downloaded `model/assets/hand_landmarker.task` (MediaPipe hand landmarker model, 7.8 MB).

### Tests
- `model/tests/test_data_pipeline.py` — 9 tests covering shape, packing, normalization scale-invariance, augmentation shape preservation, mirror hand-swap, and dataset round-trip.
- Verified: `ruff check .` clean, `pytest` 9 passed (model) + 2 passed (backend).
- End-to-end webcam capture verified (40-frame sequence saved; zeros when no hand in frame).

## [Phase 0] — 2026-09-19 — Project Scaffold

### Added
- Monorepo skeleton: `app/`, `backend/`, `model/`, `data/`, `docs/`, `scripts/`.
- Persistent agent memory files: `AGENTS.md`, `CHANGELOG.md`, `TODO.md`, `README.md`, `docs/ADR.md`.
- Canonical ISL vocabulary contract: `model/configs/glosses.json`.
- Backend scaffold: `pyproject.toml`, `ruff` config, minimal package + sample test.
- Environment/config standards: `.env.example`, `.gitignore`.

### Changed
- Established clean rebuild of legacy `silent_link/` toy into `silent-link_v1-prod/`. Legacy folders left untouched.

### Fixed
- Documented the `.env` path inconsistency from the legacy repo (`GIF_DIR=gifs` vs actual `ISL_Gifs`) — resolved via canonical config in this repo.
