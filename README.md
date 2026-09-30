# 🤟 Silent Link — v1-prod

Offline-first **Indian Sign Language (ISL)** communication app, rebuilt properly.

The legacy prototype (`../silent_link/`, `../ActionDetectionforSignLanguage/`) is a toy: an overfit 3-layer LSTM on 3 hand-picked words + a 90-GIF dictionary for text→sign. **v1-prod** rebuilds it as a real system:

- **Gesture → Text:** camera → MediaPipe Hand Landmarker → ST-GCN / baseline model → ISL gloss + text.
- **Text/Speech → Sign:** sentence→gloss normalization + **concatenative landmark synthesis** rendered as an animated skeleton (replacing the GIF dictionary).
- **ISL only.** Offline-first, on-device inference. Cloud only for training.

## Repo Layout

```
app/            React Native (Expo) — Phase 1+
backend/        FastAPI service — Phase 1+
model/          PyTorch training, configs, datasets — Phase 1+
data/           raw/ + processed/ (keypoints, shards)
docs/           ADR.md, ARCHITECTURE.md
scripts/        automation
```

## Persistent Memory

- `AGENTS.md` — how to work here (read first)
- `TODO.md` — current/upcoming work
- `CHANGELOG.md` — released changes
- `docs/ADR.md` — architecture decisions

## Status

- **Phase 0 (scaffold):** complete.
- **Phase 1+ :** pending user confirmation (see `TODO.md`).

## Getting Started

Python (backend): `cd backend && pip install -e ".[dev]"` then `ruff check . && pytest`.
RN app and model training arrive in Phase 1.
