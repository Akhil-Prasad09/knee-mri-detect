# Handoff

Repo: https://github.com/Akhil-Prasad09/knee-mri-detect (local: Major Project/knee-mri-detect — the repo *is* this directory now)
Plan: PLAN.md · Product context: PRODUCT.md · Stack: PyTorch + EfficientNet, Grad-CAM, FastAPI, React/Vite, ReportLab, Docker.

## State (2026-08-20)
- All three original commits pushed (scaffold, synthetic e2e, evaluate/thresholds/EDA).
- API: `GET /api/v1/exams` (history, per-exam `flagged` count using tuned thresholds), `planes` slice counts on exam detail,
  `GET /api/v1/exams/{id}/slice/{plane}/{i}` PNG, `gradcam_meta` = plane + best slice per label.
- Frontend redesigned (Apple HIG light): sidebar history, intake with client-side .npy shape preview, slice viewer
  (native range + radio segmented control), findings with threshold ticks, Grad-CAM jump-to-slice, PDF download.
- Docker: nginx proxies /api → api; psycopg2-binary added; `TRAIN_CONFIG` env passthrough. `docker compose config` validates; full `up --build` not run yet.

## Verified
- `TRAIN_CONFIG=ml/training/smoke.yaml pytest -q` → 4 passed (after `python scripts/make_fake_mrnet.py` + smoke train).
- `cd frontend && npm run build` OK; desktop + 390px checked in Chrome against the live API.
- Not visually verified: the in-progress ("Analysing") findings state — CPU inference finishes in ~5 s, too fast to catch.

## Conventions
- ponytail: minimal code, stdlib/native first. superpowers: TDD, verify before claiming done.
- Labels/backbone from TRAIN_CONFIG; thresholds from ml/models/eval.json. Never hardcode labels in the UI (NAMES map is display-only).
- Local SQLite `knee.db` is disposable; schema changes = delete it (no migrations).

## Next
1. **Train on Colab — follow docs/COLAB_RUNBOOK.md step by step** (notebook: notebooks/train_colab.ipynb). MRNet is already
   downloaded locally (Kaggle mirror cjinny/mrnet-v1) for local evaluate/tests; local training was abandoned (16 GB laptop swaps).
2. `docker compose up --build` end-to-end once on a machine with disk for the torch image.
3. DICOM ingestion (pydicom) if non-MRNet data arrives.
4. Optional: Impeccable v4.1.1 is available (`npx impeccable update`).
