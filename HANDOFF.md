# Handoff

Repo: https://github.com/Akhil-Prasad09/knee-mri-detect (local: Major Project/knee-mri-detect — the repo *is* this directory now)
Plan: PLAN.md · Product context: PRODUCT.md · Stack: PyTorch + EfficientNet, Grad-CAM, FastAPI, React/Vite, ReportLab, Docker.

## Real-data training (2026-08-20 evening, Colab T4)
Dataset: Kaggle mirror `cjinny/mrnet-v1` = MRNet v1.0 (1130 train / 120 valid x 3 planes). Also downloaded locally to
`data/raw/MRNet-v1.0/` (git-ignored) so evaluate.py and the e2e test can run on the laptop.
EfficientNet-B3, 10 epochs/plane (see docs/COLAB_RUNBOOK.md for why not 20). Best-mean-AUC checkpoint per plane.

| plane | best epoch | abnormal | acl | meniscus | mean |
|---|---|---|---|---|---|
| sagittal | 9 | 0.929 | 0.936 | 0.773 | 0.879 |
| coronal | 2 | 0.907 | 0.926 | 0.853 | 0.895 |
| axial | running | | | | |

Reference (MRNet paper, 3-plane ensemble): abnormal 0.937, acl 0.965, meniscus 0.847.
Weights live in Google Drive `MyDrive/knee-mri-detect/models` (+ `models.zip`); `*.pt` and `eval.json` are git-ignored.
Colab notebook (Drive copy, stable identity): https://colab.research.google.com/drive/1cZylO7shWCSbwjKxwkEGd77GPazZm8JC

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
