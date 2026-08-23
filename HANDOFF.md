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
| axial | 4 (10/10 epochs, retrained 2026-08-23) | 0.893 | 0.922 | 0.758 | 0.858 |

All three planes are now trained for the full 10 epochs.

### Ensemble on the MRNet valid split (evaluate.py, run locally 2026-08-23)
| label | AUC | threshold | sensitivity | specificity |
|---|---|---|---|---|
| abnormal | **0.943** | 0.463 | 0.937 | 0.840 |
| acl | **0.960** | 0.465 | 0.963 | 0.864 |
| meniscus | **0.847** | 0.792 | 0.808 | 0.779 |

Reference (MRNet paper, 3-plane ensemble): abnormal 0.937, acl 0.965, meniscus 0.847.
We exceed the paper on abnormal, match it on meniscus, and sit 0.005 below on ACL. Mean 0.917 vs their 0.916.
Confirmed twice: once on Colab (cell 5) and once locally with `python -m ml.training.evaluate`.

Weights are now on the laptop in `ml/models/` (git-ignored), pulled with
`gdown --folder <shared Drive folder> -O ml/models`. Full suite passes with the real model (`pytest -q` -> 4 passed).
Verified on valid exam 1172 (ground truth abnormal=1, acl=1, meniscus=1): abnormal 92.8% POSITIVE,
acl 69.8% POSITIVE, meniscus 71.5% negative (below its 79.2% threshold) — 2 of 3, meniscus a false negative.
Weights live in Google Drive `MyDrive/knee-mri-detect/models` (+ `models.zip`); `*.pt` and `eval.json` are git-ignored.
Colab notebook (Drive copy, stable identity): https://colab.research.google.com/drive/1cZylO7shWCSbwjKxwkEGd77GPazZm8JC

### DONE — axial finished 2026-08-23. Kept below for reference only.
### Finish axial (~90 min, unattended once started)
Free GPU quota resets roughly 24 h after it was hit (2026-08-21 ~01:00). Then:
1. Open the Drive notebook (link below). Runtime -> Change runtime type -> T4 GPU.
2. Delete `axial.pt` and `axial_metrics.json` from `MyDrive/knee-mri-detect/models` (else cell 4 skips axial).
3. In cell 4 set `FRESH = False` (keeps sagittal/coronal, retrains only axial).
4. Run all. Click the Drive permission prompt PROMPTLY — it times out and fails with `ValueError: mount failed`.
5. Cell 5 writes eval.json (ensemble + tuned thresholds), cell 6 writes models.zip.

### Getting the weights to the laptop
The files already live in Drive at `MyDrive/knee-mri-detect/models/`. Simplest route needs no Colab: open
drive.google.com, right-click that folder, Download (Drive zips it), then `unzip -o ~/Downloads/<file>.zip -d ml/models/`.
Then locally: `pytest -q`, and `python -m ml.training.evaluate --config ml/training/config.yaml` (the real MRNet data
is already in data/raw/, so evaluation and the ensemble thresholds can be produced on the laptop, no GPU needed).

## State (2026-08-20)
- All three original commits pushed (scaffold, synthetic e2e, evaluate/thresholds/EDA).
- API: `GET /api/v1/exams` (history, per-exam `flagged` count using tuned thresholds), `planes` slice counts on exam detail,
  `GET /api/v1/exams/{id}/slice/{plane}/{i}` PNG, `gradcam_meta` = plane + best slice per label.
- Frontend redesigned (Apple HIG light): sidebar history, intake with client-side .npy shape preview, slice viewer
  (native range + radio segmented control), findings with threshold ticks, Grad-CAM jump-to-slice, PDF download.
- Docker: VERIFIED with a real `docker compose up --build` (2026-08-23). All three services healthy; frontend served by
  nginx on :5173, /api proxied to the api container, Postgres connected, weights bind-mounted at /models, thresholds
  read from eval.json. Full exam upload → inference → Grad-CAM → PDF exercised through nginx; predictions identical
  to the local run. Found and fixed a Docker-only bug: `grad-cam>=1.5` is unpinned, and 1.5.6+ made `targets` a
  required argument to `cam(...)`, so Grad-CAM silently failed inside the container (caught by the best-effort
  try/except). ml/explain/gradcam.py now passes `targets=[ClassifierOutputTarget(0)]`, which works on 1.5.5 and 1.5.7.

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
3. DICOM ingestion (pydicom) if non-MRNet data arrives — the app currently only accepts MRNet-style .npy stacks,
   so anyone without the dataset cannot try it themselves.
4. Optional: Impeccable v4.1.1 is available (`npx impeccable update`).
