# knee-mri-detect

Automated Knee MRI Abnormality Detection using Deep Learning for Clinical Decision Support.
B.Tech Major Project — see [PLAN.md](PLAN.md) for the full build plan and milestones.

**Stack:** PyTorch + EfficientNet-B3 · Grad-CAM · FastAPI · React (Vite) · SQLite/PostgreSQL · ReportLab · Docker

## Quick start

```bash
make setup          # create venv + install python deps
make api            # http://localhost:8000/docs
cd frontend && npm install && npm run dev   # http://localhost:5173
make test
```

Training (Colab or local GPU):

```bash
python -m ml.training.train --config ml/training/config.yaml
```

Place the MRNet dataset under `data/raw/MRNet-v1.0/` (request access from Stanford AIMI).

> Research / decision-support use only. Not a medical device.

## Run the whole pipeline without the real dataset

```bash
python scripts/make_fake_mrnet.py                      # synthetic exams in MRNet layout
python -m ml.training.train --config ml/training/smoke.yaml --plane sagittal   # ~2 min on CPU
TRAIN_CONFIG=ml/training/smoke.yaml make api           # then upload any .npy stack at /docs
```
Swap back to `ml/training/config.yaml` (EfficientNet-B3, pretrained) once MRNet is in `data/raw/`.
