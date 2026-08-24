import os
from pathlib import Path
import yaml
_cfg = yaml.safe_load(open(os.getenv("TRAIN_CONFIG", "ml/training/config.yaml")))
LABELS = _cfg["labels"]
BACKBONE = _cfg["backbone"]
PLANES = ["sagittal", "coronal", "axial"]
MODEL_DIR = os.getenv("MODEL_DIR", "ml/models")
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
SAMPLES_DIR = os.getenv("SAMPLES_DIR", "samples")   # bundled demo exams: samples/<id>/<plane>.npy
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./knee.db")
import json
_ev = Path(MODEL_DIR) / "eval.json"
_src = json.load(open(_ev)).get("ensemble") if _ev.exists() else None
if _src is None and _ev.exists():
    _src = next(iter(json.load(open(_ev)).values()))
THRESHOLDS = {l: (_src[l]["threshold"] if _src else 0.5) for l in LABELS}  # tuned in ml/training/evaluate.py
