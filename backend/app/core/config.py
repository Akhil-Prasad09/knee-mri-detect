import os
import yaml
_cfg = yaml.safe_load(open(os.getenv("TRAIN_CONFIG", "ml/training/config.yaml")))
LABELS = _cfg["labels"]
BACKBONE = _cfg["backbone"]
PLANES = ["sagittal", "coronal", "axial"]
MODEL_DIR = os.getenv("MODEL_DIR", "ml/models")
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./knee.db")
THRESHOLD = 0.5
