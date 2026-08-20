import os
LABELS = ["abnormal", "acl", "meniscus"]
PLANES = ["sagittal", "coronal", "axial"]
MODEL_DIR = os.getenv("MODEL_DIR", "ml/models")
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./knee.db")
THRESHOLD = 0.5
