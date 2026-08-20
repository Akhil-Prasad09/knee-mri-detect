"""Loads per-plane models lazily and averages probabilities across available planes."""
from pathlib import Path
import numpy as np, torch
from ml.training.model import KneeMRINet
from ml.data.transforms import preprocess_stack
from ..core.config import LABELS, PLANES, MODEL_DIR, BACKBONE

_models: dict[str, KneeMRINet] = {}


def get_model(plane: str) -> KneeMRINet | None:
    if plane not in _models:
        w = Path(MODEL_DIR) / f"{plane}.pt"
        if not w.exists():
            return None
        m = KneeMRINet(len(LABELS), BACKBONE, pretrained=False); m.load_state_dict(torch.load(w, map_location="cpu")); m.eval()
        _models[plane] = m
    return _models[plane]


def predict(stacks: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """stacks: {plane: (S,H,W) array}. Returns ({label: prob}, {plane: preprocessed tensor})."""
    probs, tensors = [], {}
    for plane, stack in stacks.items():
        model = get_model(plane)
        if model is None:
            continue
        x = preprocess_stack(stack); tensors[plane] = x
        with torch.no_grad():
            probs.append(torch.sigmoid(model(x)).numpy())
    if not probs:
        raise RuntimeError("No trained model weights found in MODEL_DIR")
    return dict(zip(LABELS, np.mean(probs, axis=0).round(4).tolist())), tensors
