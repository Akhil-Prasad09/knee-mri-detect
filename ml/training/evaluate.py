"""Evaluate per-plane models + plane-averaged ensemble on the valid split.
Writes ml/models/eval.json with AUC, and sens/spec/F1 at the best-F1 threshold per label.
Usage: python -m ml.training.evaluate --config ml/training/config.yaml
"""
import argparse, json, yaml, torch, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from tqdm import tqdm
from ml.data.mrnet import MRNetDataset
from ml.training.model import KneeMRINet


def plane_probs(cfg, plane, device):
    w = Path(cfg["out_dir"]) / f"{plane}.pt"
    if not w.exists():
        return None, None
    m = KneeMRINet(len(cfg["labels"]), cfg["backbone"], pretrained=False); m.load_state_dict(torch.load(w, map_location=device)); m.to(device).eval()
    ds = MRNetDataset(cfg["data_dir"], "valid", plane, cfg["labels"], cfg["img_size"])
    ys, ps = [], []
    with torch.no_grad():
        for x, y in tqdm(ds, desc=plane, leave=False):
            ps.append(torch.sigmoid(m(x.to(device))).cpu().numpy()); ys.append(y.numpy())
    return np.array(ys), np.array(ps)


def metrics(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9); i = int(np.argmax(f1[:-1]))
    t = float(thr[i]); tn, fp, fn, tp = confusion_matrix(y, p >= t).ravel()
    return {"auc": float(roc_auc_score(y, p)), "threshold": t, "f1": float(f1[i]),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1)}


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"; out, per_plane = {}, {}
    for plane in cfg["planes"]:
        y, p = plane_probs(cfg, plane, device)
        if p is None:
            print(f"skip {plane}: no weights"); continue
        per_plane[plane] = p; out[plane] = {l: metrics(y[:, i], p[:, i]) for i, l in enumerate(cfg["labels"])}
    if len(per_plane) > 1:
        p = np.mean(list(per_plane.values()), axis=0)
        out["ensemble"] = {l: metrics(y[:, i], p[:, i]) for i, l in enumerate(cfg["labels"])}
    for k, v in out.items():
        print(k, {l: f"auc={m['auc']:.3f} f1={m['f1']:.2f}@{m['threshold']:.2f}" for l, m in v.items()})
    json.dump(out, open(Path(cfg["out_dir"]) / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ml/training/config.yaml")
    main(yaml.safe_load(open(ap.parse_args().config)))
