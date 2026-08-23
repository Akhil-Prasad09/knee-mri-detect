"""Evaluate per-plane models + plane-averaged ensemble on the valid split.

Writes ml/models/eval.json per label with:
  auc                      threshold-free, unbiased
  threshold                best-F1 point over all 120 exams — this is what the API ships
  sensitivity/specificity  scored at that threshold on the SAME exams it was tuned on, so
                           optimistically biased; kept because the API needs the operating point
  cv_sensitivity/...       honest generalisation estimate: 5-fold, threshold picked on 4/5 and
                           scored on the held-out 1/5, averaged. Quote these in write-ups.

Usage: python -m ml.training.evaluate --config ml/training/config.yaml
"""
import argparse, json, yaml, torch, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
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


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[int(np.argmax(f1[:-1]))])


def at_threshold(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"f1": 2 * prec * sens / max(prec + sens, 1e-9), "sensitivity": sens, "specificity": spec}


def metrics(y, p, folds=5, seed=0):
    t = best_threshold(y, p)
    m = {"auc": float(roc_auc_score(y, p)), "threshold": t, **at_threshold(y, p, t)}
    # Tuning and scoring on the same exams inflates sens/spec. Re-estimate with the threshold
    # chosen on 4/5 of the split and scored on the 1/5 it never saw.
    cv = [at_threshold(y[te], p[te], best_threshold(y[tr], p[tr]))
          for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(p[:, None], y)]
    m.update({f"cv_{k}": float(np.mean([f[k] for f in cv])) for k in cv[0]})
    return m


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
        print(k, {l: f"auc={m['auc']:.3f} sens={m['sensitivity']:.2f}/{m['cv_sensitivity']:.2f} spec={m['specificity']:.2f}/{m['cv_specificity']:.2f} (in-sample/cv)" for l, m in v.items()})
    json.dump(out, open(Path(cfg["out_dir"]) / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ml/training/config.yaml")
    main(yaml.safe_load(open(ap.parse_args().config)))
