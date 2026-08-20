"""Generate a tiny synthetic dataset in MRNet layout so the pipeline runs before real data arrives.
Positives get a bright blob so a model can actually learn something.
Usage: python scripts/make_fake_mrnet.py [out_dir] [n_train] [n_valid]
"""
import sys, numpy as np
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/MRNet-v1.0")
n_train, n_valid = (int(a) for a in (sys.argv[2:4] or (24, 8)))
rng = np.random.default_rng(0)
labels, planes = ["abnormal", "acl", "meniscus"], ["sagittal", "coronal", "axial"]

def exam(y):
    s = rng.integers(20, 31)
    x = rng.normal(80, 20, (s, 256, 256)).clip(0, 255)
    if y[0]:
        c = rng.integers(60, 196, 2); x[s // 2 - 3:s // 2 + 3, c[0]-20:c[0]+20, c[1]-20:c[1]+20] += 120
    return x.astype(np.uint8)

for split, n, start in [("train", n_train, 0), ("valid", n_valid, 1130)]:
    ys = {}
    for i in range(n):
        ab = rng.random() < 0.5
        y = [int(ab), int(ab and rng.random() < 0.5), int(ab and rng.random() < 0.5)]
        eid = f"{start + i:04d}"; ys[eid] = y
        for p in planes:
            d = out / split / p; d.mkdir(parents=True, exist_ok=True); np.save(d / f"{eid}.npy", exam(y))
    for j, l in enumerate(labels):
        (out / f"{split}-{l}.csv").write_text("".join(f"{e},{y[j]}\n" for e, y in ys.items()))
print(f"wrote {n_train}+{n_valid} exams to {out}")
