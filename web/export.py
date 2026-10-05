"""
Export the three per-plane models for the browser demo in web/.

    python web/export.py                       # writes web/models/<plane>.onnx and web/config.json
    python web/export.py --check-case 1172     # also checks ONNX vs PyTorch on a local MRNet valid exam

Each ONNX model takes the preprocessed stack (S,3,224,224) and returns:
  logits (3,)    abnormal / acl / meniscus, max-pooled over slices (same as KneeMRINet)
  best   (3,)    per label, the slice with the strongest contribution (same rule as ml/explain/gradcam.py)
  cam    (3,7,7) Grad-CAM on conv_head for that slice, computed in closed form: the head is
                 BN+SiLU -> global average pool -> linear, so d(logit)/d(conv_head) has an exact
                 formula and no backward pass is needed. Checked against pytorch_grad_cam below.
No MRNet data is written anywhere under web/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.training.model import KneeMRINet  # noqa: E402

LABELS = ["abnormal", "acl", "meniscus"]
PLANES = ["sagittal", "coronal", "axial"]
OUT = ROOT / "web" / "models"


class Export(nn.Module):
    def __init__(self, m: KneeMRINet):
        super().__init__()
        self.b, self.head = m.backbone, m.head

    def forward(self, x):
        b = self.b
        z = b.conv_head(b.blocks(b.bn1(b.conv_stem(x))))            # (S,K,7,7)  Grad-CAM target layer
        bn = b.bn2
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)      # (K,)
        y = (z - bn.running_mean[None, :, None, None]) * scale[None, :, None, None] + bn.bias[None, :, None, None]
        a = y * torch.sigmoid(y)                                     # SiLU
        feats = a.mean(dim=(2, 3))                                   # (S,K) global average pool
        logits = self.head(feats.max(dim=0).values)                  # (3,)
        scores = feats @ self.head.weight.T                          # (S,3) per-slice contribution
        best = scores.argmax(dim=0)                                  # (3,)

        hw = z.shape[2] * z.shape[3]
        sig = torch.sigmoid(y)
        dsilu = sig * (1 + y * (1 - sig))                            # d SiLU / dy
        zb, db = z[best], dsilu[best]                                # (3,K,7,7)
        grads = self.head.weight[:, :, None, None] * scale[None, :, None, None] * db / hw
        weights = grads.mean(dim=(2, 3), keepdim=True)               # Grad-CAM channel weights
        cam = torch.relu((weights * zb).sum(dim=1))                  # (3,7,7)
        return logits, best, cam


def load(plane):
    m = KneeMRINet(len(LABELS), "efficientnet_b3", pretrained=False)
    m.load_state_dict(torch.load(ROOT / "ml/models" / f"{plane}.pt", map_location="cpu", weights_only=True))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-case", default=None, help="MRNet valid exam id to check parity on (local data only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for plane in PLANES:
        exp = Export(load(plane)).eval()
        torch.onnx.export(exp, (torch.zeros(4, 3, 224, 224),), str(OUT / f"{plane}.onnx"),
                          input_names=["x"], output_names=["logits", "best", "cam"],
                          dynamic_axes={"x": {0: "slices"}}, opset_version=17, dynamo=False)
        print(f"{plane}.onnx {(OUT / f'{plane}.onnx').stat().st_size / 1e6:.1f} MB")

    ev = json.load(open(ROOT / "ml/models/eval.json"))
    cfg = {"labels": LABELS, "planes": PLANES,
           "thresholds": {l: round(ev["ensemble"][l]["threshold"], 4) for l in LABELS},
           "auc": {k: {l: round(ev[k][l]["auc"], 3) for l in LABELS} for k in PLANES + ["ensemble"]}}
    (ROOT / "web" / "config.json").write_text(json.dumps(cfg, indent=1))

    if args.check_case:
        check(args.check_case)


def check(case):
    import onnxruntime as ort
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from ml.data.transforms import preprocess_stack
    from ml.explain.gradcam import _SliceWrapper

    for plane in PLANES:
        stack = np.load(ROOT / "data/raw/MRNet-v1.0/valid" / plane / f"{case}.npy")
        x = preprocess_stack(stack)
        m = load(plane)
        with torch.no_grad():
            ref = torch.sigmoid(m(x)).numpy()
            feats = m.backbone(x)
        sess = ort.InferenceSession(str(OUT / f"{plane}.onnx"))
        logits, best, cam = sess.run(None, {"x": x.numpy()})
        prob = 1 / (1 + np.exp(-logits))
        ref_best = [int((feats @ m.head.weight[i]).argmax()) for i in range(3)]
        cam_err = 0.0
        for i in range(3):
            gc = GradCAM(model=_SliceWrapper(m, i), target_layers=[m.backbone.conv_head])
            s = ref_best[i]
            gc.activations_and_grads(x[s:s + 1])
            m.zero_grad()
            out = gc.activations_and_grads(x[s:s + 1])
            out[:, 0].sum().backward()
            act = gc.activations_and_grads.activations[0][0].numpy()
            grad = gc.activations_and_grads.gradients[0][0].numpy()
            ref_cam = np.maximum((grad.mean(axis=(1, 2))[:, None, None] * act).sum(0), 0)
            cam_err = max(cam_err, float(np.abs(ref_cam - cam[i]).max() / (np.abs(ref_cam).max() + 1e-9)))
            gc.activations_and_grads.release()
        print(f"{plane}: S={len(stack)} max|prob diff|={np.abs(prob - ref).max():.2e} "
              f"best slices match={list(best) == ref_best} max relative CAM diff={cam_err:.2e}")


if __name__ == "__main__":
    main()
