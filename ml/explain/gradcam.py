"""Grad-CAM over the slice with the strongest activation for a given label."""
import cv2, numpy as np, torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def gradcam_overlay(model, x: torch.Tensor, label_idx: int) -> tuple[int, np.ndarray]:
    """x: (S,3,H,W). Returns (best_slice_idx, RGB overlay uint8)."""
    model.eval()
    with torch.no_grad():
        feats = model.backbone(x)                      # (S,F)
        scores = feats @ model.head.weight[label_idx]  # per-slice contribution
        s = int(scores.argmax())
    target_layer = model.backbone.conv_head
    cam = GradCAM(model=_SliceWrapper(model, label_idx), target_layers=[target_layer])
    # targets is required from grad-cam 1.5.6 on; _SliceWrapper already emits a single column, so index 0.
    heat = cam(input_tensor=x[s:s + 1], targets=[ClassifierOutputTarget(0)])[0]
    base = x[s, 0].cpu().numpy(); base = (base - base.min()) / (np.ptp(base) + 1e-6)
    return s, show_cam_on_image(np.repeat(base[..., None], 3, -1).astype(np.float32), heat, use_rgb=True)


class _SliceWrapper(torch.nn.Module):
    def __init__(self, m, i): super().__init__(); self.m, self.i = m, i
    def forward(self, x): return self.m.head(self.m.backbone(x))[:, self.i:self.i + 1]
