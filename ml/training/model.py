"""EfficientNet-B3 slice encoder + max-pool over slices -> multi-label head."""
import timm, torch, torch.nn as nn


class KneeMRINet(nn.Module):
    def __init__(self, n_labels: int, backbone: str = "efficientnet_b3", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, n_labels)

    def forward(self, x):               # x: (S,3,H,W) — one exam
        feats = self.backbone(x)        # (S,F)
        pooled = feats.max(dim=0).values  # (F,)
        return self.head(pooled)        # (n_labels,) logits
