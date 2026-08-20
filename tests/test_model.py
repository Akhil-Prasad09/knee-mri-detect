import torch
from ml.training.model import KneeMRINet


def test_forward_shape():
    m = KneeMRINet(3, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(4, 3, 224, 224)).shape == (3,)
