import numpy as np
from ml.data.transforms import preprocess_stack


def test_preprocess_shape_and_range():
    x = preprocess_stack(np.random.rand(5, 256, 256) * 4000, size=224)
    assert x.shape == (5, 3, 224, 224)
    assert np.isfinite(x.numpy()).all()
