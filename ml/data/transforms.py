"""Preprocessing shared by training and the API: resize -> CLAHE -> normalise."""
import cv2, numpy as np, torch

MEAN, STD = 0.485, 0.229  # ImageNet grey approximations
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_slice(img: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(img.astype(np.float32), (size, size))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = _clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img.astype(np.float32) / 255.0


def preprocess_stack(stack: np.ndarray, size: int = 224, augment: bool = False) -> torch.Tensor:
    slices = np.stack([preprocess_slice(s, size) for s in stack])  # (S,H,W)
    if augment:
        if np.random.rand() < 0.5:
            slices = slices[:, :, ::-1]
        shift = np.random.randint(-12, 13, size=2)
        slices = np.roll(slices, shift, axis=(1, 2))
    slices = (slices - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(slices)).unsqueeze(1)  # (S,1,H,W)
    return x.repeat(1, 3, 1, 1)
