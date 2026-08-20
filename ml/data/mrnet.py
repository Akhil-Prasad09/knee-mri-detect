"""MRNet dataset loader. Each exam is a .npy stack of shape (S, 256, 256)."""
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from .transforms import preprocess_stack


class MRNetDataset(Dataset):
    def __init__(self, root, split, plane, labels, img_size=224, train=False):
        self.root, self.plane, self.labels = Path(root), plane, labels
        self.img_size, self.train = img_size, train
        frames = [pd.read_csv(self.root / f"{split}-{l}.csv", header=None, names=["id", l], dtype={"id": str}) for l in labels]
        df = frames[0]
        for f in frames[1:]:
            df = df.merge(f, on="id")
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        split = "train" if (self.root / "train" / self.plane / f"{row.id}.npy").exists() else "valid"
        stack = np.load(self.root / split / self.plane / f"{row.id}.npy")
        x = preprocess_stack(stack, self.img_size, augment=self.train)   # (S,3,H,W)
        y = torch.tensor(row[self.labels].values.astype(np.float32))
        return x, y
