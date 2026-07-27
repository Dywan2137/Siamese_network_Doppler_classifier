"""
1. Load M9 Bayonet renders
2. Crop to the blade
3. Split into train/test.

"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import config as cfg


def blade_crop(img):
    """
    Crop each render to the fixed blade.
    """
    w, h = img.size
    return img.crop((cfg.CROP_LEFT, cfg.CROP_TOP, w, h - cfg.CROP_BOTTOM_REMOVE))


# Crop -> resize -> tensor. Defined once and reused for train and test.
transform = transforms.Compose([
    transforms.Lambda(blade_crop),
    transforms.Resize((cfg.IMG_H, cfg.IMG_W)),
    transforms.ToTensor(),
])


class DopplerDataset(Dataset):
    """Every image whose (finish, phase) folder is a known label."""

    def __init__(self, root, transform, labels, max_per_class=None):
        self.transform = transform
        self.samples = []
        per_class = {}

        for leaf in sorted(Path(root).rglob("*")):
            if not leaf.is_dir():
                continue
            parts = leaf.relative_to(root).parts
            if len(parts) != 2:
                continue
            key = (parts[0], parts[1])
            if key not in labels:
                continue

            label = labels[key]
            for p in sorted(leaf.glob("*.png")):
                if max_per_class and per_class.get(label, 0) >= max_per_class:
                    break
                self.samples.append((p, label))
                per_class[label] = per_class.get(label, 0) + 1

        n_classes = len({s[1] for s in self.samples})
        print(f"Loaded {len(self.samples)} images across {n_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_split():

    full = DopplerDataset(cfg.DATASET_ROOT, transform, cfg.LABELS,
                          max_per_class=cfg.MAX_PER_CLASS)

    rng = np.random.default_rng(cfg.SEED)
    train_idx, test_idx = [], []
    for c in range(cfg.NUM_CLASSES):
        idx = [i for i, s in enumerate(full.samples) if s[1] == c]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * cfg.TEST_FRAC))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_ds = Subset(full, train_idx)
    test_ds = Subset(full, test_idx)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE,
                             shuffle=False, num_workers=0)
    return train_ds, test_ds, train_loader, test_loader
