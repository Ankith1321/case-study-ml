from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TuSimpleSegmentationDataset(Dataset):
    def __init__(self, root="data/processed/tusimple", split="train", img_size=(360, 640)):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.split_file = self.root / "splits" / f"{split}.txt"
        self.img_h, self.img_w = img_size

        if not self.split_file.exists():
            raise FileNotFoundError(f"Missing split file: {self.split_file}")

        self.items = [line.strip() for line in self.split_file.open("r", encoding="utf-8") if line.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name = self.items[idx]
        img_path = self.img_dir / name
        mask_path = self.mask_dir / name

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # Resize for stable training
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)  # binary

        # To tensors: CHW for image, HW for mask
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # 3xHxW
        mask_t = torch.from_numpy(mask).unsqueeze(0)    # 1xHxW

        return img_t, mask_t
