from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def stem_key(path: Path) -> str:
    return path.stem


def pair_images_masks(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    imgs = {stem_key(p): p for p in list_images(img_dir)}
    masks = {stem_key(p): p for p in list_images(mask_dir)}
    keys = sorted(set(imgs) & set(masks))
    return [(imgs[k], masks[k]) for k in keys]


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_mask(path: Path) -> np.ndarray:
    mask = read_gray(path)
    return (mask > 0).astype(np.uint8)


class UltrasoundSegDataset(Dataset):
    """Paired ultrasound image + binary mask dataset."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        img_size: int = 512,
        transform: Callable | None = None,
    ) -> None:
        self.pairs = pairs
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path, mask_path = self.pairs[idx]
        image = read_gray(img_path)
        mask = read_mask(mask_path)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # 3-channel for ImageNet encoders
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return {"image": image_t, "mask": mask_t, "id": stem_key(img_path)}
