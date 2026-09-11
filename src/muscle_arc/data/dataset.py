from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    """Collect image files under ``folder`` (recursive; Kaggle uses nested dirs)."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


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


def letterbox(
    image: np.ndarray,
    size: int,
    is_mask: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Aspect-preserving resize into a square canvas (pad with zeros).

    Returns (canvas, meta) where meta has scale, pad_left, pad_top, new_h, new_w
    needed to map predictions back to the original resolution.
    """
    h, w = image.shape[:2]
    scale = float(size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((size, size), dtype=image.dtype)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    meta = {
        "scale": scale,
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "new_h": int(new_h),
        "new_w": int(new_w),
        "orig_h": int(h),
        "orig_w": int(w),
    }
    return canvas, meta


def unletterbox(
    pred: np.ndarray,
    meta: dict[str, float | int],
) -> np.ndarray:
    """Crop letterbox padding and resize prediction back to original HxW."""
    pad_left = int(meta["pad_left"])
    pad_top = int(meta["pad_top"])
    new_h = int(meta["new_h"])
    new_w = int(meta["new_w"])
    orig_h = int(meta["orig_h"])
    orig_w = int(meta["orig_w"])
    crop = pred[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    return cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


class UltrasoundSegDataset(Dataset):
    """Paired ultrasound image + binary mask dataset."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        img_size: int = 512,
        transform: Callable | None = None,
        letterbox_resize: bool = True,
    ) -> None:
        self.pairs = pairs
        self.img_size = img_size
        self.transform = transform
        self.letterbox_resize = letterbox_resize

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path, mask_path = self.pairs[idx]
        image = read_gray(img_path)
        mask = read_mask(mask_path)

        if self.letterbox_resize:
            # Align mask FOV to image native size first (host masks often differ
            # in HxW/aspect from training images). Then letterbox both together.
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            assert mask.shape[:2] == image.shape[:2], (
                f"mask/image shape mismatch after align: {mask.shape} vs {image.shape}"
            )
            image, meta = letterbox(image, self.img_size, is_mask=False)
            mask, _ = letterbox(mask, self.img_size, is_mask=True)
            # Same canvas geometry
            assert image.shape[:2] == mask.shape[:2]
        else:
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
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
