"""Train-time ultrasound augmentations (albumentations)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def build_train_transform(
    img_size: int = 512,
    strength: Literal["mild", "fasc", "console"] = "mild",
) -> Any:
    """Augs that preserve apo/fasc topology.

    ``fasc`` = stronger contrast/CLAHE + slightly larger geometry jitter.
    ``console`` = fasc + anisotropic affine for console-domain robustness.
    """
    import albumentations as A

    if strength in ("fasc", "console"):
        geo = [
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.25, p=0.6
            ),
            A.RandomGamma(gamma_limit=(75, 130), p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.25),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=12,
                border_mode=0,
                p=0.5,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.15, border_mode=0, p=0.2),
        ]
        if strength == "console":
            geo.append(
                A.Affine(
                    scale={"x": (0.85, 1.2), "y": (0.85, 1.2)},
                    fit_output=False,
                    mode=0,
                    p=0.35,
                )
            )
        return A.Compose(geo)

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.05,
                rotate_limit=8,
                border_mode=0,
                p=0.4,
            ),
        ]
    )


def paste_console_chrome(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Paste synthetic side rulers / colour bars around a B-mode crop."""
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    pad = int(rng.integers(40, 120))
    canvas = np.zeros((h + pad, w + 2 * pad), dtype=image.dtype)
    canvas[0:h, pad : pad + w] = image
    pitch = float(rng.uniform(8, 25))
    for y in np.arange(5, h, pitch):
        yy = int(y)
        length = int(rng.integers(6, 18))
        canvas[yy : yy + 2, pad - length : pad] = 220
    bar = np.linspace(0, 255, h).astype(np.uint8)[:, None]
    canvas[:h, pad + w : pad + w + min(12, pad)] = bar
    tx0 = pad + w // 4
    canvas[h : h + min(20, pad), tx0 : tx0 + 80] = int(rng.integers(180, 240))
    return canvas
