"""Scale detectors: legacy border strips + native-resolution crop CNN.

Native crops preserve pixel pitch (resizing strips destroys mm/px signal).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MM_LO = 0.025
MM_HI = 0.12


def extract_border_strips(
    gray: "np.ndarray",
    strip: int = 48,
    out_h: int = 256,
    out_w: int = 48,
) -> "np.ndarray":
    """Legacy 4-strip tensor (kept for old checkpoints)."""
    import cv2
    import numpy as np

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    s = max(8, min(strip, h // 4, w // 4))
    left = gray[:, :s]
    right = gray[:, w - s :]
    top = gray[:s, :].T
    bottom = gray[h - s :, :].T
    strips = []
    for region in (left, right, top, bottom):
        r = cv2.resize(region, (out_w, out_h), interpolation=cv2.INTER_AREA)
        strips.append(r.astype(np.float32) / 255.0)
    return np.stack(strips, axis=0)


def extract_native_crop(
    gray: "np.ndarray",
    size: int = 256,
    rng: "np.random.Generator | None" = None,
) -> "np.ndarray":
    """Random (or center) native-resolution crop — no resize of the patch content."""
    import numpy as np

    if gray.ndim == 3:
        import cv2

        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < size or w < size:
        # Pad then take corner — still no downscale of content
        canvas = np.zeros((max(h, size), max(w, size)), dtype=gray.dtype)
        canvas[:h, :w] = gray
        gray = canvas
        h, w = gray.shape[:2]
    rng = rng or np.random.default_rng()
    y0 = int(rng.integers(0, h - size + 1))
    x0 = int(rng.integers(0, w - size + 1))
    crop = gray[y0 : y0 + size, x0 : x0 + size].astype(np.float32) / 255.0
    return crop[None]  # (1,H,W)


class ScaleStripDetector(nn.Module):
    """Legacy: 4 border strips → mm_per_pixel."""

    def __init__(self, in_ch: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        raw = self.head(h).squeeze(-1)
        return MM_LO + (MM_HI - MM_LO) * torch.sigmoid(raw)


class ScaleNativeCropDetector(nn.Module):
    """Native 256 crop → log(mm/px) mapped to band. Preserves pitch."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        # Also discrete depth-class logits (9 bins 3.0..7.0) as auxiliary
        self.depth_head = nn.Linear(128 * 16, 9)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        flat = feat.flatten(1)
        raw = self.head(flat).squeeze(-1)
        mm = MM_LO + (MM_HI - MM_LO) * torch.sigmoid(raw)
        depth_logits = self.depth_head(flat)
        return mm, depth_logits


def huber_mm_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)


def log_mm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(torch.log(pred.clamp_min(1e-4)), torch.log(target.clamp_min(1e-4)), delta=0.05)


def load_scale_detector(path: Path | str, device: torch.device | None = None) -> nn.Module:
    device = device or torch.device("cpu")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    kind = "strip"
    if isinstance(payload, dict):
        kind = payload.get("kind", "strip")
        state = payload["model"] if "model" in payload else payload
    else:
        state = payload
    if kind == "native":
        model: nn.Module = ScaleNativeCropDetector().to(device)
    else:
        model = ScaleStripDetector().to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_mm_per_pixel(
    model: nn.Module,
    gray: "np.ndarray",
    device: torch.device | None = None,
) -> float:
    device = device or next(model.parameters()).device
    if isinstance(model, ScaleNativeCropDetector):
        import numpy as np

        crops = [extract_native_crop(gray, rng=np.random.default_rng(i)) for i in range(4)]
        x = torch.from_numpy(np.stack(crops, axis=0)).to(device)
        mm, _ = model(x)
        return float(mm.mean().item())
    x = torch.from_numpy(extract_border_strips(gray)).unsqueeze(0).to(device)
    return float(model(x).item())
