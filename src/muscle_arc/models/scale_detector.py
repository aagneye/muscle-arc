"""Trainable border-strip scale detector (mm per pixel).

Learns from heuristic OCR/tick / OSF pseudo-labels so we can retrain whenever
labels improve. Input is four edge strips stacked as a 4-channel tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Match depth_scale ultrasound band
MM_LO = 0.025
MM_HI = 0.12


def extract_border_strips(
    gray: "np.ndarray",
    strip: int = 48,
    out_h: int = 256,
    out_w: int = 48,
) -> "np.ndarray":
    """Return float32 array (4, out_h, out_w) — L,R,T,B strips, normalized 0–1."""
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


class ScaleStripDetector(nn.Module):
    """Small CNN: 4 border strips → mm_per_pixel in [MM_LO, MM_HI]."""

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
        """Return mm_per_pixel, shape (N,)."""
        h = self.backbone(x)
        raw = self.head(h).squeeze(-1)
        # Map unbounded logit → (MM_LO, MM_HI) via sigmoid
        return MM_LO + (MM_HI - MM_LO) * torch.sigmoid(raw)


def huber_mm_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)


def load_scale_detector(path: "Path | str", device: torch.device | None = None) -> ScaleStripDetector:
    from pathlib import Path

    device = device or torch.device("cpu")
    model = ScaleStripDetector().to(device)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_mm_per_pixel(
    model: ScaleStripDetector,
    gray: "np.ndarray",
    device: torch.device | None = None,
) -> float:
    """Single-image mm/px from a trained ScaleStripDetector."""
    device = device or next(model.parameters()).device
    x = torch.from_numpy(extract_border_strips(gray)).unsqueeze(0).to(device)
    return float(model(x).item())
