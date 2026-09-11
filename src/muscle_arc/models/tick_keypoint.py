"""TickKeypointNet: RulerNet-style 1D heatmap keypoint detector for tick marks.

Reformulates scale reading as keypoint detection (Pan et al., RulerNet,
arXiv:2507.07077v2 S3.1/S3.3) rather than direct mm/px regression: a small
1D CNN predicts a Gaussian-target heatmap over a border-strip intensity
profile, tick centers are recovered as local maxima, and mm_per_pixel is
derived from a robust statistic over adjacent tick spacings
("RulerNet-Median" — see docs/research_scale_reader.md for why we use Median
rather than full GP/DeepGP: our tick marks are device-rendered into a
fronto-parallel raster with no perspective distortion, so GP's non-uniform
spacing correction has no signal to fit).

Loss: CE + DICE on the heatmap (RulerNet Eq. 1-2), NOT raw L1/L2 — their
ablation showed unweighted L1/L2 heatmap regression collapses to
all-background on sparse targets. We reproduce that lesson here rather than
relearn it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Same ultrasound mm/px band used across scale_detector.py / depth_scale.py
MM_LO = 0.025
MM_HI = 0.16


class TickKeypointNet(nn.Module):
    """1D CNN heatmap regressor over a border-strip max-intensity profile.

    Input: (B, 1, L) profile (L e.g. 256). Output: (B, 1, L) heatmap logits.
    Lightweight by design — RulerNet's ablation shows resolution gains
    plateau well below our strip lengths, and our target signal (evenly
    spaced bright ticks) is far simpler than an in-the-wild photographed
    ruler.
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        c = base_ch
        self.enc = nn.Sequential(
            nn.Conv1d(1, c, 7, padding=3),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 5, padding=2),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c * 2, 5, padding=2),
            nn.BatchNorm1d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c * 2, c * 2, 3, padding=1),
            nn.BatchNorm1d(c * 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv1d(c * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x)
        return self.head(feat)  # logits, shape (B, 1, L)


def dice_loss_continuous(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """DICE adapted for continuous [0,1] targets (RulerNet Eq. 1): squares in denom."""
    p = pred.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(dim=1)
    denom = (p * p).sum(dim=1) + (t * t).sum(dim=1)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def tick_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_ce: float = 0.2,
    lambda_dice: float = 0.8,
) -> torch.Tensor:
    """CE + DICE heatmap loss (RulerNet Eq. 2, same weights as their paper).

    `logits` are raw scores (B,1,L); `target` is a [0,1] Gaussian-target
    heatmap (B,1,L) — see synth_ticks.make_heatmap_target.
    """
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), target.clamp(0.0, 1.0))
    dice = dice_loss_continuous(probs, target)
    return lambda_ce * ce + lambda_dice * dice


def extract_peaks(
    heatmap: np.ndarray,
    *,
    threshold: float = 0.5,
    min_separation: int = 4,
) -> np.ndarray:
    """Local-maxima peak extraction (simplified RulerNet Algorithm 2 for 1D).

    No Hough grouping needed: a single border-strip profile is already a 1D
    line by construction (unlike an arbitrary 2D photograph with possibly
    multiple rulers).
    """
    h = np.asarray(heatmap, dtype=np.float64)
    peaks = []
    n = len(h)
    i = 1
    while i < n - 1:
        if h[i] >= threshold and h[i] >= h[i - 1] and h[i] >= h[i + 1]:
            # Merge close peaks (keep the strongest within min_separation)
            if peaks and (i - peaks[-1]) < min_separation:
                if h[i] > h[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
        i += 1
    return np.array(peaks, dtype=np.float64)


def robust_median_pitch(
    centers: np.ndarray,
    *,
    median_tol: float = 0.20,
    ratio_tol: float = 0.10,
) -> tuple[float | None, float]:
    """RulerNet-Median: robust spacing statistic over adjacent tick centers.

    Simplified, scope-appropriate version of RulerNet's Median baseline
    (Table 2: 1.33 mAPE/cm vs 1.53 for naive Direct averaging) — reject
    adjacent spacings that deviate >`median_tol` from the overall median, or
    whose consecutive ratio deviates >`ratio_tol` from 1 (RulerNet's
    "retain only those within a 20% deviation from median distance and a 10%
    deviation from median ratio" ablation setting, S4.5).

    Returns (pitch_px, confidence) where confidence reflects the fraction of
    spacings retained after outlier rejection (more agreement -> higher
    confidence, consumed by scale_fusion.py's jury vote).
    """
    centers = np.sort(np.asarray(centers, dtype=np.float64))
    if len(centers) < 3:
        return None, 0.0
    spacings = np.diff(centers)
    spacings = spacings[spacings > 0.5]
    if len(spacings) < 2:
        return (float(spacings[0]), 0.3) if len(spacings) == 1 else (None, 0.0)

    med = float(np.median(spacings))
    if med <= 0:
        return None, 0.0
    keep_median = np.abs(spacings - med) / med <= median_tol

    ratios = spacings[1:] / np.clip(spacings[:-1], 1e-6, None)
    keep_ratio = np.ones(len(spacings), dtype=bool)
    # a spacing is suspect if either adjacent ratio deviates too much
    bad_ratio = np.abs(ratios - 1.0) > ratio_tol
    keep_ratio[:-1] &= ~bad_ratio
    keep_ratio[1:] &= ~bad_ratio

    keep = keep_median & keep_ratio
    if keep.sum() == 0:
        keep = keep_median  # relax to median-only filter
    if keep.sum() == 0:
        return med, 0.2

    clean = spacings[keep]
    conf = float(keep.sum() / len(spacings))
    return float(np.median(clean)), conf


@torch.no_grad()
def predict_pitch_px(
    model: nn.Module,
    profile: np.ndarray,
    device: torch.device | None = None,
    *,
    threshold: float = 0.5,
) -> tuple[float | None, float]:
    """Run TickKeypointNet on a 1D profile (0..1 normalized) -> (pitch_px, confidence)."""
    device = device or next(model.parameters()).device
    x = torch.from_numpy(np.asarray(profile, dtype=np.float32)).view(1, 1, -1).to(device)
    logits = model(x)
    heat = torch.sigmoid(logits)[0, 0].cpu().numpy()
    peaks = extract_peaks(heat, threshold=threshold)
    return robust_median_pitch(peaks)


def pitch_to_mm(pitch_px: float, mm_ocr_hint: float | None = None) -> tuple[float, str]:
    """Disambiguate 5mm vs 10mm tick unit — same pattern as depth_scale.py's
    ``estimate_depth_scale`` tick_order logic, reused here for consistency."""
    best_mm, best_src, best_score = None, "ticks_5mm", 1e9
    for tick_mm in (5.0, 10.0):
        mm = tick_mm / pitch_px
        if not (MM_LO <= mm <= MM_HI):
            continue
        score = abs(mm - mm_ocr_hint) / max(mm_ocr_hint, 1e-6) if mm_ocr_hint else abs(mm - 0.09)
        if score < best_score:
            best_score, best_mm, best_src = score, mm, f"ticks_{tick_mm:g}mm"
    if best_mm is None:
        # Fall back to 5mm even if outside nominal band, caller applies its own gating.
        best_mm = 5.0 / pitch_px
        best_src = "ticks_5mm_oob"
    return float(best_mm), best_src


def save_checkpoint(model: nn.Module, path: Path | str, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta or {}}, path)


def load_tick_keypoint_net(path: Path | str, device: torch.device | None = None) -> nn.Module:
    device = device or torch.device("cpu")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = TickKeypointNet().to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
