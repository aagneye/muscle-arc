"""Losses for Surfaces + Orientation Field training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        inter = (probs * target).sum(dims)
        den = probs.sum(dims) + target.sum(dims) + self.eps
        return 1.0 - (2.0 * inter / den).mean()


class TverskyBCELoss(nn.Module):
    """Tversky + down-weighted BCE for incomplete fascicle labels (PU-ish)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, bce_weight: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        tp = (probs * target).sum(dims)
        fp = (probs * (1 - target)).sum(dims)
        fn = ((1 - probs) * target).sum(dims)
        tversky = (tp + 1e-5) / (tp + self.alpha * fp + self.beta * fn + 1e-5)
        return (1.0 - tversky.mean()) + self.bce_weight * self.bce(logits, target)


def masked_orientation_loss(
    pred: torch.Tensor,
    cos2: torch.Tensor,
    sin2: torch.Tensor,
    presence: torch.Tensor,
) -> torch.Tensor:
    """Cosine loss on double-angle vector only where presence > 0."""
    # pred: (N,2,H,W)
    pc, ps = pred[:, 0], pred[:, 1]
    # normalize
    norm = torch.sqrt(pc * pc + ps * ps + 1e-8)
    pc, ps = pc / norm, ps / norm
    mask = (presence > 0.5).float()
    # target already unit-ish
    tn = torch.sqrt(cos2 * cos2 + sin2 * sin2 + 1e-8)
    tc, ts = cos2 / tn, sin2 / tn
    cos_sim = pc * tc + ps * ts
    loss = (1.0 - cos_sim) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def orientation_tv_loss(pred: torch.Tensor, belly: torch.Tensor) -> torch.Tensor:
    """Total variation on orientation inside predicted belly."""
    # pred (N,2,H,W), belly (N,1,H,W) or (N,H,W)
    if belly.ndim == 3:
        belly = belly.unsqueeze(1)
    mask = (belly > 0.5).float()
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
    mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    return (dx * mx).sum() / mx.sum().clamp_min(1.0) + (dy * my).sum() / my.sum().clamp_min(1.0)


def surface_l1_loss(
    y_pred: torch.Tensor,
    y_tgt: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """y_pred/y_tgt (N,W) in [0,1]; valid (N,W)."""
    mask = valid > 0.5
    if mask.sum() < 1:
        return y_pred.sum() * 0.0
    return (y_pred - y_tgt).abs()[mask].mean()


def sof_total_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    bce_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    dice = SoftDiceLoss()
    tversky = TverskyBCELoss(bce_weight=bce_weight)
    apo_tgt = batch["apo"]
    fasc_tgt = batch["fasc"]
    loss_apo = F.binary_cross_entropy_with_logits(out["apo_bin"], apo_tgt) + dice(
        out["apo_bin"], apo_tgt
    )
    loss_fasc = tversky(out["fasc_bin"], fasc_tgt)
    loss_cls = F.cross_entropy(out["apo_logits"], batch["apo_cls"].long())
    loss_surf = surface_l1_loss(out["y_super"], batch["y_super"], batch["surf_valid"]) + surface_l1_loss(
        out["y_deep"], batch["y_deep"], batch["surf_valid"]
    )
    loss_ori = masked_orientation_loss(
        out["ori"], batch["ori_cos2"], batch["ori_sin2"], batch["fasc"].squeeze(1)
    )
    belly = (out["apo_bin"].sigmoid() > 0.3).float()
    # rough belly: between surfaces if available — use fasc∪ dilated apo
    loss_tv = orientation_tv_loss(out["ori"], (fasc_tgt + belly).clamp(0, 1))
    # Differentiable MT proxy from surfaces (normalized); only where valid
    mt_pred = (out["y_deep"] - out["y_super"]).clamp_min(0)
    mt_tgt = (batch["y_deep"] - batch["y_super"]).clamp_min(0)
    loss_mt = surface_l1_loss(mt_pred, mt_tgt, batch["surf_valid"])

    total = (
        1.0 * loss_apo
        + 1.0 * loss_fasc
        + 0.5 * loss_cls
        + 0.5 * loss_surf
        + 0.5 * loss_ori
        + 0.05 * loss_tv
        + 0.3 * loss_mt
    )
    stats = {
        "apo": float(loss_apo.detach()),
        "fasc": float(loss_fasc.detach()),
        "cls": float(loss_cls.detach()),
        "surf": float(loss_surf.detach()),
        "ori": float(loss_ori.detach()),
        "tv": float(loss_tv.detach()),
        "mt": float(loss_mt.detach()),
        "total": float(total.detach()),
    }
    return total, stats
