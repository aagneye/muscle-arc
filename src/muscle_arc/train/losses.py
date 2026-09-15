"""Losses for Surfaces + Orientation Field training."""

from __future__ import annotations

import numpy as np
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


class SkeletonRecallLoss(nn.Module):
    """Skeleton Recall Loss (Kirchhoff et al., arXiv:2404.03010, ECCV 2024).

    Designed for exactly our fascicle-branch problem: thin, sparse,
    easily-disconnected structures where plain Dice/Tversky rewards
    volumetric overlap but not connectivity. Standard Dice can score well
    on a segmentation that is fragmented into disconnected pieces; this
    loss explicitly penalizes missing pixels *on the ground-truth
    skeleton*, which is far more sensitive to broken/gapped thin
    structures than overlap alone.

    Algorithm (paper's "Tubed Skeletonization", Algorithm 1, binary case):
      1. Binarize the ground-truth mask.
      2. Skeletonize it (skimage.morphology.skeletonize) -> single-pixel-wide
         centerline.
      3. Dilate the skeleton with a disk kernel of radius `dilate_radius`
         (paper uses radius 2) to make it "tubed" -- gives the loss more
         than a single pixel of signal per skeleton location.
      4. Soft recall of the *prediction* on this tubed skeleton (Eq. 1):
         L = -mean_c[ sum_i(Y_skel * Y_hat) / sum_i(Y_skel) ]
         i.e. "what fraction of predicted probability mass lands on
         pixels that are actually part of the true skeleton", maximized.

    Steps 1-3 (skeletonization) are pure CPU/numpy ops on the target mask,
    matching the paper's stated design goal of avoiding GPU-based
    differentiable skeletonization entirely -- here they can be precomputed
    once per mask in the dataset/collate step (cheap, non-differentiable
    preprocessing) rather than every forward pass, though this
    implementation also supports computing them lazily inside forward()
    for simplicity when no precomputed skeleton is supplied.

    This is a plug-in *addition* to an existing generic loss (paper Eq. 2:
    L = L_generic + w * L_SkelRecall), not a replacement — combine with
    TverskyBCELoss via a config-selected weight, do not use alone.
    """

    def __init__(self, dilate_radius: int = 2, eps: float = 1e-6) -> None:
        super().__init__()
        self.dilate_radius = dilate_radius
        self.eps = eps

    def compute_tubed_skeleton(self, target: torch.Tensor) -> torch.Tensor:
        """Precompute the tubed skeleton for a batch of binary target masks.

        target: (N, 1, H, W) or (N, H, W), values in {0, 1} (or probabilities
        thresholded at 0.5). Returns a same-shape float tensor on the same
        device, differentiable-free (skeletonization runs on CPU numpy).
        """
        from skimage.morphology import dilation, disk, skeletonize

        squeeze_channel = target.ndim == 4 and target.shape[1] == 1
        arr = target.detach().cpu().numpy()
        if squeeze_channel:
            arr = arr[:, 0]
        binary = arr > 0.5
        kernel = disk(self.dilate_radius)
        out = np.zeros_like(binary, dtype=np.float32)
        for i in range(binary.shape[0]):
            skel = skeletonize(binary[i])
            tubed = dilation(skel, footprint=kernel)
            out[i] = tubed.astype(np.float32)
        skel_t = torch.from_numpy(out).to(target.device)
        if squeeze_channel:
            skel_t = skel_t.unsqueeze(1)
        return skel_t

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        tubed_skeleton: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """logits/target: (N, C, H, W). `tubed_skeleton` may be precomputed
        (recommended — see class docstring) and passed in directly; if not
        supplied it is computed here from `target` on every call."""
        probs = torch.sigmoid(logits)
        skel = tubed_skeleton if tubed_skeleton is not None else self.compute_tubed_skeleton(target)
        dims = tuple(range(2, probs.ndim))
        numer = (skel * probs).sum(dim=dims)
        denom = skel.sum(dim=dims) + self.eps
        recall = numer / denom
        # Average over channels (C dim, index 1), then batch — matches the
        # paper's 1/|C| * sum_c formulation (Eq. 1); batch mean is standard
        # loss reduction not specified explicitly in the paper's per-sample
        # equation.
        return -recall.mean()


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
    fasc_loss: str = "tversky",
    skel_recall_weight: float = 0.3,
    skel_dilate_radius: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``fasc_loss`` selects the fascicle-branch loss:
      - "tversky" (default): existing TverskyBCELoss only, unchanged
        behaviour.
      - "tversky_skelrecall": adds SkeletonRecallLoss (arXiv:2404.03010)
        on the fascicle branch, weighted by `skel_recall_weight`, following
        the paper's own combination scheme L = L_generic + w*L_SkelRecall
        (their Eq. 2). Targets exactly this project's known weak point:
        thin, easily-fragmented fascicle strands, where Tversky/Dice can
        score acceptably even when the prediction is disconnected.
    """
    dice = SoftDiceLoss()
    tversky = TverskyBCELoss(bce_weight=bce_weight)
    apo_tgt = batch["apo"]
    fasc_tgt = batch["fasc"]
    loss_apo = F.binary_cross_entropy_with_logits(out["apo_bin"], apo_tgt) + dice(
        out["apo_bin"], apo_tgt
    )
    loss_fasc = tversky(out["fasc_bin"], fasc_tgt)
    loss_fasc_skel = torch.zeros((), device=loss_fasc.device)
    if fasc_loss == "tversky_skelrecall":
        skel_recall = SkeletonRecallLoss(dilate_radius=skel_dilate_radius)
        loss_fasc_skel = skel_recall(out["fasc_bin"], fasc_tgt)
        loss_fasc = loss_fasc + skel_recall_weight * loss_fasc_skel
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
        "fasc_skel": float(loss_fasc_skel.detach()),
        "cls": float(loss_cls.detach()),
        "surf": float(loss_surf.detach()),
        "ori": float(loss_ori.detach()),
        "tv": float(loss_tv.detach()),
        "mt": float(loss_mt.detach()),
        "total": float(total.detach()),
    }
    return total, stats
