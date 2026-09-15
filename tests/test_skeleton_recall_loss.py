"""Unit tests for SkeletonRecallLoss (arXiv:2404.03010)."""

from __future__ import annotations

import math

import numpy as np
import torch

from muscle_arc.train.losses import SkeletonRecallLoss, sof_total_loss


def _line_mask(h: int = 64, w: int = 64, y: int = 32) -> torch.Tensor:
    """A single horizontal line mask (N=1, C=1, H, W)."""
    m = np.zeros((h, w), dtype=np.float32)
    m[y, :] = 1.0
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0)


def _broken_line_mask(h: int = 64, w: int = 64, y: int = 32, gap: tuple[int, int] = (28, 36)) -> torch.Tensor:
    """Same line but with a gap in the middle (disconnected)."""
    m = np.zeros((h, w), dtype=np.float32)
    m[y, :] = 1.0
    m[y, gap[0] : gap[1]] = 0.0
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0)


def _logits_from_mask(mask: torch.Tensor, high: float = 10.0, low: float = -10.0) -> torch.Tensor:
    """Turn a {0,1} mask into confident logits matching it exactly."""
    return torch.where(mask > 0.5, torch.full_like(mask, high), torch.full_like(mask, low))


def test_skeleton_recall_loss_is_near_minus_one_for_near_perfect_recall() -> None:
    # Use a thick line (radius >= dilate_radius) so a confident prediction
    # covers the entire tubed skeleton region, not just a 1px centerline —
    # otherwise even a "perfect" copy of the target under-covers the
    # dilated tube and recall is diluted by design (this is the loss
    # working as intended: it wants coverage of the *tubed* skeleton).
    h, w = 64, 64
    thick = np.zeros((h, w), dtype=np.float32)
    thick[28:37, :] = 1.0  # 9px-thick band, comfortably covers a radius-2 tube
    target = torch.from_numpy(thick).unsqueeze(0).unsqueeze(0)
    logits = _logits_from_mask(target)
    loss_fn = SkeletonRecallLoss()

    loss = loss_fn(logits, target)

    # Near-perfect recall on the (fully-covered) skeleton tube -> loss ~ -1.
    assert math.isclose(float(loss), -1.0, abs_tol=0.05)


def test_skeleton_recall_loss_penalizes_missing_skeleton_coverage() -> None:
    target = _line_mask()
    # Prediction that only covers the left half of the line.
    partial_pred = target.clone()
    partial_pred[:, :, :, 32:] = 0.0
    logits = _logits_from_mask(partial_pred)
    loss_fn = SkeletonRecallLoss()

    loss_full = loss_fn(_logits_from_mask(target), target)
    loss_partial = loss_fn(logits, target)

    # Partial coverage of the true skeleton must score worse (higher/less
    # negative loss) than full coverage.
    assert float(loss_partial) > float(loss_full)


def test_skeleton_recall_loss_is_symmetric_to_dice_style_equal_iou_but_broken_topology() -> None:
    """Core motivating property of the paper: a prediction that recreates
    the ground-truth skeleton continuously should score at least as well
    as (and typically better than, for connectivity-sensitive downstream
    use) a prediction with the same pixel count but gaps in the middle —
    even though plain Dice/IoU would be identical for equal-area
    predictions regardless of connectivity."""
    target = _line_mask()
    continuous_pred = target.clone()
    broken_pred = _broken_line_mask()  # same total mass minus the gap region

    loss_fn = SkeletonRecallLoss()
    loss_continuous = loss_fn(_logits_from_mask(continuous_pred), target)
    loss_broken = loss_fn(_logits_from_mask(broken_pred), target)

    # Recall-on-skeleton penalizes the missing gap directly: broken
    # prediction must score worse than the fully-continuous one.
    assert float(loss_broken) > float(loss_continuous)


def test_skeleton_recall_loss_accepts_precomputed_tubed_skeleton() -> None:
    target = _line_mask()
    logits = _logits_from_mask(target)
    loss_fn = SkeletonRecallLoss()

    precomputed = loss_fn.compute_tubed_skeleton(target)
    loss_with_precomputed = loss_fn(logits, target, tubed_skeleton=precomputed)
    loss_lazy = loss_fn(logits, target)

    assert math.isclose(float(loss_with_precomputed), float(loss_lazy), rel_tol=1e-5)


def test_sof_total_loss_defaults_to_tversky_only_no_behavior_change() -> None:
    """protocol-style guard: fasc_loss='tversky' (default) must not invoke
    SkeletonRecallLoss at all, and fasc_skel stat must be exactly zero."""
    out, batch = _make_sof_batch()

    _total, stats = sof_total_loss(out, batch)

    assert stats["fasc_skel"] == 0.0


def test_sof_total_loss_tversky_skelrecall_adds_nonzero_skel_term() -> None:
    out, batch = _make_sof_batch()

    _total, stats = sof_total_loss(out, batch, fasc_loss="tversky_skelrecall")

    assert stats["fasc_skel"] != 0.0


def _make_sof_batch() -> tuple[dict, dict]:
    n, h, w = 1, 32, 32
    out = {
        "apo_bin": torch.randn(n, 1, h, w, requires_grad=False),
        "fasc_bin": torch.randn(n, 1, h, w, requires_grad=False),
        "apo_logits": torch.randn(n, 3, h, w, requires_grad=False),
        "y_super": torch.rand(n, w),
        "y_deep": torch.rand(n, w) + 0.5,
        "ori": torch.randn(n, 2, h, w, requires_grad=False),
    }
    fasc_tgt = torch.zeros(n, 1, h, w)
    fasc_tgt[:, :, h // 2, :] = 1.0
    batch = {
        "apo": torch.zeros(n, 1, h, w),
        "fasc": fasc_tgt,
        "apo_cls": torch.zeros(n, h, w, dtype=torch.long),
        "surf_valid": torch.ones(n, w),
        "y_super": torch.rand(n, w),
        "y_deep": torch.rand(n, w) + 0.5,
        "ori_cos2": torch.ones(n, h, w),
        "ori_sin2": torch.zeros(n, h, w),
    }
    return out, batch
