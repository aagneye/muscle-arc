"""Cheap Gate1-lite geometry score for checkpoint selection (val only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from muscle_arc.data.dataset import read_gray
from muscle_arc.geometry.metrics import (
    fascicle_length_px,
    muscle_thickness_px,
    pennation_angle_deg,
)
from muscle_arc.infer.predict import predict_prob


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _geom(
    apo: np.ndarray,
    fasc: np.ndarray,
    gray: np.ndarray | None = None,
) -> tuple[float, float, float]:
    mt = muscle_thickness_px(apo, fasc_mask=fasc)
    pa = pennation_angle_deg(fasc, apo, gray=gray)
    fl = fascicle_length_px(
        fasc,
        apo,
        gray=gray,
        pa_deg=pa if np.isfinite(pa) else None,
        mt_px=mt if np.isfinite(mt) else None,
    )
    return (
        float(pa) if np.isfinite(pa) else float("nan"),
        float(fl) if np.isfinite(fl) else float("nan"),
        float(mt) if np.isfinite(mt) else float("nan"),
    )


@torch.no_grad()
def gate1_lite_score(
    *,
    branch: str,
    model: nn.Module,
    fasc_by_stem: dict[str, tuple[Path, Path]],
    apo_by_stem: dict[str, tuple[Path, Path]],
    stems: list[str],
    img_size: int,
    device: torch.device,
    apo_thr: float = 0.35,
    fasc_thr: float = 0.10,
    max_samples: int = 24,
) -> dict[str, float]:
    """Score predicted masks vs GT geometry on a small val subset.

    For ``fasc`` training: GT apo + predicted fasc (isolates PA/FL drawing).
    For ``apo`` training: predicted apo + GT fasc (isolates MT drawing).
    Higher ``score`` is better (negative weighted relative/absolute error).
    """
    common = [s for s in stems if s in apo_by_stem and s in fasc_by_stem]
    sample = common[: max(1, min(len(common), max_samples))]
    pa_e, fl_e, mt_e = [], [], []

    for stem in sample:
        fasc_img, fasc_m = fasc_by_stem[stem]
        apo_img, apo_m = apo_by_stem[stem]
        gray = read_gray(fasc_img)
        apo_gt = _read_mask(apo_m, gray.shape[:2])
        fasc_gt = _read_mask(fasc_m, gray.shape[:2])
        pa_g, fl_g, mt_g = _geom(apo_gt, fasc_gt, gray=gray)

        if branch == "fasc":
            fasc_prob = predict_prob(
                model, gray, img_size, device, True, True, False, False
            )
            fasc_pr = (fasc_prob > fasc_thr).astype(np.uint8)
            pa_p, fl_p, mt_p = _geom(apo_gt, fasc_pr, gray=gray)
        else:
            try:
                gray_apo = read_gray(apo_img)
            except Exception:
                gray_apo = gray
            apo_prob = predict_prob(
                model, gray_apo, img_size, device, True, True, False, False
            )
            apo_pr = (apo_prob > apo_thr).astype(np.uint8)
            if apo_pr.shape != fasc_gt.shape:
                apo_pr = cv2.resize(
                    apo_pr,
                    (fasc_gt.shape[1], fasc_gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            pa_p, fl_p, mt_p = _geom(apo_pr, fasc_gt, gray=gray)

        if np.isfinite(pa_g) and np.isfinite(pa_p):
            pa_e.append(abs(pa_p - pa_g))
        if np.isfinite(fl_g) and np.isfinite(fl_p) and fl_g > 1:
            fl_e.append(abs(fl_p - fl_g) / fl_g)
        if np.isfinite(mt_g) and np.isfinite(mt_p) and mt_g > 1:
            mt_e.append(abs(mt_p - mt_g) / mt_g)

    pa_mae = float(np.mean(pa_e)) if pa_e else 20.0
    fl_rel = float(np.mean(fl_e)) if fl_e else 1.0
    mt_rel = float(np.mean(mt_e)) if mt_e else 1.0
    err = (pa_mae / 6.0) + fl_rel + mt_rel
    return {
        "n": float(len(sample)),
        "pa_mae_deg": pa_mae,
        "fl_rel_mae": fl_rel,
        "mt_rel_mae": mt_rel,
        "score": float(-err),
    }
