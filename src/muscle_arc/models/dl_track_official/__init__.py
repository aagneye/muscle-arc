"""Thin wrapper around vendored DL_Track doCalculations."""

from __future__ import annotations

from typing import Any

import numpy as np

from muscle_arc.models.dl_track_official.do_calculations import doCalculations


def official_metrics(
    apo: np.ndarray,
    fasc: np.ndarray,
    full_gray: np.ndarray,
    *,
    mm_per_pixel: float,
    apo_thr: float = 0.35,
    fasc_thr: float = 0.10,
    model_apo: Any = None,
    model_fasc: Any = None,
) -> tuple[float, float, float]:
    """Return (pa_deg, fl_mm, mt_mm) via official geometry + OCR/ticks scale.

    Masks must be HxW of ``full_gray`` (letterbox predict + chrome-mask OK).
    ``calib_dist`` is pixels spanning 10 mm.
    """
    h, w = full_gray.shape[:2]
    mm = float(mm_per_pixel)
    if not (0.025 <= mm <= 0.16):
        mm = 0.06
    calib_dist = max(1, int(round(10.0 / mm)))
    dictionary = {
        "fascicle_length_threshold": "40",
        "minimal_muscle_width": "60",
        "maximal_pennation_angle": "40",
        "minimal_pennation_angle": "10",
        "aponeurosis_detection_threshold": str(apo_thr),
        "fascicle_detection_threshold": str(fasc_thr),
        "aponeurosis_length_threshold": "600",
    }
    dummy = np.zeros((1, 512, 512, 3), dtype=np.float32)
    rgb = np.stack([full_gray, full_gray, full_gray], axis=-1).astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    fasc_l, pennation, _xl, _xh, midthick, _fig = doCalculations(
        dummy,
        rgb,
        h,
        w,
        calib_dist,
        10,
        model_apo,
        model_fasc,
        dictionary,
        filter_fasc=True,
        image_callback=None,
        apo_mask=apo,
        fasc_mask=fasc,
    )
    if fasc_l is None or len(fasc_l) == 0:
        pa = float("nan")
        fl = float("nan")
    else:
        pa = float(np.median(pennation))
        fl = float(np.median(fasc_l))
    mt = float(midthick) if midthick is not None else float("nan")
    return pa, fl, mt
