"""Deterministic px/cm and B-mode sector box from device-specific tick layouts.

The UMUD test set contains a handful of console formats. Each format burns its
depth ruler at fixed pixel columns/rows, so the scale can be read exactly by
sampling those columns rather than by generic autocorrelation. The layouts and
tick columns below follow the public Kaggle notebook *UMUD Quick and Dirty*
(AmbrosM, Apache-2.0 notebook license), which this module ports; the host
confirmed on the forum that the vertical ruler is the single px/cm for an
image (square pixels) and that bottom ticks are 1 cm apart.

Coverage on ``test_images_v2`` is 309/309. Unknown formats return ``ok=False``
so callers can fall back to the heuristic reader in ``depth_scale.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# 800x1200 Siemens exports with the ruler on the right border: the sector box
# depends on the number of 0.5 cm ticks (= imaging depth).
_RIGHT_RULER_BOX = {
    7: (142, 91, 1058),
    8: (163, 91, 1037),
    9: (211, 91, 989),
    10: (249, 91, 951),
    11: (282, 91, 918),
    12: (308, 91, 892),
    13: (331, 91, 869),
    14: (349, 91, 851),
}


@dataclass
class ExactScale:
    px_per_cm: float
    mm_per_pixel: float
    box: tuple[int, int, int, int]  # (left, top, right, bottom), exclusive right/bottom
    source: str
    ok: bool

    def as_dict(self) -> dict:
        d = asdict(self)
        d["box"] = list(self.box)
        return d


def _rgb(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)
    if a.shape[2] == 4:
        return a[..., :3]
    return a


def _fail(h: int, w: int, source: str) -> ExactScale:
    return ExactScale(-1.0, float("nan"), (0, 0, w, h), source, False)


def decode_exact_scale(image: np.ndarray, filename: str) -> ExactScale:
    """Return exact px/cm and sector box for a known console format.

    ``image`` is the raw array as read from disk (grayscale or RGB);
    ``filename`` is needed because the PNG exports use a different layout
    from the TIFFs of the same size.
    """
    a = _rgb(np.asarray(image))
    h, w = a.shape[:2]
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "png":
        # Variable depth (3-6 cm), ticks on the left edge at columns 6 (minor) and 9 (major).
        col6 = a[:, 6].mean(axis=-1)
        col9 = a[:, 9].mean(axis=-1)
        first = int(np.argmax(col6 > 50))
        minor = 150 + int(np.argmax(col6[150:] > 50))
        major = 150 + int(np.argmax(col9[150:] > 50))
        last = h - 1 - int(np.argmax(col6[::-1] > 50))
        px_per_cm = float(major - first if (major - first) < 3 * (minor - first) else minor - first)
        hw = w // 2
        w2 = int(np.argmin(a[:, hw:].sum(axis=(0, 2))))
        box = (hw - w2, first, hw + w2, last)
        ok = px_per_cm > 10 and 2.8 < (box[2] - box[0]) / px_per_cm < 2.9
        return ExactScale(px_per_cm, 10.0 / px_per_cm, box, "png_left_ruler", bool(ok))

    if (h, w) == (800, 1200):
        if (a[87, 1147:1157] == 175).all():
            col = a[:, 1150].mean(axis=-1)
            first = int(np.argmax(col > 50))
            major = first + 20 + int(np.argmax(col[first + 20 :] > 50))
            last = h - 1 - int(np.argmax(col[::-1] > 50))
            n_ticks = int(round((last - first) / max(1, major - first)))
            if n_ticks in _RIGHT_RULER_BOX:
                px_per_cm = (last - first) / n_ticks * 2.0  # two ticks per cm
                l, t, r = _RIGHT_RULER_BOX[n_ticks]
            elif n_ticks == 15:
                px_per_cm = (last - first) / 3.0  # five ticks per cm
                l, t, r = 142, 91, 1058
            else:
                return _fail(h, w, "siemens_right_ruler_unknown_ticks")
            box = (l, t, r, last)
            ok = 5.7 < (r - l) / px_per_cm < 5.8
            return ExactScale(float(px_per_cm), 10.0 / px_per_cm, box, "siemens_right_ruler", bool(ok))
        if (a[42, 67:74, 0] > 115).all():
            px_per_cm = (783.0 - 42.0) / 5.0
            return ExactScale(px_per_cm, 10.0 / px_per_cm, (171, 42, 1029, 798), "left_ruler_5cm", True)
        return _fail(h, w, "800x1200_unknown")

    if (h, w) == (644, 1088):
        px_per_cm = 630.5 / 5.0
        return ExactScale(px_per_cm, 10.0 / px_per_cm, (140, 0, 947, 643), "fixed_644x1088", True)

    if h in (512, 513):
        for left, right in ((49, 438), (52, 441), (53, 442)):
            if a[-5, left, 0] == 168 and a[-5, right, 0] == 168:
                px_per_cm = (442.0 - 53.0) / 5.0
                return ExactScale(px_per_cm, 10.0 / px_per_cm, (0, 0, w, h - 10), "bottom_ruler_512", True)
        return _fail(h, w, "512_unknown")

    if h == 853:
        for left, right in ((100, 934), (44, 879)):
            if a[-5, left, 0] == 170 and a[-5, right, 0] == 170:
                px_per_cm = (right - left) / 5.0
                return ExactScale(px_per_cm, 10.0 / px_per_cm, (0, 0, w, h), "bottom_ruler_853", True)
        return _fail(h, w, "853_unknown")

    return _fail(h, w, "unknown_format")
