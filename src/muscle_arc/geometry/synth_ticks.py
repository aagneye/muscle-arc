"""Synthetic tick-mark generator for training TickKeypointNet.

Graphics-based, on-the-fly generator in the spirit of RulerNet's
`draw_ruler(...)` (Pan et al., arXiv:2507.07077v2, S3.2), adapted for our
setting: instead of drawing a photographed-looking ruler on an arbitrary
ImageNet background, we stamp a strip of bright tick marks — matching the
appearance of ultrasound console depth markers / duty-cycle side ticks —
directly onto the border region of a **real** ultrasound image at a
randomized, known pixel pitch. This gives (image, tick_centers_px,
mm_per_pixel) triples with exact ground truth and a realistic background
(speckle, console text, colour bars) without needing manual annotation or a
ControlNet realism pass (see docs/research_scale_reader.md for the full
rationale on why we skip ControlNet/GP/DeepGP).

No perspective distortion is modeled: device-rendered tick marks are drawn
into a fronto-parallel raster with uniform pixel pitch by construction, so
unlike RulerNet's photographed rulers we do not need geometric-progression
spacing — ticks are generated at (near-)uniform pitch plus small jitter to
emulate rendering/aliasing noise only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ultrasound mm/px band already used throughout depth_scale.py / scale_detector.py
MM_LO = 0.025
MM_HI = 0.16


@dataclass
class SyntheticTickSample:
    image: np.ndarray  # (H, W) uint8 grayscale, ticks stamped along one edge
    tick_centers_px: np.ndarray  # 1D float array, tick center coordinate along the strip axis
    pitch_px: float  # ground-truth spacing between adjacent ticks
    mm_per_pixel: float
    axis: str  # "y" (ticks along vertical/left-right strip) or "x" (top/bottom strip)


def _gaussian_tick_profile(length: int, center: float, half_width: float) -> np.ndarray:
    """1D Gaussian bump used to stamp a soft-edged tick mark."""
    xs = np.arange(length, dtype=np.float64)
    return np.exp(-0.5 * ((xs - center) / max(half_width, 1e-3)) ** 2)


def generate_synthetic_tick_strip(
    background: np.ndarray,
    *,
    axis: str = "y",
    side: str = "left",
    strip_px: int = 32,
    pitch_px: float | None = None,
    mm_per_pixel: float | None = None,
    tick_len_px: float | None = None,
    tick_thickness_px: float = 1.5,
    intensity: float = 255.0,
    jitter_px: float = 0.4,
    dropout_prob: float = 0.05,
    spurious_prob: float = 0.03,
    faint: bool = False,
    rng: np.random.Generator | None = None,
) -> SyntheticTickSample:
    """Stamp a synthetic tick strip onto a copy of `background`.

    Parameters mirror RulerNet's draw_ruler controllable-parameter philosophy
    (spacing, thickness, length, color/intensity, random imperfections) but
    scoped to a single bright-tick-on-dark-background style, which is the
    only style ultrasound console ticks use.
    """
    rng = rng or np.random.default_rng()
    img = background.copy()
    if img.ndim == 3:
        import cv2

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    h, w = img.shape[:2]

    extent = h if axis == "y" else w
    if extent < 8:
        raise ValueError("background too small for tick strip")

    if mm_per_pixel is None:
        mm_per_pixel = float(rng.uniform(MM_LO, MM_HI))
    if pitch_px is None:
        # Typical device tick spacing is 5mm or 10mm marks.
        tick_mm = float(rng.choice([5.0, 10.0]))
        pitch_px = tick_mm / mm_per_pixel
        pitch_px = float(np.clip(pitch_px, 6.0, extent / 3.0))
    if tick_len_px is None:
        tick_len_px = float(rng.uniform(4.0, min(strip_px - 4, 20.0)))

    s = max(8, min(strip_px, h // 2, w // 2))
    if axis == "y":
        region = img[:, :s] if side == "left" else img[:, w - s :]
    else:
        region = img[:s, :] if side == "top" else img[h - s :, :]

    rh, rw = region.shape[:2]
    along_len = rh if axis == "y" else rw

    # Random phase offset so ticks don't always start at 0.
    phase = float(rng.uniform(0.0, pitch_px))
    centers = np.arange(phase, along_len, pitch_px, dtype=np.float64)
    centers += rng.normal(0.0, jitter_px, size=centers.shape)
    centers = centers[(centers > 1) & (centers < along_len - 1)]

    # Random dropout (missing ticks) — emulates faint/occluded marks.
    if len(centers) > 3 and dropout_prob > 0:
        keep = rng.random(len(centers)) > dropout_prob
        # never drop below 3 ticks, keeps ground truth pitch estimable
        if keep.sum() >= 3:
            centers = centers[keep]

    peak = intensity * (0.35 if faint else 1.0)
    tick_extent = int(round(tick_len_px))
    for c in centers:
        profile = peak * _gaussian_tick_profile(along_len, c, tick_thickness_px)
        profile = profile[:, None] if axis == "y" else profile[None, :]
        if axis == "y":
            band = region[:, :tick_extent]
            region[:, :tick_extent] = np.clip(band + profile, 0, 255)
        else:
            band = region[:tick_extent, :]
            region[:tick_extent, :] = np.clip(band + profile, 0, 255)

    # Spurious marks (console text / artifacts that look like ticks).
    n_spurious = rng.binomial(max(len(centers), 1), spurious_prob)
    for _ in range(n_spurious):
        c = float(rng.uniform(0, along_len))
        profile = peak * 0.8 * _gaussian_tick_profile(along_len, c, tick_thickness_px)
        profile = profile[:, None] if axis == "y" else profile[None, :]
        if axis == "y":
            band = region[:, :tick_extent]
            region[:, :tick_extent] = np.clip(band + profile, 0, 255)
        else:
            band = region[:tick_extent, :]
            region[:tick_extent, :] = np.clip(band + profile, 0, 255)

    if axis == "y":
        if side == "left":
            img[:, :s] = region
        else:
            img[:, w - s :] = region
    else:
        if side == "top":
            img[:s, :] = region
        else:
            img[h - s :, :] = region

    return SyntheticTickSample(
        image=np.clip(img, 0, 255).astype(np.uint8),
        tick_centers_px=np.sort(centers),
        pitch_px=float(pitch_px),
        mm_per_pixel=float(mm_per_pixel),
        axis=axis,
    )


def make_heatmap_target(
    length: int,
    centers_px: np.ndarray,
    *,
    sigma: float = 1.5,
) -> np.ndarray:
    """1D Gaussian-target heatmap for the detected tick centers (RulerNet §3.3).

    Multiple Gaussians are summed then clipped to [0, 1], matching RulerNet's
    "aggregate into a heatmap" convention.
    """
    heat = np.zeros(length, dtype=np.float32)
    xs = np.arange(length, dtype=np.float64)
    for c in centers_px:
        heat += np.exp(-0.5 * ((xs - c) / max(sigma, 1e-3)) ** 2).astype(np.float32)
    return np.clip(heat, 0.0, 1.0)


def synthetic_batch(
    backgrounds: list[np.ndarray],
    *,
    n_per_image: int = 4,
    strip_px: int = 32,
    heatmap_len: int = 256,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Generate (strip_crop, heatmap_target, mm_per_pixel) triples for training.

    Each strip_crop is a `strip_px`-wide 1-channel image resized so the
    along-axis dimension is `heatmap_len`; heatmap_target is the matching 1D
    Gaussian-target heatmap for CE+DICE supervision.
    """
    import cv2

    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    axes_sides = [("y", "left"), ("y", "right"), ("x", "top"), ("x", "bottom")]
    for bg in backgrounds:
        for _ in range(n_per_image):
            axis, side = axes_sides[int(rng.integers(0, len(axes_sides)))]
            faint = bool(rng.random() < 0.3)
            sample = generate_synthetic_tick_strip(
                bg, axis=axis, side=side, strip_px=strip_px, faint=faint, rng=rng
            )
            h, w = sample.image.shape[:2]
            s = max(8, min(strip_px, h // 2, w // 2))
            if axis == "y":
                region = sample.image[:, :s] if side == "left" else sample.image[:, w - s :]
                along_len = h
            else:
                region = sample.image[:s, :] if side == "top" else sample.image[h - s :, :]
                along_len = w
            # Take the max-intensity profile across the strip thickness —
            # this is exactly what _autocorr_duty_pitch does in depth_scale.py,
            # kept consistent so heuristic and learned paths see the same signal.
            profile = region.max(axis=1) if axis == "y" else region.max(axis=0)
            profile = profile.astype(np.float32)
            scale = heatmap_len / max(along_len, 1)
            profile_resized = cv2.resize(
                profile[None, :], (heatmap_len, 1), interpolation=cv2.INTER_LINEAR
            )[0]
            centers_resized = sample.tick_centers_px * scale
            heat = make_heatmap_target(heatmap_len, centers_resized, sigma=1.5)
            out.append((profile_resized / 255.0, heat, sample.mm_per_pixel))
    return out
