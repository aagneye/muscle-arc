"""Confidence-weighted jury fusion over independent scale estimators.

Formalizes the hardcoded if/elif priority chain previously embedded in
``estimate_depth_scale`` (geometry/depth_scale.py) as an explicit, auditable
vote across every scale-estimation source we have:

- OCR + sector height           (depth_scale.read_depth_cm / mm_ocr)
- Duty-cycle / comb tick pitch  (depth_scale.tick_pitch_px)
- OSF (h, w) shape lookup       (depth_scale.osf_shape_scale_lookup)
- TickKeypointNet               (models.tick_keypoint, NEW)
- OSF residual GBDT regressor   (geometry.residual_scale, opt-in only)

Rule: confidence-weighted median. If the high-confidence candidates disagree
beyond a tolerance band, fall back to the single highest-confidence source
rather than averaging away a real disagreement — averaging a good tick read
with a bad OCR read would produce a worse answer than picking the better one
outright (same principle RulerNet applies by flagging/rejecting low-quality
detections instead of blending them in). See docs/research_scale_reader.md
for the full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MM_LO = 0.025
MM_HI = 0.16

# Relative spread (max-min)/median among conf>=DISAGREE_MIN_CONF candidates
# above which we distrust the blended median and defer to the single best.
DISAGREE_SPREAD = 0.15
DISAGREE_MIN_CONF = 0.5


@dataclass
class ScaleCandidate:
    source: str
    mm_per_pixel: float
    confidence: float


@dataclass
class FusionResult:
    mm_per_pixel: float | None
    confidence: float
    source: str
    agreement: float  # relative spread among high-confidence voters (0 = perfect agreement)
    n_candidates: int
    candidates: list[ScaleCandidate]


def _valid(cands: list[ScaleCandidate]) -> list[ScaleCandidate]:
    out = []
    for c in cands:
        if c.mm_per_pixel is None or not np.isfinite(c.mm_per_pixel):
            continue
        if not (MM_LO <= c.mm_per_pixel <= MM_HI):
            continue
        if c.confidence <= 0:
            continue
        out.append(c)
    return out


def confidence_weighted_vote(
    candidates: list[ScaleCandidate],
    *,
    disagree_spread: float = DISAGREE_SPREAD,
    disagree_min_conf: float = DISAGREE_MIN_CONF,
) -> FusionResult:
    """Fuse independent mm_per_pixel estimates into one confidence-weighted answer.

    Steps:
      1. Drop invalid/out-of-band/zero-confidence candidates.
      2. If none remain, return an empty (None) result.
      3. Compute the confidence-weighted median over remaining candidates.
      4. Measure agreement among the *high-confidence* subset
         (confidence >= disagree_min_conf). If their relative spread exceeds
         `disagree_spread`, distrust the blend and return the single
         highest-confidence candidate instead — this is the
         "only trust the best single source when the jury disagrees a lot"
         rule from the design doc.
    """
    valid = _valid(candidates)
    if not valid:
        return FusionResult(None, 0.0, "none", 0.0, 0, list(candidates))

    values = np.array([c.mm_per_pixel for c in valid], dtype=np.float64)
    weights = np.array([c.confidence for c in valid], dtype=np.float64)

    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cum = np.cumsum(weights_sorted)
    cutoff = 0.5 * weights_sorted.sum()
    idx = int(np.searchsorted(cum, cutoff))
    idx = min(idx, len(values_sorted) - 1)
    weighted_median = float(values_sorted[idx])

    hi = [c for c in valid if c.confidence >= disagree_min_conf]
    if len(hi) >= 2:
        hi_vals = np.array([c.mm_per_pixel for c in hi], dtype=np.float64)
        med = float(np.median(hi_vals))
        spread = float((hi_vals.max() - hi_vals.min()) / max(med, 1e-6))
    else:
        spread = 0.0

    if len(hi) >= 2 and spread > disagree_spread:
        best = max(valid, key=lambda c: c.confidence)
        return FusionResult(
            mm_per_pixel=float(best.mm_per_pixel),
            confidence=float(best.confidence),
            source=f"jury_disagree_fallback:{best.source}",
            agreement=spread,
            n_candidates=len(valid),
            candidates=list(candidates),
        )

    fused_conf = float(np.average(weights, weights=weights)) if len(weights) else 0.0
    # Reward agreement: if the jury agrees, boost confidence slightly (cap 0.98).
    if len(hi) >= 2:
        fused_conf = float(min(0.98, max(fused_conf, 1.0 - spread) * (0.9 + 0.1 * len(hi))))
        fused_conf = min(fused_conf, 0.98)
    src_list = "+".join(sorted({c.source for c in valid}))
    return FusionResult(
        mm_per_pixel=weighted_median,
        confidence=fused_conf,
        source=f"jury:{src_list}",
        agreement=spread,
        n_candidates=len(valid),
        candidates=list(candidates),
    )


def build_candidates_for_image(
    *,
    mm_ocr: float | None = None,
    conf_ocr: float = 0.0,
    mm_ticks: float | None = None,
    conf_ticks: float = 0.0,
    mm_osf_shape: float | None = None,
    conf_osf_shape: float = 0.0,
    mm_tick_keypoint: float | None = None,
    conf_tick_keypoint: float = 0.0,
    mm_residual: float | None = None,
    conf_residual: float = 0.0,
    use_residual: bool = False,
) -> list[ScaleCandidate]:
    """Assemble the jury roster for one image from already-computed per-source
    estimates. Kept as a thin, explicit constructor so call sites in
    calibrate_predict.py stay readable and each source's provenance is
    preserved in the returned FusionResult for debugging/audit.

    `use_residual` defaults False and must be explicitly opted into by the
    caller — never silently included (v9 lesson: global OSF residual mm
    blend regressed public LB 0.999 -> 1.062 despite looking better on a
    2-sample local check; residual stays experimental/opt-in only).
    """
    cands = [
        ScaleCandidate("ocr+sector", mm_ocr, conf_ocr) if mm_ocr is not None else None,
        ScaleCandidate("ticks", mm_ticks, conf_ticks) if mm_ticks is not None else None,
        ScaleCandidate("osf_shape", mm_osf_shape, conf_osf_shape) if mm_osf_shape is not None else None,
        ScaleCandidate("tick_keypoint", mm_tick_keypoint, conf_tick_keypoint)
        if mm_tick_keypoint is not None
        else None,
    ]
    if use_residual and mm_residual is not None:
        cands.append(ScaleCandidate("residual_gbdt", mm_residual, conf_residual))
    return [c for c in cands if c is not None]
