# Research-backed scale reader — design

## Why this exists

Own audit: **0/309** test TIFFs carry usable scale metadata (all placeholder 72 DPI).
Own notes (`docs/curator_climb.md`, `.cursor/rules/scale-and-gates.mdc`): gap to
top-10 (~0.40 UMUD) vs public ~0.953 (v10c) is **mm/px, not segmentation**. This doc
designs a literature-grounded scale-reading subsystem to close that gap, replacing
today's ad-hoc if/elif priority chain with a principled, citable pipeline.

## Primary citation: RulerNet

Pan, Mehta, Sincerbeaux, Goldstein, Gernand, Wang. *RulerNet: Learning
Perspective-Invariant Ruler Representations for Robust Image Scale Estimation.*
arXiv:2507.07077v2 (2025–2026); DOI 10.1016/j.compmedimag.2026.102805.

Core reformulation we adopt: **stop treating scale reading as OCR/regression;
treat it as keypoint detection.** RulerNet detects cm-mark keypoints via a
heatmap CNN (HRNet-style, CE+DICE loss), then recovers scale from the detected
mark sequence with a robust spacing statistic. Three findings directly shape
our design:

1. **Loss ablation:** a plain unweighted L1/L2 heatmap-regression baseline
   *collapsed to all-background* and produced no usable detections under their
   training setup. CE+DICE on Gaussian-target heatmaps was required because
   marks occupy a tiny foreground fraction. → We use **CE+DICE**, not raw
   MSE regression, for `TickKeypointNet` (see below).
2. **Scale-extraction ablation** (Table 2, arXiv v2): accuracy ordering
   `Direct (1.53) < Median (1.33) < GP (1.38*) < DeepGP (0.84)` mAPE/cm@768
   (*GP alone is worse than Median on AnyRuler but better on the harder
   Rulers2023 set — GP mainly helps under strong perspective distortion).
   **Median-of-adjacent-spacings already recovers most of the gain over naive
   Direct averaging**, cheaply.
3. **Synthetic data is the primary lever**, not secondary: their ablation
   (Table 4/5) shows synthetic-ruler volume monotonically improves
   generalization to an unseen ruler dataset (Rulers2023), with combined
   graphics+generative synthetic pretraining giving the largest jump
   (1.53→1.50 in-domain, 4.43→1.82 out-of-domain mAPE/cm). We adopt their
   **graphics-based on-the-fly generator** pattern; we skip their
   ControlNet-realism stage (image-generation model overkill for a tick
   pattern stamped on a real US image background — our "ruler" is a strip of
   tick marks on genuine ultrasound border pixels, not a photographed object
   needing photorealism transfer).

### Deliberate reduced scope vs. full RulerNet

RulerNet's GP / DeepGP machinery exists to handle **strong, unknown
perspective distortion** — a ruler photographed at an oblique angle, at
unknown distance, with unknown lens parameters. Our setting is materially
easier and the difference matters for scope:

- The ultrasound probe's on-screen tick marks (console depth markers or
  duty-cycle side ticks) are drawn by the machine's own renderer directly
  into a **fronto-parallel raster**, not photographed. There is no camera,
  no lens, no oblique viewing angle, no perspective homography — the tick
  pitch is uniform in pixel-space by construction (device firmware draws
  evenly-spaced marks).
- Therefore the GP model's whole purpose (approximating perspective-induced
  *non-uniform* spacing with a geometric progression ratio `r`) has no
  analogue here: `r ≈ 1` always, and fitting `r` would just be fitting noise.
- We adopt **RulerNet-Median** (robust median of adjacent detected-tick
  spacings with an outlier-rejection band) as the scope-appropriate
  simplification, explicitly *not* GP/DeepGP. This is a documented,
  literature-justified reduction, not a missed step: full DeepGP (a 1.6M-param
  1D U-Net trained on 1.23B synthetic sequences) targets a strictly harder
  problem (unknown perspective + missing/spurious marks from photographic
  noise) than ours (near-uniform pitch + occasional occluded/faint tick from
  console overlay text).

## Architecture

```
                    ┌─────────────────────────┐
   full frame ──────▶  sector_crop/mask_chrome │  (existing)
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 ┌─────────────┐      ┌──────────────────┐      ┌───────────────────┐
 │ OCR + sector │      │ duty-cycle tick   │      │ TickKeypointNet    │  (NEW)
 │ height (mm)  │      │ autocorr/comb (mm)│      │ heatmap→ticks (mm) │
 │ existing     │      │ existing          │      │ synthetic-trained  │
 └──────┬───────┘      └─────────┬─────────┘      └─────────┬──────────┘
        │                        │                          │
        │              ┌─────────▼─────────┐      ┌─────────▼──────────┐
        │              │ OSF shape lookup   │      │ residual regressor │
        │              │ (h,w)→mm existing  │      │ (GBDT) existing    │
        │              └─────────┬──────────┘      └─────────┬──────────┘
        └────────────────────────┴─────────────┬─────────────┘
                                                ▼
                              ┌──────────────────────────────┐
                              │  scale_fusion.py — jury vote  │   (NEW)
                              │  confidence-weighted median   │
                              └───────────────┬───────────────┘
                                                ▼
                              ┌──────────────────────────────┐
                              │ share_scales_in_groups()      │  (existing,
                              │ 5-frame video consensus       │   wire new source)
                              └───────────────┬───────────────┘
                                                ▼
                              ┌──────────────────────────────┐
                              │ eval_osf_pipeline.py gate      │  (existing,
                              │ mandatory before submit        │   enforce)
                              └──────────────────────────────┘
```

### (a) Synthetic tick generator — `src/muscle_arc/geometry/synth_ticks.py`

Graphics-based, on-the-fly (RulerNet §3.2 "Graphics-based Synthetic Rulers"
pattern, adapted): stamp a synthetic tick/depth-marker strip onto a **real**
ultrasound image's border region at a randomized, known pixel pitch, so the
model trains on realistic backgrounds (speckle, console text, colour bars)
with ground-truth spacing. Parametrized like RulerNet's `draw_ruler(...)`:
mark spacing (px), thickness, length, color/polarity (bright-on-dark is our
only case — B-mode ticks are always bright), jitter, dropout (missing ticks),
spurious-mark injection, faint/low-contrast variant. Produces
`(image, tick_centers_px, mm_per_pixel)` triples — no manual annotation, unlike
RulerNet's real `AnyRuler` collection (we don't need one; our marks are
synthetic-only, drawn with known ground truth by construction, same
justification RulerNet gives for why synthetic volume is uncapped).

### (b) TickKeypointNet — `src/muscle_arc/models/tick_keypoint.py`

Small heatmap CNN (1-channel Gaussian-target heatmap over a border strip),
trained with **CE+DICE** (RulerNet §3.3 Eq. 1–2), not L1/L2 regression —
directly avoiding their documented collapse failure mode. Inference: extract
local-maxima peaks from the heatmap (RulerNet Algorithm 2, simplified —
no Hough grouping needed since a single strip is already a 1D line by
construction), compute adjacent spacings, apply RulerNet-Median (reject
spacings >20% from the median, deviating >10% in ratio from neighbours —
same thresholds as RulerNet §4.5), and convert pitch→mm/px using the same
5mm/10mm tick-unit disambiguation logic already in `depth_scale.py`.

Training script: `scripts/train_tick_keypoint.py`, synthetic data only
(cheap, unlimited, compute is not the constraint per project owner).

### (c) `scale_fusion.py` — confidence-weighted jury

Replaces the hardcoded if/elif priority chain in `estimate_depth_scale()`
with an explicit, auditable vote:

```python
candidates = [
    (mm_ocr_sector, conf_ocr),        # existing OCR + sector height
    (mm_ticks_heuristic, conf_ticks), # existing duty-cycle/comb autocorr
    (mm_osf_shape, conf_shape),       # existing (h,w) shape lookup
    (mm_tick_keypoint, conf_keypoint),# NEW TickKeypointNet
    (mm_residual_gbdt, conf_residual),# existing OSF-trained regressor (opt-in)
]
fused_mm, fused_conf, agreement = confidence_weighted_vote(candidates)
```

Vote rule: weighted median by confidence; if candidates disagree beyond a
tolerance band (default >15% relative spread among conf≥0.5 members), fall
back to the single highest-confidence source rather than averaging away a
real disagreement (silently averaging a good tick read with a bad OCR read
would be worse than picking the better one — same reasoning as RulerNet's
choice to reject/flag rather than blindly average low-confidence detections).
This directly operationalizes the plan point: *"only trust the CNN alone
when heuristics disagree a lot."*

### (d) Video-group consensus — already exists, wire in new source

`share_scales_in_groups()` already implements the median-consensus described
in the plan (5-frame runs from `sequence_groups()` in `calibrate_predict.py`,
min-confidence gate 0.55, high-conf members keep their own value, low-conf
members inherit the group median). The only change needed is contributing
`scale_fusion.py`'s fused output (with its fused confidence) into the same
`live_scales` / `live_conf` dicts already threaded through
`calibrate_predict.py`, so the existing group-consensus machinery picks it up
for free.

### (e) Honest evaluation gate — already exists, enforce as mandatory

`eval_osf_pipeline.py` / `eval_umud_local.py` already run the full pipeline
against the public OSF expert-benchmark set. **v9 lesson (2026-09-05,
recorded in `.cursor/rules/scale-and-gates.mdc`):** an OSF direct-mm blend
looked better on a **2-sample** local GT check (n=2) but scored **1.062** on
the real leaderboard vs **0.999** for the v8 rollback — a same-direction
regression the small sample didn't catch. Rule going forward, restated here
as a hard gate: **no submission without a green OSF Gate2/Gate2b run**;
n=2 sample MAE alone is explicitly insufficient (see "Do not" list in
`curator_climb.md`).

## Non-goals (explicitly out of scope, and why)

- **No DeepGP / GP fitting.** Justified above — no real perspective
  distortion in fronto-parallel device-rendered tick marks.
- **No ControlNet / diffusion realism pass** for synthetic data. Our marks
  are stamped onto real US pixels, not synthesized from scratch onto random
  ImageNet backgrounds — the domain-gap problem RulerNet solves with
  ControlNet doesn't apply the same way.
- **No global OSF residual mm blend** (v9 mistake, stays off by default per
  existing project rule).

## References

- Pan, Y., Mehta, M., Sincerbeaux, G., Goldstein, J.A., Gernand, A.D., Wang,
  J.Z. *RulerNet: Learning Perspective-Invariant Ruler Representations for
  Robust Image Scale Estimation.* arXiv:2507.07077v2. DOI:
  10.1016/j.compmedimag.2026.102805.
- Ritsche, P., Cronin, N.J., et al. *DL_Track_US: Fully automated analysis of
  muscle architecture from B-mode ultrasound images with deep learning.*
  Ultrasound in Medicine & Biology (2024); JOSS 10.21105/joss.05206. (Already
  vendored: `src/muscle_arc/models/dl_track_official`.)
- Wang, J. et al. *Deep High-Resolution Representation Learning for Visual
  Recognition* (HRNet). IEEE TPAMI 43, 2020 — heatmap keypoint architecture
  RulerNet builds on and we reference for the CE+DICE heatmap convention.
