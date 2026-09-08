# Method design

## Primary pipeline

**Segmentation → geometry → PA / FL / MT** (not direct end-to-end regression).

Reason: training labels are apo/fasc masks; architecture parameters are defined geometrically. Segmentation also supports classical post-processing and temporal smoothing on video frames.

Optional later: light regression head or GBDT on geometric features + image embeddings for residual correction / ensemble.

## Models

Two binary U-Nets (or one multi-task net with two heads):

| Branch | Target | Used for |
|--------|--------|----------|
| Apo | superficial + deep aponeurosis mask | MT (and apo orientation for PA) |
| Fasc | fascicle fragments | PA orientation, FL length |

- Architecture: `smp.Unet`
- Encoder: `tu-efficientnet_b4` (ImageNet) — strong single-GPU default
- Upgrade path: `tu-efficientnet_b5` / `mit_b2` if VRAM allows
- Loss: BCE + Dice; optional Tversky if thin fascicles under-segment

## Geometry (from masks)

Assume image coordinates with +y downward. Calibrate `mm_per_pixel` from known scale or label-derived stats.

**MT** — Fit superficial and deep apo polylines (or top/bottom apo bands). Thickness = median perpendicular distance between the two lines × `mm_per_pixel`.

**PA** — Fit fascicle direction (PCA / `cv2.fitLine` on fascicle pixels) and deep apo orientation. Pennation = acute angle between fascicle direction and deep apo (degrees).

**FL** — Intersect representative fascicle ray with superficial and deep apo; Euclidean distance between intersection points × `mm_per_pixel`. Fallback: extent of fascicle blob along its major axis if intersections fail.

Clip to physiological ranges (see `configs/default.yaml`).

## Scale calibration

Absolute FL/MT require mm-per-pixel. TIFF audit (`scripts/audit_tiff_scales.py`) found
**0/309 usable scales on `test_images_v2`** — every candidate tag was classic 72 DPI
placeholder (`mm/px ≈ 0.353`). Decision: **Phase A′** (no metadata-first path on this set).

Pipeline:

1. Audit TIFF/ImageJ tags; reject placeholder 72 DPI at write time.
2. Prefer usable metadata when present (`muscle_arc.geometry.scale.assign_scales`).
3. Else richer clustering (aspect + height + brightness) fit from the 2 labeled
   `sample_submission` rows.
4. **Phase A′**: train Huber/HistGBDT residual models on UMUD OSF expert architecture
   benchmarks (`Scale_pixel_per_cm` + median R1–R7 PA/FL/MT) via
   `scripts/train_residual_scale.py` → `experiments/residual_models/`. Applied in
   `calibrate_predict.py` for non-metadata rows (`--residual-prefer blend|mm|scale`).
5. Small residual gain correction on sample GT (clamped).

### External public data (challenge-allowed)

- OSF node `xbawc` — Expert Analysed Benchmark Image Datasets
  (`scripts/download_osf_benchmarks.py`)
- Architecture set v0.1.0: 35 images, four devices, expert MT/FL/PA + known
  `Scale_pixel_per_cm` → `experiments/osf_expert_gt.csv`
- Local MAE gate: `scripts/eval_umud_local.py` (sample GT + optional `--external-gt`)

Domain-shift risk: OSF muscles/devices may not match the Kaggle test mix; LOO/CV MAE on
OSF is a necessary but not sufficient gate before submit.

**v9 lesson (2026-09-05):** applying OSF direct-mm blend to all 309 test rows
over-compressed FL/MT (std collapsed) and scored **1.062** vs v8 **0.999**.
Local sample MAE (n=2) improved while LB worsened — do not trust Gate-B alone.
Residual defaults to `--residual-prefer off`; use only experimentally on
`primary`/`cluster_prefix` rows. Keep v8 as rollback. Next: Phase B ensemble/TTA
while refining A′ (scale-only soft prior, not direct mm replace).

## Training

- Separate datasets for apo vs fasc (different image counts)
- Ultrasound augs: brightness/contrast, gamma, blur/sharpen, mild elastic, horizontal flip (flip carefully if later using absolute orientation priors)
- Resize/pad to `img_size` (512)
- Hold out ~15% by filename/group for val; if video IDs exist, split by sequence
- AMP + AdamW

## Temporal stability

Test set includes 5-frame sequences. After per-frame prediction, optionally median/EMA-smooth PA/FL/MT within each group of 5 to reduce jitter without harming independent frames.

## Module map

| Module | Role |
|--------|------|
| `muscle_arc.data.paths` | Resolve Kaggle folder layout |
| `muscle_arc.data.dataset` | Image/mask datasets + augs |
| `muscle_arc.models.segmentation` | Build SMP U-Net |
| `muscle_arc.geometry.metrics` | MT / PA / FL from masks |
| `muscle_arc.train.loop` | Train/val loop + checkpoints |
| `muscle_arc.infer.predict` | Masks → params → CSV |
| `scripts/*` | CLI entrypoints |

## Milestones

1. Download data + EDA notebook; baseline geometry on GT masks (sanity)
2. Train apo + fasc segmenters; first valid Kaggle CSV
3. Scale calibration, TTA, temporal smooth, encoder sweep
