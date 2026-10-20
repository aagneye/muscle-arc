# Curator climb plan — own scale detector + gated train

## Problem

Public LB ~**0.953** (v10c). Gap to top-10 (~0.40) is mostly **mm/px**, not segmentation.
TIFF metadata is dead (0/309). Heuristic OCR covers console shots; cropped frames need ticks.
Console ticks in our code latch on colour bars; competitor duty-cycle autocorr hit **255/309**.

## Own detector (so we can train a lot)

Two layers, now three:

1. **Heuristic ruler** (`geometry/depth_scale.py`) — duty-cycle pitch + OCR/tick fusion.
   Fast, no GPU, used at infer and to mint **pseudo-labels**.
2. **Trainable scale head** (`models/scale_detector.py` + `scripts/train_scale_detector.py`) —
   small CNN on border strips → `mm_per_pixel`. Trained on:
   - OCR/tick pseudo-labels from test/train frames (high-conf only)
   - OSF expert `Scale_pixel_per_cm` when available
   Retrain anytime labels improve; ensemble with heuristic at infer.
3. **Tick-keypoint reader** (`models/tick_keypoint.py` + `scripts/train_tick_keypoint.py`) —
   RulerNet-style (Pan et al., arXiv:2507.07077v2; see `docs/research_scale_reader.md`)
   heatmap keypoint CNN over border-strip profiles, trained purely on synthetic
   ticks (`geometry/synth_ticks.py`) stamped onto real US backgrounds — zero
   manual labels, ground truth exact by construction. `geometry/scale_fusion.py`
   combines all three layers (plus OSF shape lookup, opt-in residual GBDT) into
   one confidence-weighted jury vote, replacing the old hardcoded if/elif
   priority chain. Wired into `calibrate_predict.py` / `eval_osf_pipeline.py`
   via optional `--tick-keypoint-ckpt` (off by default until a checkpoint is
   trained and validated on OSF).

Segmentation stays as apo/fasc U-Nets (`grow` / `grow_v2`) — train those when Gate1 says
`geometry_first`.

## Session order

1. Start `rogii-gpu`; keep v8 / v10c rollback.
2. Ship duty-cycle console ticks + OCR fusion (atomic commits). ✅
3. Ship trainable scale detector + train script (atomic commits). ✅
4. Ship synthetic tick generator + TickKeypointNet + jury fusion (atomic
   commits). ✅ code shipped; checkpoint not yet trained/validated on VM.
5. Train `TickKeypointNet` on VM (`scripts/train_tick_keypoint.py`, synthetic
   data, cheap — no GPU-hour concern per project owner).
6. Gate1 route → grow fasc only if geometry-first.
7. Audit depth coverage; validate the jury (`--tick-keypoint-ckpt`) beats the
   heuristic-only baseline on OSF Gate2b before enabling by default.
8. Infer candidate; hard Gate2+Gate3; SUBMIT or HOLD.

## Gate2 protocol A/B result (2026-09-16)

Ran `eval_osf_pipeline.py --masks-from dl_track --scale-mode gt --geom ours`
on all 35 real OSF expert-GT images (DL_Track VGG16 pretrained masks,
GT scale — isolates geometry from scale error):

| Protocol   | PA MAE | FL MAE  | MT MAE  | UMUD  |
|------------|--------|---------|---------|-------|
| `legacy`   | 1.661° | 7.04 mm | 1.46 mm | **0.450** |
| `host_v1`  | 1.919° | 9.87 mm | 1.55 mm | 0.553 |

`host_v1` (mean-of-3 host-protocol convention) is **worse on every
parameter** vs `legacy` (median-of-many, DL_Track-style extrapolation) —
23% worse UMUD overall, FL MAE 40% worse. Likely cause: mean-of-3 is far
less robust to imperfect (non-GT) masks than the legacy path's
many-sample median + Hough/Radon fallback design. **Decision: keep
`legacy` as the default.** `host_v1` code stays available behind
`--geometry-protocol host_v1` for future reference/debugging but is not
adopted. Full JSON: `experiments/gate2_legacy.json`,
`experiments/gate2_host_v1.json`.

## Gate2b tick-keypoint validation (2026-09-16) — INCONCLUSIVE on OSF

Ran `eval_osf_pipeline.py --scale-mode pred` with and without
`--tick-keypoint-ckpt experiments/checkpoints_tick/tick_keypoint.pt`.
Results were **byte-identical** (UMUD 0.484 both runs) — not because the
flag is broken, but because on 5/6 sampled OSF images the model's own
`predict_pitch_px` correctly returns `(None, 0.0)`: these OSF benchmark
images don't have visible ruler/tick marks in the border-strip region the
model looks at (they're cropped B-mode frames, not console screenshots
with chrome). With zero confidence, the tick-keypoint candidate never
wins the `scale_fusion` jury vote against `osf_shape` (0.88 conf), which
is correct behavior, not a bug — verified by direct `predict_pitch_px`
calls returning `pitch_px=None` for images 1-5 and a low-confidence
`(20.0px, 0.43)` for image 6.

**Conclusion: OSF is not a representative test bed for TickKeypointNet**
(it targets console/ruler-chrome images, which OSF's expert-analysed set
mostly lacks) — Gate2b as specified cannot validate this specific model on
this specific dataset. Per the v9-lesson discipline, this means
`--tick-keypoint-ckpt` **remains NOT validated and NOT enabled by
default** — the absence of evidence-of-improvement is not evidence of
no-improvement, but the hard gate requires proof of improvement before
adoption, and that proof isn't obtainable from OSF alone. Would need
either: (a) the actual Kaggle test set (which does have console/ruler
chrome per earlier TIFF audits) run through Gate2b-equivalent evaluation
with sample GT, or (b) a different benchmark with ruler chrome present.

## Do not

- Global OSF residual mm blend (v9 → 1.06).
- Soft Gate2 as submit bar.
- Submit on n=2 sample MAE alone.
- Enable `--tick-keypoint-ckpt` by default before it has been validated to
  improve (not just match) the existing heuristic on OSF Gate2b — same
  discipline as the residual-regressor rule above.

## 2026-10-18 public-LB results (dual_v3 pipeline)

| Submission | Change vs previous best | Public |
|---|---|---|
| dual_v3 + exact scale + fixed 5-frame clips | new dual U-Nets (dedup splits), deterministic px/cm | 0.46010 |
| + PA +1.9° | pennation offset toward DL_Track insertion convention | **0.44273** (rank 120/433) |
| S3 | FL = mean of 3 least-extrapolated fragments, 25 % trig, OSF-calibrated (FL -6 mm avg) | 0.47343 |
| S4 | PA without Radon blend (+2°) | 0.44684 |

Lesson: the 35-image OSF benchmark (six raters, cropped GM/SOL/VL) is NOT a reliable
selector for the test raters (two raters, console frames, VL/GM/RF incl. CP). On OSF the
least-extrapolation FL and no-Radon PA both won (OSF UMUD 0.199) and both lost on public.
Test raters read PA about 2° steeper and FL longer than the OSF raters. Keep the 0.44273
geometry; gains must come from mask quality (5-fold ensemble, `scripts/vm_train_folds.sh`,
`scripts/vm_post_k5.sh`).
| k5 | 5-fold ensemble (dedup group splits), 0.4427 geometry | 0.41936 (rank 107/434) |
| k10 | 2 seeds x 5 folds (20 models), same geometry | **0.40307** (2026-10-20) |

Next: `docs/competition_intelligence.md` (leaders, discussions, rules) and
`docs/superpowers/plans/2026-10-18-top10-strategy.md` (Line A automated, Line B declared
manual test labels + calibration, Line C merger).
