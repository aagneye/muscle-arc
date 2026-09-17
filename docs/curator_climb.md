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
