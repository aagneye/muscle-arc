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

## Do not

- Global OSF residual mm blend (v9 → 1.06).
- Soft Gate2 as submit bar.
- Submit on n=2 sample MAE alone.
- Enable `--tick-keypoint-ckpt` by default before it has been validated to
  improve (not just match) the existing heuristic on OSF Gate2b — same
  discipline as the residual-regressor rule above.
