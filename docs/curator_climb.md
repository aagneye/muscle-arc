# Curator climb plan — own scale detector + gated train

## Problem

Public LB ~**0.953** (v10c). Gap to top-10 (~0.40) is mostly **mm/px**, not segmentation.
TIFF metadata is dead (0/309). Heuristic OCR covers console shots; cropped frames need ticks.
Console ticks in our code latch on colour bars; competitor duty-cycle autocorr hit **255/309**.

## Own detector (so we can train a lot)

Two layers:

1. **Heuristic ruler** (`geometry/depth_scale.py`) — duty-cycle pitch + OCR/tick fusion.
   Fast, no GPU, used at infer and to mint **pseudo-labels**.
2. **Trainable scale head** (`models/scale_detector.py` + `scripts/train_scale_detector.py`) —
   small CNN on border strips → `mm_per_pixel`. Trained on:
   - OCR/tick pseudo-labels from test/train frames (high-conf only)
   - OSF expert `Scale_pixel_per_cm` when available
   Retrain anytime labels improve; ensemble with heuristic at infer.

Segmentation stay as apo/fasc U-Nets (`grow` / `grow_v2`) — train those when Gate1 says
`geometry_first`.

## Session order

1. Start `rogii-gpu`; keep v8 / v10c rollback.
2. Ship duty-cycle console ticks + OCR fusion (atomic commits).
3. Ship trainable scale detector + train script (atomic commits).
4. Gate1 route → grow fasc only if geometry-first.
5. Audit depth coverage; train scale detector on VM.
6. Infer candidate; hard Gate2+Gate3; SUBMIT or HOLD.

## Do not

- Global OSF residual mm blend (v9 → 1.06).
- Soft Gate2 as submit bar.
- Submit on n=2 sample MAE alone.
