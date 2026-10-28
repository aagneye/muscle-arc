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

## 2026-10-22: factory mode, goal public < 0.23

- Best public **0.40307** (k10, rank 112/466); rank 10 = 0.2225.
- Built the declared manual-measurement line (host-allowed external data): `tools/labeler/`
  served by `scripts/label_server.py` on the VM (`scripts/vm_label_server.sh`, reach via
  `ssh -N -L 8765:127.0.0.1:8765 azureuser@<vm>`), saves to `data/labeling/labels.json`
  (git-ignored; public repo). 203 main images (clip middles), 106 optional clip frames.
- `scripts/eval_local.py` = local replica of the public score against those measurements;
  candidates are ranked locally before Kaggle.
- Seed 3 of the dual U-Net ensemble queued (`configs/dual_v3_s3.yaml`).

## 2026-10-26: leaderboard probe decomposition of k10 (0.40307)

Five one-parameter-shift probes of `cand_k10.csv` (PA +-100 deg, FL +-1000 mm, MT +100 mm
scored 5.85605 / 5.81660 / 27.90299 / 27.98273 / 11.44966). Decoded with `scripts/lb_probe.py`:

| Parameter | MAE | Bias | Share of score |
|---|---|---|---|
| PA | 2.20 deg | +0.36 deg | 30 % |
| FL | 8.57 mm | -1.44 mm | 59 % |
| MT | 0.385 mm | -0.20 mm | 11 % |

Lesson: biases are worth about 0.003 in total, so offsets and global calibration are
finished; the loss is per-image variance, dominated by FL, which inherits the PA error
through the linear extrapolation (2.8 to 5 mm per degree at PA 15 to 20 deg). MT is at rater
repeatability. Full analysis and plan: `docs/score_loss_analysis.md`. Queued: per-layout FL
probes (siemens_right_ruler, left_ruler_5cm), k15 build, 768-px training line.

### Built on 2026-10-26 (no new public scores: the five daily submissions went to the probes)

- `scripts/lb_probe.py`: probe files and exact decode (public subset is about 33 % of
  the test set, N about 102; a full-set shift cancels N).
- `scripts/geometry_sweep.py`: `fl_contour` (contour route without the internal 0.5 trig),
  pure trig variants, `--smooth-w`; re-sweeping k10 reproduces `cand_k10.csv` exactly.
- `scripts/fuse_measurements.py`: measurement-level fusion (per-image median across runs)
  with a between-run spread per image, following the challenge evidence in
  `docs/research/2026-10-26-rater-calibration-and-fusion.md`.
- `scripts/calibrate_to_labels.py` + `geometry/calibration.py`: per-device gain/offset
  with partial pooling, gated by leave-one-out, maps saved as JSON (ready for the manual
  measurements; none made yet).
- `scripts/train.py`: `tversky_skelrecall` fascicle loss (`configs/dual_v3_skel.yaml`);
  `configs/dual_v3_768.yaml`; `scripts/vm_train_line.sh` queue launcher with retries.
- Running on the VM: k15 build, per-seed caches (s2, s3), 768-px fascicle line (GPU 0 after
  the caches), skeleton-recall fascicle line (GPUs 1-3 when the other project frees one).
- Scheduled 2026-10-27 00:05 UTC (`scripts/vm_submit_batch.sh`, log
  `logs/submit_batch_1027.log`): `cand_k15.csv` + FL +-10000 probes on siemens_right_ruler
  and left_ruler_5cm (base k10). Day after: k15 FL variants (`fl_contour|trig0.5`,
  `fl_contour|trig0`, `fl_legacy|trig0.25`), clip median, `cand_med3.csv`.

## 2026-10-26 (later): visual audit finds an aponeurosis-pairing bug (fixed, behind a flag)

`scripts/overlay_audit.py` renders what the geometry sees (bands, fitted lines, every fascicle
extrapolation). Per-layout images where the FL estimators disagree most showed that on
multi-band frames (two stacked muscles, or skin / bone bands detected as aponeurosis):

- the per-column surface fit rejected any muscle thicker than 0.42 of the frame, so clip
  IMG_00121-00125 measured the muscle *below* (MT 14.5 mm instead of 24.5 mm, FL 70 vs 128 mm);
- it mixed band pairs across columns (IMG_00047: diagonal lines through no aponeurosis);
- `_apo_bands` took a 30-px fragment as the deep band (IMG_00045, 00048, 00167, 00303) or
  skipped a long middle aponeurosis (IMG_00047, PA 21.1 -> 15.7 deg after the fix).

Fix: `_select_band_pair` (`APO_PAIRING = "upper_dense"`, `geometry_sweep.py --apo-pairing`):
bands spanning >= 20 % of the apo extent, vertically adjacent pairs only, physiological gap,
uppermost pair with >= half the best fascicle density, used by both PA and the line fit.

| k10 cache | legacy | upper_dense |
|---|---|---|
| OSF MT MAE | 0.420 mm | 0.205 mm |
| OSF PA+2 MAE | 2.00 deg | 1.85 deg |
| OSF FL MAE (fl_legacy) | 4.29 mm | 3.80 mm |
| OSF UMUD (pick) | 0.277 | 0.231 |
| Agreement with the 0.451 anchor | 0.498 | 0.473 |

All 5 OSF images that changed improved (e.g. im_07 MT error 5.4 -> 0.05 mm); 11 of 309 test
images change materially, all in multi-band layouts; the between-seed FL spread p99 fell from
14.4 to 6.1 mm. This is a bug fix, not a convention, so OSF is a valid check here. Default
stays legacy until the board confirms. Batch 2026-10-27 00:05 UTC: `cand_k10ud` (A/B vs
0.40307), `cand_k15ud`, `cand_med3ud` (median of the 3 seeds' measurements, same 15 fold
models as k15), FL +-10000 probes on siemens_right_ruler.
- Null result: dropping fascicle fragments that lean against the pennation (majority-lean
  filter) moves PA by 0.01 deg on average (3 of 309 test images > 0.5 deg, OSF unchanged);
  opposite-lean fragments are rare in our masks and do not explain the +2 deg offset. Reverted.
- Labeler quick mode: one line of each kind per image, to get more images measured.

## 2026-10-27 public results

| Submission | Change | Public |
|---|---|---|
| `cand_k10ud` | k10 + upper_dense aponeurosis pairing (A/B vs k10 0.40307) | 0.38457 |
| `cand_k15ud` | k15 (30 models, map-averaged) + pairing fix | 0.38652 |
| `cand_med3ud` | median of the three seeds' measurements (same 15 fold models as k15) + fix | **0.38391** (best) |
| probe FL +10000 on siemens_right_ruler (base k10) | diagnosis | 86.69086 |
| probe FL -10000 on siemens_right_ruler (base k10) | diagnosis | 86.54941 |

Lessons:
- The pairing bug fix is worth -0.0185 on its own (11 test images changed).
- With the fix, map-averaging 30 models (k15) is no better than 20 (k10); fusing the three seeds at
  the measurement level beats map-averaging the same 15 models by 0.0026, as the challenge
  literature predicted.
- The public board scores exactly 103 images (309 / 3): with N = 103 the siemens subgroup size
  comes out integer (32), with 100-106 otherwise it does not. `scripts/lb_probe.py` default fixed.
- **FL error is mostly per-device bias, not scatter.** Siemens (32 public images): FL MAE 9.60 mm,
  bias +8.20 mm (our FL too long). Since all-image FL bias is -1.44 mm, the other 71 public images
  have bias -5.78 mm (too short) and MAE 8.10 mm. On Siemens frames MT / sin(submitted PA)
  averages 74 mm vs our 83 mm, matching the implied rater mean (~75 mm); on the other devices
  the raters measure longer than any of our estimators. The earlier "biases are exhausted"
  reading held only for the all-image mean, where the device biases cancel.
- Implication: per-device calibration from the owner's manual measurements (declared external
  data, `scripts/calibrate_to_labels.py`) can remove most of the FL error; the labeler queue
  already interleaves devices (first 60 images = about 10 per device). Hard-coding per-device
  offsets read off the board would break the "no leaderboard values" guardrail: owner's call.

## 2026-10-28

| Submission | Public | Note |
|---|---|---|
| cand_med5ud (median of k5ud, s2ud, s3ud, k768ud, kskelud) | **0.37731** | new best; -0.0066 vs med3ud |
| FL +/-10000 on left_ruler_5cm | 83.88120 / 83.99584 | n=31, FL MAE 8.08 mm, bias -6.86 mm |
| FL +/-10000 on png_left_ruler | 54.25290 / 54.34047 | n=20, FL MAE 8.12 mm, bias -8.12 mm |

Lessons:
- More diverse runs (768-px fascicles, skeleton-recall loss) help through measurement-level median fusion: med5 beats med3 by 0.0066.
- Per-device FL bias map (k10, public): siemens +8.20 (n=32), left_ruler_5cm -6.86 (n=31), png_left_ruler -8.12 (n=20). On png the error is almost all bias (MAE 8.12 = |bias|). The remaining 20 public images (bottom_ruler_853/512, fixed_644x1088) are not probed yet.
- Removing these biases is the largest available gain. Pipeline correction must come from the owner's manual labels (labeler quick mode), not from these leaderboard numbers.
