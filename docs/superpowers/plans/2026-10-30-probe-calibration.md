# Per-scanner calibration from leaderboard probes (2026-10-30 to 11-14)

Goal: public < 0.23. State on 2026-10-30: **0.29604, rank 58 of 545** (rank 20 = 0.230,
rank 10 = 0.208) with `cand_med6ud_dc1` = median of 6 runs + `configs/device_calibration.yaml` v1.

## Why this line

The probes (`scripts/lb_probe.py`) showed that most of the remaining error is a fixed offset per
device layout (FL siemens +8.2 mm, left_ruler_5cm -6.9, png -8.1; PA png +3.3 deg). One set of
per-layout constants took 0.377 to 0.296. Ensembling more runs had stopped paying
(med5 0.37731, med6 0.37699). The owner approved probe-derived constants on 2026-10-30;
host ruling 739660 allows hand-set constants as long as the code re-runs without a
leaderboard (CLAUDE.md guardrail reworded accordingly).

## Method

- A +/-K pair on one parameter of one layout gives that layout's public n, MAE and bias.
  Use K = 10000 for FL and K = 1000 for PA, so that n can be recovered.
- FL is corrected multiplicatively: `fl_mul = 1 - bias / layout mean`. PA and MT are corrected
  additively.
- A correction is applied only when |bias| exceeds the noise level: 2 mm for FL, 0.4 deg for PA.
- A new estimator (an FL variant) inherits the calibration through `--match-means`, which matches
  the calibrated candidate's per-layout means; no new probes are needed.
- Per-image variance: `*_shrink` pulls each image toward its layout mean. The expected gain is
  -0.005 to -0.015 (see the 10-30 analysis below).

## Automated schedule on the VM (one-off waiters in `logs/`)

| Day (00:05 UTC) | Submissions | Then |
|---|---|---|
| 10-31 | FL +/-10000 fixed_644x1088, PA +/-1000 siemens (base dc1) | `post_1031.sh` decodes them, writes v2 (`experiments/device_calibration_v2.yaml`) and submits dc2 in slot 5 |
| 11-01 | FL +/-10000 bottom rulers (pooled), PA +/-1000 fixed (base = better of dc1/dc2) | `post_1101.sh` writes v3 and submits dc3 in slot 5 |
| 11-02 | E_var (FL `fl_le5\|trig0.5`, matched means), E_fls (FL shrink 0.8 on siemens and left_ruler), E_pas (PA shrink 0.75 on siemens, left_ruler and png), PA +/-1000 bottom rulers (base = best of dc1..dc3) | `post_1102.sh` builds dc4 = best + bottom PA + each experiment that beat the base; queued for 11-03 00:05 |

## Still to do after 11-03

1. Commit the winning per-layout constants into `configs/device_calibration.yaml` (v2 to v4
   exist only on the VM).
2. MT +/-K per layout. MT is 11 % of the score, and a per-layout MT bias is plausible (for
   example, the pairing fix moved fixed MT by +1.0 mm).
3. Check the residual FL on siemens and left_ruler on the calibrated base. The k10 to med6
   shift and the multiplicative form were both approximations.
4. Within-layout structure: probe on the upper half of predicted FL for the siemens and
   left_ruler layouts to estimate the slope (shrinkage) directly.
5. Clip smoothing strength (w = 1, the clip median, vs 0.5).
6. Final two submissions: the best calibrated one, plus an uncorrected med6ud as a hedge.

## Expected-value notes (2026-10-30)

- Expert benchmark (35 images, raters R1 to R7) with the bias removed: FL `fl_le5|trig0.5` has a
  centred MAE of 2.72 mm vs 3.19 for the current `fl_legacy|trig0`. PA `pa_noradon` scores 1.15
  vs 1.30 deg, but on test it correlates 0.98 to 0.999 with pa_legacy, so it was not tested.
- Within-layout prediction spreads give these optimal shrink factors: about 0.8 for FL (siemens,
  left_ruler), 0.5 to 0.75 for PA, and none for png FL (its residual is small).
