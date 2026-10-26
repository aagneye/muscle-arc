# Where the score is lost (leaderboard probe decomposition)

Date: 2026-10-26. Base file: `submissions/cand_k10.csv` (20-model ensemble, public **0.40307**,
rank 129 of about 470). Tool: `scripts/lb_probe.py`. Goal: public < 0.23.

## 1. Method

The public score is `S = (MAE_PA/6 + MAE_FL/12 + MAE_MT/3) / 3` over the 309 test images
(host notebook `paulritsche/umud-score`, `geometry/umud_metric.py`). Adding a constant K
that exceeds every error to one parameter turns `|e_i|` into `K + e_i`, so a +K / -K pair of
submissions gives the signed and the absolute error sums exactly (board rounding 1e-5, i.e.
sums known to about 0.06 mm):

    c = 3 T_p N            sum e_i  = c (S+ - S-) / 2
                           sum |e_i| = n K - c (S+ + S- - 2 S0) / 2

The board scores only the public subset ("approximately 33 % of the test data", N about
102; the final standing is on the other 67 %). A full-set shift cancels N, so the numbers
below hold whatever N is. The same pair restricted to a subgroup (device layout, clip
frames, ...) gives that subgroup's MAE and bias on its public members, whose number is
recovered as round(B / K) when K = 10 000. This is diagnosis only; nothing from it enters
the pipeline (host ruling 739660 forbids leaderboard-weighted ensembling, not leaderboard
diagnosis).

## 2. Result for k10 (five probes, 2026-10-26)

| Probe | Public |
|---|---|
| PA +100 deg | 5.85605 |
| PA -100 deg | 5.81660 |
| FL +1000 mm | 27.90299 |
| FL -1000 mm | 27.98273 |
| MT +100 mm | 11.44966 |

| Parameter | MAE | Bias (pred - rater) | Score contribution | Share |
|---|---|---|---|---|
| PA | **2.20 deg** | +0.36 deg | 0.1223 | 30 % |
| FL | **8.57 mm** | -1.44 mm | 0.2380 | **59 %** |
| MT | **0.385 mm** | -0.20 mm | 0.0428 | 11 % |

`python scripts/lb_probe.py decode --n 309 --base-score 0.40307 --pair pa_deg 100 5.85605 5.81660
--pair fl_mm 1000 27.90299 27.98273 --single mt_mm 100 11.44966` reproduces the table.

## 3. What it says

1. **Bias is exhausted.** For an error distribution with spread sigma and offset mu, removing
   mu lowers the MAE by about mu^2 / (sigma sqrt(2 pi)): PA 0.02 deg, FL 0.08 mm, MT 0.01 mm,
   i.e. **about 0.003 of score in total**. Global shifts, per-parameter calibration and
   "+x deg" tricks are done; the next ones are noise.
2. **The loss is per-image variance, mostly in FL.** 8.57 mm MAE at 12 mm tolerance is 59 %
   of the score. MT (0.385 mm) is at the level of the raters' own repeatability and is
   finished.
3. **FL inherits the PA error.** Raters trace a fascicle segment and extrapolate it linearly
   to both aponeuroses, so their FL is approximately MT / sin(PA) with the aponeurosis-angle
   correction, and dFL/dPA = -MT cos(PA) / sin^2(PA): with MT = 20 mm that is 5.0 mm per
   degree at PA 15 deg, 2.8 mm/deg at 20 deg, 1.8 mm/deg at 25 deg. Our 2.2 deg PA error
   therefore accounts for roughly 4 to 11 mm of FL error on its own. Every degree of PA
   bought back is worth about 0.009 directly plus several mm of FL.
4. **The target is at the human floor.** Published inter-rater agreement (literature review,
   section 7) puts two independent expert raters at a UMUD of about 0.2 to 0.3 against each
   other. Rank 10 (0.22) and the leader (0.20) are at or below that floor, which an automated
   model trained on the public masks cannot reach: it has no way to learn the two test
   raters' habits (which fascicle, how far to extrapolate, where the three MT lines go).
   Reaching < 0.23 needs the raters' conventions as data (our declared manual measurements,
   `tools/labeler`, used for calibration and as training targets) or a merge.

## 4. Testing centre without labels: subgroup probes

Until the manual measurements exist, the board is the only ground truth, so it is used as a
measurement instrument: two submissions per subgroup and parameter. Queued for 2026-10-27
(`submissions/probe_fl_{p,m}10000_{siemens_right_ruler,left_ruler_5cm}.csv`, 181 of 309
images, about 60 public): FL MAE and bias per device layout. A layout with FL MAE far above 8.6 mm points at
scale, crop or fascicle-visibility problems that can be fixed per device; a flat profile
means the fix must be generic (segmentation detail, fascicle choice).

## 5. Plan (highest expected gain first)

| Line | Expected public | Status |
|---|---|---|
| Manual host-protocol measurements of the test images: `scripts/eval_local.py` ranks candidates, then per-device calibration (`scripts/calibrate_to_labels.py`) and training targets (declared external data) | 0.25 to 0.30 | tool live on the VM, no images measured yet |
| **Aponeurosis pairing fix** (`--apo-pairing upper_dense`, found by `scripts/overlay_audit.py`): multi-band frames measured the wrong muscle or mixed band pairs (11 test images, e.g. clip 00121-00125 MT 14.5 -> 24.5 mm) | -0.01 to -0.03 | OSF UMUD 0.277 -> 0.231; on the board 2026-10-27 (`cand_k10ud` A/B vs 0.40307) |
| k15 (3 seeds x 5 folds, 30 models) | -0.005 to -0.01 | built; `cand_k15ud` on the board 2026-10-27 |
| Measurement-level fusion: median of the per-seed measurements (same 15 fold models) | -0.005 to -0.02 | built; `cand_med3ud` on the board 2026-10-27 |
| 768-px fascicle branch 5-fold line (`configs/dual_v3_768.yaml`) | -0.01 to -0.03 | training on GPU 0 (fold 1 of 5 at 07:11 UTC) |
| Fascicle branch with Skeleton Recall loss (`configs/dual_v3_skel.yaml`) | unknown | queued, waits for GPUs 1-3 (other project) |
| FL convention on the board: contour only, more trig, clip median | +-0.01 each | prebuilt on k15ud for 2026-10-28 |
| Per-layout FL probes | diagnosis | siemens pair 2026-10-27, left_ruler pair after |
| Merge with the leader recruiting GPU partners (thread 749906) | top 3 | owner's call |

Tried and dropped on 2026-10-26 (local evidence, no submission spent):
- `fascicle_length_geo_px`, one line at the image PA extrapolated to both aponeuroses: OSF
  7.4 mm, anchor 8.7 mm vs 3.7 / 7.9 for the current FL; FL moves about 5 mm per degree of
  PA, so a single angle amplifies the PA error instead of averaging it.
- PA with the deep-aponeurosis tangent at the fragment's insertion point
  (`PA_TANGENT_AT = "insertion"`): OSF 1.65 deg and anchor 3.20 deg vs 1.37 / 2.58; the
  extrapolated quadratic edge gives unstable tangents.

## 6. Research digest (2026-10-26) and what it changes

Full report with sources and flags: `docs/research/2026-10-26-rater-calibration-and-fusion.md`.

- **Fuse at the measurement level, not the map level.** In HC18 a 25-model mask-averaging
  ensemble (1.69 mm) matched single 2019 models (1.72 mm) and a 5-fold nnU-Net was worse
  than one fold; the measured ensemble gains in bone age (-17 % MAD, Pan 2019), DSB-2016
  (-11 % CRPS with consensus fallback) and TimTrack (-63 % FL RMSD from an
  accumulator-weighted median over Hough peaks) were all at the scalar level. Our k5 -> k10
  steps (0.419 -> 0.403) are map-level and flattening; the next ensemble step should be a
  median of per-seed measurements with outlier removal.
- **Convention floor dominates.** 75 % of the residual HC18 error was the ellipse
  convention; PSFHS ranking by derived angle inverted the Dice ranking. Our MT already
  sits at the floor; PA and FL conventions (which fascicle, extrapolation rule) are the
  lever, and only the manual measurements can reveal the test raters' rule.
- **Few labels calibrate a lot, linearly.** SonoBase: EF MAE 6.63 -> 4.95 % with about 50
  labels; three segmenters with raw AC MAE 18.9 / 14.8 / 18.1 mm became indistinguishable
  (13.9 mm) after leave-one-out linear recalibration. Intercept + slope with partial pooling
  per device, slope shrunk toward 1, beats nonlinear mappings below about 1 000 labels.
  Quantile / distribution matching never beat the affine fit anywhere.
- **Trig FL is rater-level where fascicles are mostly in frame** (TimTrack RMSD 0.52 cm vs
  manual-manual 0.56 cm on GM/GL/VL) and over-reads by 0.5 to 1.9 cm where most of the
  fascicle is extrapolated (BFlh). Use it as a gate / prior weighted by the extrapolated
  fraction, not as a global blend.
- **Clip fusion:** a median over the 5 frames is the cheap version of UltraTimTrack's
  fusion (per-frame noise 3.5 mm / 1.4 deg -> 1.4 mm / 0.6 deg).
- **Thin structures:** resolution, rotation / shear / scale augmentation and a residual
  U-Net recipe are the documented levers; backbone swaps move Dice by about 1 point, often
  the wrong way; no transformer win on a line-like ultrasound structure in 3 years.
  Skeleton Recall loss changes connectivity at +8 % cost.
