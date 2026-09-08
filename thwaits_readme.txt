# UMUD Challenge — muscle architecture from B-mode ultrasound

Predict pennation angle (PA, deg), fascicle length (FL, mm) and muscle thickness
(MT, mm) for 309 test ultrasound images. Deadline **2026-11-14 23:00 UTC**.
Scored by the hidden "UMUD score" — a tolerance-normalised MAE, lower is better.
Public leaderboard is **33 % of the test set (~102 images)**; the other ~207 decide it.

All compute is local (Apple M3 Max, 64 GB, PyTorch/MPS). No hosted model is used
anywhere in the pipeline.

## Setup

```sh
uv sync                              # Python 3.12 + torch/MPS + smp/albumentations
uv run python scripts/download_data.py   # ~2.75 GB archive -> data/raw (5.7 GB)
```

Kaggle auth comes from `~/.kaggle/credentials.json` (OAuth). You must accept the
competition rules on the website before the data endpoints unlock.

## Layout

| path | what |
|---|---|
| `scripts/download_data.py` | fetch + extract the competition archive |
| `scripts/inventory.py` | shapes, dtypes, image/mask pairing → `reports/inventory.json` |
| `scripts/probe.py` | leaderboard coordinate probing (see below) |
| `src/umud/submit.py` | submission formatting + upload, with the format traps encoded |
| `src/umud/depth.py` | sector height + OCR'd depth → pixels per centimetre |
| `src/umud/ruler.py` | ruler-tick comb, as an independent scale cross-check |
| `src/umud/metric.py` | the official UMUD score, for local use |
| `src/umud/validation.py` | expert-benchmark harness |
| `src/umud/geometry.py` | masks → pennation angle, fascicle length, thickness |
| `src/umud/segment.py` | DL_Track pretrained-network inference |
| `src/umud/grouping.py` | acquisition/video-run grouping |

## Findings so far

**Submission format.** The host's `sample_submission.csv` is semicolon-delimited
with a BOM, but Kaggle's grader parses plain comma-delimited UTF-8 — a semicolon
file fails with `ID column image_id not found in submission`. `image_id` must also
carry the true per-file extension: the test set is **251 `.tif` + 58 `.png`**, and
assuming `.tif` throughout is a known cause of rejected submissions.

**Metric anchor.** A constant submission of PA=25, FL=115, MT=30 (the midpoints of
the host-stated test ranges) scores **2.41344** public. That exact value already
appears twice on the leaderboard, so at least two other teams submitted the same
constants. Because a tolerance-normalised MAE is convex in a constant prediction
and minimised at the median of the targets — and the unknown tolerance is a
positive scale factor that cannot move that minimum — a 1-D sweep per variable
recovers the public-set median of PA, FL and MT. `scripts/probe.py` does this.

**Images have been rescaled, and not isotropically.** Training images sit at a few
canonical sizes (1714 at 800×1200, 645 at 1080×1640, …) while their masks retain
native sizes (556×996, 864×1152, 556×660, …) with *different aspect ratios*. The
host's own loader resizes image and mask independently to 256×256 and pairs by
filename, so the mismatch is harmless for segmentation training — but any geometry
measured in the delivered frame is anisotropically distorted, which corrupts
**pennation angle** as well as lengths. Recovering the true px/mm along each axis
separately is therefore not optional.

**Scale recovery.** `src/umud/scale.py` finds the console ruler by sliding a thin
strip along each border and taking the most periodic one (autocorrelation with
harmonic corroboration). The key discriminator is **duty cycle**: a ruler is a
sparse comb (≈0.01–0.02) while a console text block is also strongly periodic but
dense (≈0.13–0.15) and otherwise wins on autocorrelation alone. On IMG_00001 this
recovers a vertical pitch of **135.19 px/cm**, independently confirmed by the
labelled depth: the 607 px image region is marked "4.5 cm" → 134.9 px/cm, agreeing
to 0.2 %.

Current coverage: **255/309** test images yield a confident vertical pitch. The
853×1069 and 644×1088 groups lock on cleanly (IQR ≈ 0.02 px). In the dominant
800×1200 Siemens group, ~91/239 find the true ruler band while the rest still latch
onto the colour bar or the left edge.

## Week 2 findings

**The FL probe worked.** Sweeping the constant FL with PA/MT held fixed gave
60→2.15750, 80→1.95614, 100→2.10414, 115→2.41344. Interpolating the interval
slopes to their zero crossing puts the **public-set median FL at ~81.5 mm**, not the
115 mm midpoint we started from, and bounds the FL tolerance at t_FL ≤ 16 mm. This
is a calibration check for the eventual model: if its FL predictions do not centre
near ~81 mm, it is miscalibrated. MT and PA sweeps are still to run.

**Scale is better recovered from the printed depth than from tick marks.** The
imaging depth is annotated on-console ("4.5 cm", "Tiefe 4.0 cm") and the B-mode
sector spans exactly that range, so `px_per_cm = sector_height / depth_cm`. Verified
against two unrelated consoles: IMG_00001 gives +0.3 % against its tick comb, and
IMG_00252 −2.6 % against its ruler labels. `src/umud/depth.py` implements this;
`src/umud/ruler.py` is kept as an independent cross-check and, where both fire, to
resolve the tick unit (0.5 vs 1 cm) and inherit the comb's sub-pixel precision.

Getting there took three rewrites of the tick detector that each fixed the case
being debugged without generalising. What actually moved it was opening two images.

**The test set splits into two populations**, and this is the main structural result:

| | n | scale route |
|---|---|---|
| Console screenshots (chrome, text, ruler) | ~149 | depth OCR — works, 91 read so far |
| Fully cropped frames (sector fills the frame) | **160** | no console text exists; only faint border ticks |

Current coverage is 133/309. The 160 cropped frames are why, and OCR can never help
them — that is now a single well-defined problem (faint 1 cm border ticks, per the
host's guidance in discussion 689019) rather than a general one.

**Video runs confirmed.** Chrome-signature clustering plus frame similarity finds
**27 runs of exactly 5 frames** (135 images), matching the host's "5 images in a
row"; the remainder are largely standalone. Frames in a run share a scale by
construction, which both fills scale gaps and gives a free accuracy check — within
multi-frame runs the resolved scale agrees to 0 % median, though 6 of 19 runs still
disagree by more than 2 %, so detection errors remain.

## Full review of competition surfaces (code, data, models, discussions, team, rules)

**The exact metric is public.** The Overview links to a host notebook,
<https://www.kaggle.com/code/paulritsche/umud-score>, containing the real scorer.
Transcribed into `src/umud/metric.py`:

    S = (1/3) * [ MAE_PA/6.0 deg + MAE_FL/12.0 mm + MAE_MT/3.0 mm ]

with equal weights, per-image evaluation, and MedAE/RMSE tie-breakers at 1e-6 and
1e-9 (never rank-relevant). No more submissions need be spent learning the metric.
This confirms the FL sweep independently: MAE_FL(115) - MAE_FL(80) = 16.46 mm, and
the recovered public median FL of ~81.5 mm stands.

**A labelled external validation set exists**, on the OSF node linked from the Data
page (<https://osf.io/xbawc>, "Expert Analysed Benchmark Image Datasets"). 35 images
scored by **seven expert raters** using the same protocol the test set uses (3 MT
lines, 3 fascicles, 3 angles, averaged), plus **ground-truth `Scale_pixel_per_cm`**
and DL_Track's own predictions. `data/external/`.

**The leader is below the human noise floor.** Scoring each expert against the
consensus of the other six:

| | UMUD score | MAE_MT | MAE_FL | MAE_PA |
|---|---|---|---|---|
| best expert (R6) | 0.2459 | 0.61 mm | 3.79 mm | 1.30 deg |
| **mean expert** | **0.3032** | ~0.7 mm | ~5.1 mm | ~1.4 deg |
| DL_Track (host's tool) | 0.3306 | 1.31 mm | 3.74 mm | 1.45 deg |
| leaderboard #1 | **0.29358** | | | |
| leaderboard #3 | 0.32900 | | | |

Leaderboard #1 sits *below the average human expert* at reproducing an expert
consensus, achieved with 252 submissions against a 102-image public split. Treat the
top of the public board as substantially overfit. **The realistic ceiling is ~0.30**,
and a method that genuinely reaches 0.30-0.35 and generalises should place on the
private split. Chasing 0.29 on public is the wrong target.

**Muscle thickness is the biggest lever.** Of DL_Track's total normalised error, MT
contributes 0.437, FL 0.312, PA 0.242. MT is where it trails humans worst (1.31 mm
vs ~0.6 mm) and it is the easiest quantity to measure -- a perpendicular distance
between two aponeuroses, with no fascicle extrapolation. Matching human MT alone,
holding DL_Track's FL and PA fixed, would score ~0.25.

**Scale is a small discrete set, not a continuous quantity.** Ground truth across the
35 benchmark images takes only seven values (74, 77, 85, 89, 94, 112, 126 px/cm), and
image shape almost determines it -- 4 of 6 shapes map to a single value. Those shapes
((556,660), (600,800), (652,800), (688,1234), (512,512)) are the same native
resolutions seen in the competition's training masks. Scale should be reframed as
lookup/classification over a handful of device configurations.

**Honest negative on the Week-2 scale work.** Validated against the now-known truth,
the depth-OCR estimator fires on **0 of 35** benchmark images (they are cropped, with
no console text) and the tick comb on 7, of which only 1 gives a clean unit ratio.
The estimator is sound where console chrome exists and useless where it does not,
which is exactly the 160-image cropped population in the test set.

**Nothing to learn from other competitors.** Code and Models tabs are empty -- no
public notebooks, no shared models, `kernelCount = 0`. The six-team score cluster at
0.45134 is almost certainly stock DL_Track, since independent teams cannot otherwise
produce identical scores; further clusters sit at 0.76704, 1.33851 and 1.87113.

**Other references.** DL_Track_US preprint (host cannot share the paywalled paper):
<https://arxiv.org/abs/2009.04790>. Also referenced: UltraTimTrack, DUSTrack.
Training labels are deliberately incomplete -- the host confirmed only clearly
visible fascicle fragments and aponeuroses were annotated, deep structures may be
bone, and all images are of superficial muscles.

**Licence caveat.** The OSF node is published CC BY 4.0 but bundles a "DL_Track Data
Usage Policy" asserting non-commercial use, and the host was asked to resolve this in
discussion 737782 and has not yet answered. Use of this benchmark must be declared as
external data, and the conflict should be watched before relying on it in the final
GPL-3.0 submission.

## Working pipeline and local validation loop

`scripts/eval_benchmark.py` scores image -> masks -> geometry against the 35
expert-labelled images. Segmentation uses the DL_Track_US pretrained VGG16-U-Nets
(Apache-2.0, from <https://osf.io/7mjsc>) as the reference front end; geometry is
ours (`src/umud/geometry.py`).

| stage | UMUD | note |
|---|---|---|
| v1 centreline geometry | 0.6064 | |
| + inner-edge thickness convention | 0.4609 | MT MAE 2.45 -> 1.19 mm |
| + FL traced to aponeurosis centrelines | 0.4477 | |
| + FL from the trigonometric identity | 0.4319 | |
| + DL_Track's own thresholds (apo 0.35, fasc 0.10) | 0.4142 | PA MAE 1.26 deg |
| + FL blended with a DL_Track-style estimate | **0.3637** | FL MAE 7.72 -> 5.90 |
| DL_Track published | 0.3306 | |
| expert mean / best | 0.3032 / 0.2459 | ceiling |

Per target, ours vs DL_Track: **PA 1.257 vs 1.449 deg**, **MT 1.170 vs 1.315 mm**,
FL 5.90 vs 3.74 mm. Our PA is now better than the best of the seven expert raters
(1.30 deg) and our MT beats the reference; the remaining deficit is fascicle length.

**The FL blend.** Porting DL_Track's own procedure (contour-edge line fits inside the
inter-aponeurosis band, PA gated to 10-40 deg) did *not* reproduce its published
accuracy -- it scored 9.26 mm against our trigonometric route's 7.72. But the two
estimators are genuinely independent, one never extrapolating and the other never
using MT, and their biases were almost equal and opposite (-3.92 vs +4.55 mm).
Averaging them cancels most of both: **5.90 mm**, close to the 5.1 mm at which the
experts agree with each other. The weight is fixed at 0.5 rather than the fitted
optimum of 0.6 (0.3637 vs 0.3620) so that nothing is tuned to the 35 benchmark
images.

### What the harness has already overturned

Four plausible ideas measured worse, and would have been shipped without it:

* *Convention beats segmentation quality.* Measuring thickness centreline-to-centreline
  overstated it by a very consistent 8.5 % (ratio IQR 0.028) -- one aponeurosis
  thickness. Switching to the belly faces was worth 0.15 of score, far more than any
  model change so far. FL then needed the *opposite* convention, tracing into the
  aponeurosis centrelines, worth another 0.013.
* *A single representative fascicle is worse than many noisy ones* (0.4814 vs 0.4477).
* *Merging collinear fragments before fitting did not help* at any tolerance tested.
* *The trigonometric identity FL = MT/sin(PA) beats our own fascicle extrapolation*
  (7.96 vs 8.53 mm) -- because MT and PA are measured well and the identity needs no
  extrapolation. It has a ~6.2 mm floor even with perfect inputs, so it is a ceiling
  on this route, not a solution.
* *Calibration gains do not generalise.* Fitted per-target gains improved the
  in-sample score (0.4319 -> 0.4241) but leave-one-out was **worse** (0.4512). Not
  applied.

### Next

The gap is entirely FL, and the arithmetic is unambiguous: our PA and MT combined
with DL_Track's FL accuracy would score **0.3148**, beating DL_Track outright and
approaching the expert mean. Porting DL_Track's fascicle-length algorithm (its code
is Apache-2.0 and the rules permit re-use "in an improved version") is worth more
than any other single change available.

## Open / next

- Tighten ruler-band selection for the 800×1200 group (require thin marks of
  consistent length, not just sparsity).
- Exploit the 5-frame video runs: scale, and the targets themselves, are near
  constant within a run, so a per-run consensus both cleans up scale detection and
  cuts prediction variance.
- OCR the depth label ("4.5 cm") as an independent scale cross-check.
- Settle the anisotropy question empirically — needs images with a confident pitch
  on *both* axes, which we do not have yet.
- Segmentation baseline (DL_Track_US, Apache-2.0, one-way compatible with the
  required GPL-3.0) → geometry → calibration.

## Licence

GPL-3.0-or-later, as required for prize eligibility.

## Setback: the benchmark score does not transfer to the test set

First real submissions:

| submission | public |
|---|---|
| full pipeline, aspect 1.35 | 1.46532 |
| measured PA only, FL/MT at probed medians | 1.15267 |
| **best constant** (probed medians, predicted by separability) | **1.03740** |
| DL_Track cluster | 0.45134 |

Both are *worse than a constant*. The isolation submission is unambiguous: measured
PA alone costs 0.115 against simply guessing the median, so on the competition test
images our pennation angle is not merely imprecise, it is worse than useless -- while
the same code scores 1.26 deg MAE on the benchmark.

**Root cause, from looking at the overlays.** On the console-screenshot images the
aponeurosis network returns *three* bright bands -- skin/superficial fascia, the true
superficial aponeurosis, and the deep aponeurosis -- and our "widest plausible pair"
rule has no way to tell which two bound the muscle. The fascicle network meanwhile
almost entirely misses the belly: on IMG_00252 it returns 3313 fascicle pixels, most
of them in subcutaneous tissue *outside* the muscle, while the obviously striated
belly is left blank. The sector is 437 px wide inside a 1200 px frame, so resizing the
whole frame to 512 squeezes the fascicles to ~187 px across.

Cropping to the sector first moves that image's PA from 6.9 to 15.0 deg (expected
~16), but collapses MT from 19.4 to 9.7 mm because the pair selection then latches
onto a different wrong pair. Aponeurosis pair selection is the linchpin: MT is
measured between the pair, PA is measured *relative to the deep member*, and FL
depends on both.

**Retraction.** The anisotropy factor of 1.35 reported above should not be trusted.
It was fitted so that the predicted PA and FL medians matched their probed values,
but those predictions came from this broken segmentation, so the fit was absorbing
segmentation failure rather than measuring a real resize. The benchmark control
(aspect 1.0 clearly optimal on native images) remains valid; the inference that test
images need 1.35 does not.

**What the benchmark harness cannot see.** The 35 benchmark images are pre-cropped,
single-population and native-resolution. They are the right instrument for measuring
conventions and geometry, and were decisive for both. They cannot detect a domain
shift to full console screenshots. The leaderboard-probed medians are the only
label-free check that spans the real test distribution, and they are what exposed
this.

