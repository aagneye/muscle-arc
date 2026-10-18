# Competition intelligence: how the leaders win, what the host allows, what we tried

Compiled 2026-10-18 from the public leaderboard (434 teams), all 15 discussion threads, the
rules page, the competition API, and our own 26 submissions. Companion to
`docs/literature_review.md` (methods) and `docs/curator_climb.md` (experiment log).

## 1. Where we stand

| Date | Best public | Rank | Note |
|---|---|---|---|
| 2026-09-08 | 0.85193 | 280 / 433 | SOF model, heuristic scale |
| 2026-10-18 02:57 | 0.44273 | 120 | dual_v3 U-Nets, exact scale, fixed clips, PA +1.9° |
| 2026-10-18 17:12 | **0.41936** | **107** | 5-fold ensemble of the same recipe |

Rank 10 is 0.23813, rank 20 is 0.25472, rank 50 is 0.35134. Reaching the top 10 means
halving our error again (0.42 to 0.24).

## 2. What the leaderboard tells us about the leaders

| # | Team | Score | Subs | Public notebooks |
|---|---|---|---|---|
| 1 | Kragglenote2 Forwork (Nabidnur abrar) | 0.200 (stated), 0.206 on board | 80+ | none |
| 2 | HongYongGi | 0.212 | 88 | none |
| 3 | Kainwang Roger | 0.212 | 122 | none |
| 4 | Talha Bacak | 0.215 | 109 | none |
| 5 | Bruhabcdfg | 0.219 | 172 | none |
| 8 | LuPW | 0.225 | 357 | none |
| 12 | Victor | 0.239 | 79 | asked "Clarity on Measurements" |
| 23 | PatrickAIForFun | 0.276 | 38 | manual-labelling threads |

Facts, not guesses:

- None of the top 15 users has a single public notebook. The strongest public notebook is
  0.399; the 25-team cluster at 0.45134 is a hard-coded CSV. Nobody at the top shares code.
- Every top-10 team submits daily with 80 to 357 entries. The top 7 all submitted today.
  The scores are below the expert-vs-expert floor (about 0.30 on the host's own six-rater
  benchmark; DL_Track, the host's tool, scores 0.33 there). An automated method validated
  only on training data cannot be expected to beat the raters' own agreement.
- Scores keep falling in small steps (0.287 on 2026-09-05, 0.217 on 10-12, 0.206 on 10-18,
  "0.200, will decrease more soon" per the leader). That is the signature of iterative
  calibration against the public 33 % (102 images), i.e. hyperparameter tuning read off the
  leaderboard, which the host explicitly allows as long as the final code runs without a
  leaderboard.
- The leader (post 749906, 2 days ago) is recruiting: "anyone who has resource of max
  subscription and like gpu resource can join me, sharing the top prize". The merger
  deadline equals the final deadline (2026-11-14 23:00 UTC, from the API), team size 10,
  and our 26 submissions are far below the merge cap. This is the "person suggesting how to
  win" in the discussions: an offer, not a method.
- PatrickAIForFun (rank 23, only 38 submissions) got 0.4587 from 1.5 h of rough manual
  fitting of the test images on 2026-04 and asked the host to publish the exact labelling
  protocol. Talha Bacak (rank 4) asked whether the sample-submission rows were measured
  like the test labels, which is the question you ask when you are fitting to labels.

Inference (clearly marked as such): the sub-0.25 entries combine (a) a segmentation pipeline
like ours, (b) per-parameter calibration read off the public board over many submissions,
and most likely (c) manual host-protocol measurements of the test images used as a
calibration set, which the host ruled permitted-if-declared (threads 690868, 737782).

## 3. Discussion digest (all 15 threads)

| Thread | What matters for us |
|---|---|
| 689019 Tick mark question | Bottom ticks 1 cm apart; use the vertical ruler; square pixels; 512x512 train images have no scale. |
| 748126 Stretched 1200x800, PA, split | Host used only vertical bars; PA measured between lower apo and extrapolated fascicle, apo segment length "not standardised"; **public/private split is random by image**. |
| 744818 Clarity on measurements | Raters choose fascicles needing the least extrapolation (middle of image); TA central apo: use the upper compartment. |
| 743111 Sample submission values | Not measured; made up. Do not pin them. |
| 690868 Test data leakage / manual labeling | Host: manual labels on test = external data, permitted-if-declared; "labels != architecture estimates, the computational approach matters a lot"; rater variability exists. |
| 737782 Clarifications | Host confirms permitted-if-declared (will re-run top-3 code); 2 final submissions; 10 per team; prizes CHF 1500/1000/500; benchmark data CC BY-NC-SA. |
| 739660 Reproducibility / LB ensembles | Leaderboard-weighted ensembling not eligible; manually set weights are; test: "can it re-run on a new dataset with no leaderboard?" |
| 740356 670 duplicates | Confirmed; remove one copy per group (we do). Test set has no overlap with train. |
| 702487 Image/mask shape mismatch | Host loader resizes both to 256x256 independently; masks align to the full frame. |
| 687198 Unannotated area | Only clearly visible fragments/apos labelled; deep bright structures may be bone; all images are superficial muscles. |
| 737094 Paper access | Preprint arXiv:2009.04790 is the public stand-in. |
| 682455 General inquiries | Ends 2026-11-15, private board 2026-11-22. |
| 749906 "anyone who have resource can join me" | Leader at 0.200 recruiting GPU/compute partners for a prize share. |
| 748541 Image channel confusion; 688361 Broken link | Nothing actionable. |

Rules page: 5 submissions/day, 2 finals, team size 10, mergers allowed until the merger
deadline, winners must deliver training + inference code under GPL-3.0 with a reproducible
description. Kaggle Foundational Rule 4.b ("no hand labeling of test data") conflicts with the
host's ruling; the host said the foundational rules were not updated yet and that the top-3
code will be re-run, so the practical rule is: declare it, and make sure the delivered code
regenerates the submission from the training data plus the declared external file.

## 4. What we tried (26 submissions) and what each taught

| Phase | Public | Lesson |
|---|---|---|
| Sep 4-5: U-Net + geometry, heuristic scale | 3.51 to 1.00 | Scale (mm/px) dominated the error, not segmentation. |
| Sep 5: OSF residual blend (v9) | 1.06 | Local 2-image checks are worthless; needs a real gate. |
| Sep 6: constant-value probes | 1.06 to 1.44 | Public median FL about 81.5 mm; MT about 22 mm. |
| Sep 8: SOF multitask + hybrid blends | 0.852 | SOF fascicle head was almost empty (later found). |
| Sep 8-9: stock DL_Track variants | 0.96 to 1.54 | Cropping/letterboxing console frames breaks DL_Track; public anchor at 0.451 was still better than all of these. |
| Oct 12-18: exact per-vendor scale table (309/309), fixed 5-frame grouping, dual U-Nets on dedup splits | 0.460 | Two bugs (missing scales for 102 images, smoothing across 164 unrelated images) were worth 0.4. |
| Oct 18: PA +1.9° | 0.443 | Test raters read pennation about 2° steeper than our geometry. |
| Oct 18: OSF-selected conventions (least-extrapolation FL, no-Radon PA) | 0.473, 0.447 | The 35-image six-rater benchmark does NOT predict the two test raters. Stop using it as a selector. |
| Oct 18: 5-fold ensemble | **0.419** | Mask quality still pays; about 0.025 per ensemble step. |

Measured against the 0.451 public anchor, our best file now differs by MT 1.0 mm, FL 7.8 mm,
PA 2.9°. In tolerance units FL (0.65) and PA (0.48) carry the gap, MT (0.33) is nearly done.

## 5. The three ways to the top 10

| Path | Expected public | Effort | Risk |
|---|---|---|---|
| A. Automated only: more ensemble members, SegFormer encoder, apo-edge heads, TTA, muscle-wise priors | 0.36 to 0.40 (rank 40 to 70) | 1 to 2 weeks GPU | Low. Cannot reach 0.24 by 11-14; the error floor of unlabelled calibration is the rater gap. |
| B. Manual host-protocol measurements of the 309 test images (declared external data) + calibration of our automated pipeline to them, LB used only to confirm | 0.24 to 0.30 (rank 10 to 25) | 1 to 2 days of your labelling with a tool I build; then 2 days fitting | Medium. Must be declared; delivered code must re-run from train + the declared file; Kaggle rule 4.b is the formal risk the host says he will handle. |
| C. Merge with the leader (0.200) who is recruiting GPU partners | top 1 to 3 immediately | one message | High. Their pipeline is unknown; if it is not reproducible under GPL-3.0 the team is ineligible for prizes; prize share negotiated; our own work becomes a minor contribution. |

Recommendation: do B as the main line and A in parallel on the GPUs (they compound: better
masks make the calibration cleaner). C is your call; it is the only way to a guaranteed
top-3 on the public board, and the only one that costs us control.

## 6. Why B works, in numbers

The metric divides errors by 6°, 12 mm, 3 mm. Our pipeline already produces masks good
enough for MT within 1 mm. What it cannot learn from training data is the raters' habits:
how steep they read pennation (about +2°), how long they extrapolate fascicles (longer than
the OSF raters), where they place the three thickness lines. With labels on the test images
those habits become a per-parameter, per-device map that is fitted once and then runs on any
new image; the host's own answer in 690868 anticipates exactly this ("in practical settings
we quite often do this ... label 100 images and re-train").
