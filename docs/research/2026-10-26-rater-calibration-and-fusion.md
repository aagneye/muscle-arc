# Research digest 2026-10-26: automated measurement vs human raters

Subagent literature report (read on 2026-10-26) commissioned for the UMUD score-loss analysis
(`docs/score_loss_analysis.md`): what top entries of comparable challenges did beyond
segmentation, calibrating to raters with few labels, fascicle-length error, fusion level, and
thin-structure architectures. Flags inside mark unverified numbers.

---


Scope note: `docs/literature_review.md` already covers the muscle-specific literature (DL_Track lineage, TimTrack/UltraTimTrack, Ando/Franchi/Noorkoiv extrapolation numbers, Muramatsu curvature, rater reliability). This report adds cross-domain challenge evidence and only repeats in-domain numbers where they anchor a conclusion. Everything below was read from the cited pages on 2026-10-26; where a number comes from a leaderboard comment, a slide deck, or a search snippet rather than a paper, it is flagged. Nothing is invented; where a source gave no number, that is stated.

---

## 1. What top entries did beyond segmentation in comparable "automated measurement vs human rater" challenges

### 1.1 HC18 fetal head circumference (Grand Challenge)
- Setup (van den Heuvel et al. 2018, PLOS ONE): 999 train / 335 test images; ground truth = one sonographer's hand-drawn ellipse; metric = mean absolute difference (MAD) of HC in mm. The second observer (a trained researcher) measured HC "a few millimetres larger" than the sonographer, i.e. the label itself carries a rater-convention offset. The organisers' classical system scored 2.8 ± 3.3 mm.
- Live leaderboard (read in the browser, 2,252 entries, self-described methods, not peer-reviewed):
  - 1st: "nnU-Net, 1 fold, 1000 epochs" 0.74 ± 0.73 mm (June 2023). 3rd: same user, "nnU-Net 5 fold" 0.81 ± 1.26 mm. FLAG: unverified; the 0.74 is below the 1.37 mm "ground-truth ellipse-fitting floor" that SonoBase reports for HC18 (see below), so it must involve a different HC-from-mask convention, not just better masks. Note the 5-fold ensemble was worse on the mean and much worse on SD (outliers) than one fold.
  - 4th: 25-model segmentation ensemble (EfficientNet-B4/B5 encoders x UNet++/DeepLabV3+/FPN/MANet, mean-mask averaging, OpenCV ellipse fit) 1.69 ± 1.60 mm (June 2026): no better than single Attention U-Nets from 2019 (1.72 ± 1.55) and UNet++ variants (1.72 ± 1.61).
- Published numbers on the HC18 test set: Huawei Cloud 2018 winner 1.89 mm (press release); DAG V-Net 1.77 ± 1.69 mm (ranked 5th at the time); Fiorentino et al. 2021 regression framework (region-proposal CNN + distance-field regression of the contour, no whole-head segmentation) 1.90 ± 1.76 mm, Dice 97.75 % — a regression head did not beat segmentation+ellipse; Budd et al. 2019 deterministic U-Net 1.81 ± 1.65 mm, and averaging 10 MC-dropout samples made it worse (2.15 ± 2.09), Probabilistic U-Net worse still (2.22 ± 2.15); uncertainty-based rejection only helped when rejecting the very worst cases (no numbers given; authors "cannot claim" their variance is true uncertainty).
- Convention floor (SonoBase, arXiv 2609.19230): HC error 1.81 mm (box prompt) against a 1.37 mm floor obtained by fitting an ellipse to the ground-truth mask itself. About 75 % of the residual error of a good segmenter on HC18 is the ellipse/perimeter convention, not pixels.
- Stacking ensemble (Sarcheshmeh et al. 2026, Discover AI, open access): YOLOv12-M + SegFormer-B3, fused by a pixel-wise 2-8-8-1 MLP meta-learner on the two probability maps, then closing, largest component, Gaussian-smoothed contour, Fitzgibbon ellipse, Ramanujan-II perimeter: 0.91 ± 1.32 mm on their own OpenFetus set (not HC18 test). They claim the meta-learner beats "simple average ensembles" but the ablation numbers are in table images (not readable). Useful side results: SegFormer B3 to B5 gives only marginal gains; input size 64 to 512 px gives large gains.
- Lesson: in HC18 the measurable levers were the measurement convention (ellipse fit + perimeter formula) and resolution, not backbone or ensemble size. Mask-averaging ensembles of 25 models landed where single 2019 models were.

### 1.2 RSNA Pediatric Bone Age 2017 (pure regression to radiologist readings)
- Halabi et al. 2019 (Radiology): 14,236 radiographs (12,611/1,425/200); ground truth of the 200-image test set = weighted average of six reviewers (weights 0.14 to 0.21; Pan 2019); the mean reviewer MAD against that consensus was 5.8 months. Top-5 MAD: 4.2, 4.4, 4.4, 4.5, 4.5 months; the top five were all ensembles (Pan 2019).
- Winner 16 Bit (Cicero & Bilbily): modified Inception-V3 taking biological sex as input, MAD 4.265 months; press release: individual radiologists differed by 7.32 months from a 3-radiologist gold standard. FLAG: the oft-cited "500x500 input, 5-network ensemble, TTA-10" comes from Kalinin's RSNA 2017 slides (PDF not text-extractable here) and is unverified.
- Pan et al. 2019 (Radiology: AI, PMC6884060), post-hoc ensembling of the 48 submissions: single model 4.55 months; best 4-model ensemble 3.79 months (−17 %); top-N selection 3.93 (8 models) and 4.01 (4 models). The best ensembles combined less-correlated models (mean inter-model correlation 0.47 vs 0.68 for top-N); most of the gain came from ~3 models; MAD95 (95th percentile) 12.54 to 12.37 months.
- Iglovikov/Rakhlin et al. 2018 (arXiv 1712.05053): U-Net hand segmentation + 3-keypoint affine registration to a standard pose, region-specific networks ensembled: regression mixed-sex 8.08 (whole hand) to 7.52 months (region ensemble); male 6.67 to 6.30; female 6.79 to 6.49; classification male 6.44 to 6.16; regression+classification ensemble 6.10 (best); sex-specific models beat mixed; classification heads slightly beat regression; test MAD 4.97 months, better than validation, which the authors attribute to the test labels being a 3-expert average vs one rater for training.
- Lessons: scalar-level ensembling of decorrelated models buys 4 to 17 %; stratified (sex-specific) models beat pooled ones; cleaner multi-rater labels lower the achievable error.

### 1.3 Kaggle Second Data Science Bowl 2016 (LV volumes from cine MRI, ground truth = clinical volumes, CRPS metric)
- 1st (Tencia Woshialex / Qi Liu, GitHub README): two segmentation CNNs, contour areas averaged across the two, volumes computed; combined with a sex-age model, a largest-slice model and a 4-channel model via an "average model"; "fit simple models based on the train dataset to correct systematic errors" (form not stated); score 0.009485 test.
- 2nd (Ghent, 317070.github.io/heart): direct CDF regression (600-way softmax or Gaussian mu/sigma) trained on CRPS, with a differentiable frustum (slice areas to volume) layer; segmentation-mask models "never achieved comparable results" (no mask labels existed); ~250 models; ensemble weights optimised on validation, each model scaled by its KL divergence from the consensus and cross-entropy; models disagreeing too much with the consensus removed; per-patient fallback to an outlier-robust ensemble when the two ensembles disagreed by > 0.02 CRPS; test-time augmentation by averaging over quasi-randomly augmented copies "helped a lot" (no number); selective prediction improved the public LB from 0.0148 to 0.0132; best patient model 0.0105 (val), final 0.0110 (public LB).
- 3rd (Julian de Wit blog): U-Net on self-labelled slices, then a calibration step: gradient-boosting on the residual (true − estimated volume) using segmenter-uncertainty, largest slice, dropped-slice count and DICOM age/sex/angle/slice thickness; residual target beat ratio and direct-volume targets; trained in folds to avoid leakage; gain ≈ 1 ml MAE and ≈ 0.0005 CRPS on the LB ("close to the gap between #3 and #4"); CDF sigma set from the error SD of 30 to 60 similar-volume cases.
- Lesson: post-hoc residual calibration against the clinical ground truth was small but rank-relevant; consensus-based outlier handling at the prediction level gave ~10 % CRPS.

### 1.4 PSFHS / FH-PS-AOP, MICCAI 2023 (segmentation scored by a derived clinical angle)
- 5,101 intrapartum ultrasound images, 179 entries, top 8 advanced (challenge report arXiv 2409.10980; per-team table from MiTU-Net, arXiv 2401.15513): 1st Stock/Köhler/Maier-Hein (nnU-Net v2) delta-AoP 6.54°, Dice-all 0.9236; 2nd Elbatel (SAM ViT-H) 7.97°, Dice 0.9346; 3rd Qiu (UperNet) 7.65°, Dice 0.9295; 4th 8.56°; 5th MiTU-Net 8.72°; 6th 9.28°; 7th 12.20°.
- The winner had the lowest Dice of the top three but the best angle error: pixel-overlap ranking and derived-measurement ranking disagree. Ensemble/TTA details of the top teams were not retrievable (code folders show nnUNetv2, SAM_VIT_H, UperNet, DSEU-net, BRAU-Net, UNet). Inter-observer AoP not reported in what was read.

### 1.5 Cardiac measurements, carotid IMT, fetal biometry
- EchoNet-Dynamic (Ouyang et al. 2020, Nature, PMC8979576): direct video regression (R(2+1)D) of EF; beat-to-beat averaging MAE 4.05 % vs 4.22 % for a single 32-frame clip; literature inter-observer 7.6 to 13.9 %; prospective same-day repeat: median difference 2.6 % (SD 6.4) for the model vs 5.2 % (SD 6.9) for sonographers' Simpson's biplane; segmentation Dice 0.90 to 0.93 used only to locate beats. "The segmentation ceiling" (arXiv 2609.19730): adding predicted masks, ED/ES sampling or area-consistency objectives did not beat raw-video regression (weight averaging + strong augmentation R² 0.806 / MAE 4.08 vs R(2+1)D R² 0.811); their closed-form break-even requires per-frame area error < ~10 %, typical segmenters sit at ~14 %.
- CAMUS: encoder-decoder EF MAE 5.6 %, below inter-observer; SonoBase's CAMUS nnU-Net 6.28 % EF error vs 5.9 to 6.9 % inter-observer.
- EchoNet-LVH (Duffy et al. 2022, JAMA Cardiology, from the Stanford-hosted copy): linear-caliper measurements vs sonographers: ~1.2 to 1.4 mm wall thickness, 2.4 mm LV internal diameter internally; externally 1.7 / 3.8 / 1.8 mm (domestic) and 1.7 / 2.9 / 2.3 mm (international). FLAG: sources disagree on which wall is 1.2 vs 1.4 mm.
- Carotid IMT (pre-deep-learning automated systems): bias 0.001 ± 0.035 mm vs mean of two manual readings; semi-automated vs manual bias 0.023 ± 0.052 mm with manual reading higher; automated inter-observer spread 0.01 mm vs manual 0.044 mm.
- Fetal biometry: AutoFB (Bano et al., MICCAI 2021): HC 2.67 mm, AC 3.77 mm, FL 2.10 mm (ellipse for HC/AC, rectangle diagonal for FL). A 2025 modular framework (PMC12859377) that used a convex-hull perimeter instead of an ellipse got HC MAE 9.74 mm with Bland-Altman bias +8.2 mm (LoA −5.0 to +21.6): a textbook convention mismatch.

### 1.6 Synthesis of "tricks" with measured effect
| Trick | Evidence | Size |
|---|---|---|
| Scalar-level ensembling of decorrelated models | Pan 2019; Iglovikov 2018 | −17 % MAD (4 models); −4 to −7 % per region ensemble |
| Consensus / outlier-robust fallback at prediction level | DSB-2016 2nd | 0.0148 to 0.0132 CRPS (−11 %) |
| Post-hoc residual calibration to ground truth | DSB-2016 3rd; SonoBase; LLM-judge (sec. 2) | ~1 ml / 0.0005 CRPS; EF 6.63 to 4.95 %; MAE 0.738 to 0.356 |
| Matching the rater's measurement convention | SonoBase floor; modular fetal HC; PSFHS Dice vs AoP | 1.37 mm of 1.81 mm; +8.2 mm bias; rank inversion |
| Averaging over frames/beats | EchoNet | 4.22 to 4.05 % MAE; repeatability 2.6 vs 5.2 % |
| Stratified (sex/muscle-specific) models | Iglovikov | 8.08 to 6.7 months |
| Regression head instead of geometry | EchoNet (yes, with 10k videos); Fiorentino HC18 (no, 1.90 vs 1.77 mm); segmentation-ceiling | mixed; wins only with large label sets |
| MC-dropout / probabilistic-U-Net averaging | Budd 2019 | hurt (1.81 to 2.15 mm) |
| TTA | DSB 2nd ("helped a lot"); breast-US U-Net Dice 0.708 to 0.754 (snippet) | anecdotal for measurements |
| Mask-level ensembling for a derived scalar | HC18 leaderboard | 25 models = single model; 5-fold worse than 1-fold |

---

## 2. Calibrating an automated measurement to raters with few or zero labels

- Linear (intercept + slope) recalibration with tens of labels is the best-evidenced option:
  - SonoBase (arXiv 2609.19230): CAMUS EF, linear fit on one random half of 100 patients (~50 labels), evaluated on the other half over 2,000 splits: MAE 6.63 % to 4.95 %, kappa at 35 % 0.457 to 0.546. ACOUSLIC abdominal circumference, leave-one-out linear recalibration: raw MAE 18.92 / 14.77 / 18.11 mm (SonoBase / MedSAM2 / SAM2) collapsed to 13.9 / 13.9 / 13.8 mm (p = 0.998) — once gain/offset are removed, three different segmenters are indistinguishable, i.e. most of the between-model difference was systematic bias.
  - Hierarchical-Bayesian linear calibration vs neural-ODE score transport (arXiv 2605.09227, LLM-judge scores, continuous targets): uncorrected MAE 0.738; with 100 anchor labels, Bayesian linear (per-cell intercept/slope with partial pooling = shrinkage across cells) 0.356 MAE / KL 0.031 vs flow 0.340 / 0.058; with 1,500 labels linear saturates at 0.359 while flow reaches 0.320. Conclusion: at 50 to 100 labels the linear corrector with partial pooling is preferred; nonlinear transport only pays off at ~1,000+ labels. They also note a marginal optimal-transport map matches the distribution exactly but ignores per-item pairing.
  - Few-shot decoder fine-tuning (SonoBase, ACOUSLIC): N = 5 labels gave 17.17 ± 5.52 mm vs 29.80 ± 16.83 for MedSAM2; N = 30 reached 13.7 mm — i.e. 5 labels already recover most of what 30 do, when the base model is good.
  - DSB-2016 3rd place: residual GBM with metadata features, trained in folds, ~1 ml gain (sec. 1.3).
- Shrinkage: Van Calster et al. (Stat Methods Med Res, shrinkage simulation) found shrinkage improves the calibration slope on average but raises between-sample variability and "often worked poorly in individual datasets, in particular when it was most needed"; bootstrap uniform shrinkage behaved best; penalised methods over-shrink and have unstable tuning at small n. Niculescu-Mizil & Caruana 2005: isotonic regression needs substantially more data than 2-parameter Platt scaling (exact threshold not verified here). Practical reading: with ~35 OSF images per parameter, fit intercept + slope only, pool slopes across muscles/devices, and shrink the slope toward 1.
- Quantile mapping / distribution matching: the only quantitative head-to-head found is the climate bias-correction study (PMC7922393): raw MBE −0.79, MAE 4.84; linear QM, empirical QM, robust QM, normal mapping and QDM all gave MAE 4.35 to 4.37 (linear 4.36). All variants removed the mean bias; none of the nonlinear mappings improved MAE beyond the linear one. No published evidence was found that quantile/marginal matching improves MAE against human raters; theory argues against it: the MAE-optimal point prediction is the conditional median, whose spread is smaller than the target marginal, so forcing the predicted marginal onto the label marginal inflates random error unless the miscalibration is monotone and systematic (in which case the affine fit already captures it). FLAG: this last sentence is reasoning, not a cited result.
- Protocol rules as constraints: the convention floor dominates in every challenge reviewed (HC18 ellipse floor 1.37 of 1.81 mm; +8.2 mm HC bias from convex-hull perimeter; Han 2013's 1.4 mm MT bias from apo centreline vs inner edge in the repo review; PSFHS winner has lower Dice but lower angle error). Reproducing the rater's geometric rule (where lines are placed, which edge, which fascicles, extrapolation rule) is worth more than any model change in these datasets.
- Agreement among estimators as a proxy (no labels): Pan 2019 (decorrelated ensemble 4.55 to 3.79 months); DSB-2016 2nd (consensus weighting by KL divergence, outlier-model removal, robust fallback: 0.0148 to 0.0132); TimTrack (van der Zee & Kuo 2022): using the accumulator-weighted median of the top-K Hough peaks instead of the single dominant angle cut fascicle-length RMSD by ~63 %; Budd 2019: averaging stochastic samples hurt, rejecting the most uncertain few helped.
- Self-consistency across video frames: EchoNet 4.22 to 4.05 % MAE and repeatability 2.6 vs 5.2 % (human); UltraTimTrack (PeerJ CS 2025): per-frame TimTrack cycle-to-cycle noise 3.5 ± 1.0 mm / 1.4 ± 0.5° fell to 1.4 ± 0.4 mm / 0.6 ± 0.3° after Kalman fusion with optical flow, while drift stayed at 2.1 mm vs 67 mm for pure tracking; RMSD vs manual 2.3 to 2.6 mm / 0.8 to 0.9° where DL_Track gave 3.8 to 8.0 mm / 3.9 to 4.8° on the same sequences. For 5-frame clips the cheap version is a median or flow-weighted mean of per-frame estimates.

---

## 3. Reducing fascicle-length error specifically (beyond what the repo review already has)

- TimTrack (van der Zee & Kuo 2022, PLOS ONE): fascicle length computed trigonometrically from perpendicular thickness and the pennation angle (deep/superficial apo angle minus the accumulator-weighted median fascicle angle), with no extrapolation step applied in the validation, reached RMSD 0.52 cm vs the mean of three manual observers — essentially the manual-manual RMSD of 0.56 cm (MT 0.11 vs 0.06 cm; PA 1.82 vs 1.83°). Regression slopes 1.01 / 0.97 / 1.01 and R² 0.93 to 0.97: errors were random, not systematic. Per dataset: VL isometric 0.63 ± 0.20 cm, GM 0.23 to 0.48 cm, GL 0.36 to 0.41 cm. Two implementation details reduced error: the weighted median over many Hough peaks (−63 % FL RMSD vs dominant angle) and correcting the Hough orientation bias by re-running on a rotated image. Caveat: these muscles/images had fascicles largely within the frame; Franchi 2020 (in the repo review) shows trig equations over-read BFlh by 0.5 to 1.9 cm when most of the fascicle is extrapolated. So trig FL is a rater-level estimator for GM/GL and a strong prior elsewhere, not a replacement.
- UltraTimTrack 2025 models fascicles and aponeuroses as straight lines and lists curvature as a limitation; still 2.3 to 2.6 mm vs manual. Sensitivity to image noise, not geometry, is what the authors blame for DL_Track's 8.0 mm on TA.
- Bizet et al. 2025 (J Biomech 180:112520, "Accounting for fascicle curvature affects muscle architecture characterization in dynamic conditions"): VL at 50 % and 83 % femur length during isokinetic contractions; straight-line tracking significantly over-estimated fascicle length, belly length and thickness and under-estimated pennation angle vs 2- and 4-segment linear extrapolation, worst distally; MLE4 intra-rater ICC 0.88 to 0.93; inter-rater "varied". The abstract gives no mm values (full text paywalled). Direction matches the host's three-segment manual trace: a curvature-following extrapolation shortens FL and raises PA relative to a straight line.
- Hamstring wide-FOV reliability study (PLOS ONE 2022, pone.0277400): when the whole fascicle is visible (no extrapolation), intra-rater FL ICC 0.92 to 0.98 with SEM 1.1 to 2.4 mm, PA ICC 0.77 to 0.87 with SEM 1 to 1.6°, MT SEM 0.8 to 1.4 mm — the digitisation-only noise floor is 1 to 2.5 mm FL; everything above that in extrapolated muscles is extrapolation/convention.
- DL_Track_US v0.3 README (GitHub): confirms IFSS-Net (BiLSTM temporal context, Dice+BCE) "performed best in a trade-off between pennation angle and fascicle length RMSD" over SegFormer / VGG16-U-Net / U-Net3+ on videos, but all numbers are in figure files; the values quoted in the repo review (FL RMSD IFSS 3.16, SegFormer 4.67, VGG 5.89 vs UltraTimTrack 2.65 mm) are read off those figures. v0.3 also applies a Hampel filter per frame and Savitzky-Golay over the median fascicle.
- Not retrievable: Van Hooren 2020 systematic review (403), Ripley/Comfort/McMahon 2022 BFlh three-equation comparison (paywall, API rate-limited), the Harvard "U-net-LSTM fascicle length" preprint (PDF not text-extractable). Leads only.
- Practical synthesis for FL: (1) weighted median over many fascicle candidates rather than a single/mean fascicle (TimTrack −63 %); (2) choose fascicles needing the least extrapolation (host protocol) and extrapolate with a curvature-following 2-segment rule rather than a straight line in VL/RF (Bizet 2025 direction, host's 3-segment trace); (3) use MT/sin(PA) as a regulariser/outlier gate, and as the primary estimate only where the extrapolated fraction is small (TimTrack reaches rater-level there); (4) median over the 5 clip frames (UltraTimTrack shows per-frame line-detection noise of ~3.5 mm that fusion more than halves).

---

## 4. Fusing heterogeneous segmentation models for a derived scalar

- No study was found that directly compares probability-map averaging vs mask voting vs measurement-level averaging/median for a derived clinical scalar. Indirect evidence:
  - HC18 leaderboard: a 25-model mean-mask ensemble scored 1.69 mm, no better than single 2019 models (1.72); the same user's nnU-Net 5-fold ensemble (0.81 ± 1.26 mm) was worse than one fold (0.74 ± 0.73 mm), with the SD indicating that map averaging introduced outliers rather than removing them (unverified entries).
  - J Pathol Inform 2023 (Comparing ensemble methods... micrograph cell segmentation, DOI 10.1016/j.jpi.2023.100304): mean vs median vs alpha-stable location vs majority vote on class probabilities — "the choice of the ensemble averaging method only has a marginal influence" on accuracy and Dice; the mean is competitive.
  - Learned pixel-wise stacking (Sarcheshmeh 2026) claims to beat simple averaging on HC error (0.91 ± 1.32 mm) but the ablation numbers were not readable.
  - Majority voting "erodes boundaries and under-segments thin structures" (MedSAM2-Anatomy preprint, arXiv 2608.00195) — anecdotal, no numbers.
  - Scalar-level fusion is where the measured gains are: Pan 2019 (−17 % MAD), Iglovikov (−4 to −7 %), DSB-2016 2nd (CDF-level weighted ensemble + consensus fallback, −11 % CRPS), DSB-2016 1st (contour areas averaged across two CNNs, then volume). TimTrack's measurement-level weighted median (−63 % FL RMSD) is the in-domain analogue.
  - PSFHS shows Dice and derived-angle rankings invert, so optimising the fusion for Dice (which map averaging does) is not the same as optimising it for the measurement.
- Recommendation supported by the above: for thin, oriented structures fuse at the measurement level (median of per-model/per-estimator PA, FL, MT with a consensus-distance outlier rule as in DSB 2nd) or at the edge/line level; if probability maps must be fused, do not rely on a 0.5 threshold of the mean for thin fascicles. FLAG: this is inference from the evidence, not a published comparison.

---

## 5. Architectures and losses for thin, line-like structures in ultrasound (last 3 years)

- Topology-aware losses (no ultrasound datasets in either paper):
  - clDice (Shit et al., CVPR 2021, alpha = 0.5): Dice changes −0.47 to +0.96 points; clDice +0.3 to +4.4 points (DRIVE FCN 78.0 to 81.0; Roads FCN 70.8 to 75.2; Vessap U-Net 90.2 to 92.8); Betti-0/1 errors fell in every row. Cost: +88 % training time, +52 % VRAM on nnU-Net (per the Skeleton Recall paper); out of memory for 13-class TopCoW.
  - Skeleton Recall Loss (Kirchhoff et al., ECCV 2024, nnU-Net backbone, w in {0.1, 1.0}, dilated "tubed" skeleton): Roads Dice 78.99 to 79.25, clDice 88.79 to 89.06, Betti-0 5.77 to 4.85; DRIVE Dice 80.87 to 80.99, clDice 80.26 to 80.83, Betti-0 57.0 to 38.75; Cracks 94.59 to 94.88; ToothFairy 71.80 to 74.42 Dice; TopCoW multi-class 85.36 to 86.59 (clDice OOM). Overhead +8 % time, +2 % VRAM. Caveat: with an HRNet backbone on DRIVE, Dice dropped 80.34 to 76.08 while clDice rose 77.83 to 84.20 and Betti-0 fell 149 to 27 — connectivity gains can cost overlap. For fascicles, where the downstream quantity is orientation/length per fragment, fewer broken fragments matter more than Dice, and this loss is cheap enough to add to the fascicle branch.
- Backbone comparisons in ultrasound:
  - Ulnar nerve 2025 (PMC12843675, six architectures x several encoders): U-Net+ResNet34 Dice 0.8808 best; UNet++ 0.8746; DeepLabV3+ 0.8737; PAN 0.8727; LinkNet 0.8699; SegFormer MiT-b1 0.8685. "Newer architectures do not provide any metric improvements over traditional U-Net"; rotation, shear and resize augmentations were significant, flips and contrast were not.
  - Nerve ultrasound with a diffusion+edge module (PMC13391844, brachial plexus 5,635 images): nnU-Net 89.9 > UNet++ 88.1 > residual U-Net 86.4 > U-Net 84.1 > TransUNet 82.0 Dice (all with their module); ablation on Attention U-Net 0.84 to 0.91. FLAG: internal inconsistencies (94.6 vs 94.2 vs 0.91 for the same model; accuracy equals Dice in a table).
  - Rectus femoris (PMC11726732, n = 38 test, one cohort): AutoSAM Dice 0.9205 / HD 1.72 > nnU-Net 0.8910 / 3.21 > U-Net 0.8178 / 4.27; CSA r² 0.92 / 0.86 / 0.67. Large structure, not a line.
  - CAMUS architecture benchmark (arXiv 2601.00839): plain U-Net ~94 % Dice best overall; TransUNet strongest on hard frames with self-supervised pretraining; native dynamic range (NIfTI) beat 16-bit PNG by ~3 Dice points.
  - nnU-Net Revisited (MICCAI 2024, arXiv 2404.09556), six 3D datasets: original nnU-Net 86.8 avg Dice; ResEnc XL 87.6; MedNeXt L 88.1; SwinUNETR 83.7; nnFormer 83.1; CoTr 85.7; U-Mamba 86.6 to 87.1. "Transformer-based architectures fail to match the performance of CNNs" when trained from scratch on benchmark-sized data.
  - In-domain (DL_Track_US v0.3 README figures, values per repo review): fascicle IoU SegFormer 0.29 vs VGG16-U-Net 0.22 vs U-Net3+ 0.21 vs IFSS 0.20 on 120 unseen images; but on video FL RMSD IFSS 3.16 mm < SegFormer 4.67 < VGG 5.89, all worse than UltraTimTrack 2.65 — higher fascicle IoU did not translate into lower FL error.
  - Needle (line) segmentation in ultrasound: tip error 0.37 ± 0.29 mm and angle error 1.19 ± 0.29° reached by tuning loss-function combinations rather than backbones (arXiv 2407.18357); no backbone head-to-head retrieved.
- Synthesis: in ultrasound thin-structure work, backbone swaps move Dice by about 1 point and sometimes in the wrong direction; resolution, augmentation (rotation/shear/scale), native intensity range and a residual-encoder U-Net recipe are the documented levers; topology losses change connectivity far more than Dice at near-zero cost (Skeleton Recall). No paper in the last three years reports a transformer winning on a line-like ultrasound structure under controlled training.

---

## What could not be verified (do not cite as fact)
- HC18 0.74 mm nnU-Net entry: leaderboard comment only; below the ellipse-convention floor reported by SonoBase.
- Bone-age winner's 500x500 input / 5-net ensemble / TTA-10: Kalinin slides, unreadable PDF.
- PSFHS top-team ensemble/TTA details and inter-observer AoP: not in the retrieved text.
- Stacking-ensemble ablation (average vs meta-learner) numbers: table images only.
- Bizet 2025 mm differences, Ripley 2022 BFlh equations, Van Hooren 2020 numbers, Harvard U-Net-LSTM FL paper: paywalled or not text-extractable.
- Any quantified TTA gain for a derived measurement: only anecdotal ("helped a lot", DSB 2nd) and a breast-US Dice snippet (0.708 to 0.754).

---

## Sources
- HC18 challenge paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0200412
- HC18 live leaderboard: https://hc18.grand-challenge.org/evaluation/challenge/leaderboard/
- Fiorentino et al. 2021 regression framework (Europe PMC record): https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.cmpb.2020.105771&resultType=core&format=json
- Budd et al. 2019 Confident HC (ar5iv): https://ar5iv.labs.arxiv.org/html/1908.02582
- Huawei Cloud HC18 1st place 2018: https://intl.huaweicloud.com/news/huawei-cloud-ei-wins-first-place-in-the-international-ultrasonic.html
- DAG V-Net HC18: https://link.springer.com/article/10.1007/s11548-021-02430-0
- Stacking ensemble HC (Sarcheshmeh 2026): https://link.springer.com/article/10.1007/s44163-026-02026-0
- SonoBase open ultrasound foundation model (recalibration, HC floor, few-shot): https://arxiv.org/html/2609.19230
- Halabi et al. 2019 RSNA bone age challenge: https://pubs.rsna.org/doi/10.1148/radiol.2018180736
- Pan et al. 2019 bone-age ensembles: https://pmc.ncbi.nlm.nih.gov/articles/PMC6884060/
- Iglovikov et al. 2018 bone age: https://arxiv.org/html/1712.05053v2
- 16 Bit winner details (press): https://www.santelog.com/actualites-sante-nasdaq/rsna17-ai-bone-age-challenge-champion-16-bit-offers-award-winning-algorithm and https://atlas.rsna.org/cards/909f06e2-61ac-4576-ae2b-55dd92632307
- Kalinin RSNA 2017 slides (unverified): https://alxndrkalinin.github.io/pdf/2017-12_CFT_BoneAge.pdf
- DSB 2016 1st place: https://github.com/woshialex/diagnose-heart
- DSB 2016 2nd place: https://317070.github.io/heart/
- DSB 2016 3rd place: http://juliandewit.github.io/kaggle-ndsb/
- PSFHS challenge report: https://arxiv.org/abs/2409.10980 ; top-8 code: https://github.com/maskoffs/PS-FH-MICCAI23 ; MiTU-Net with results table: https://arxiv.org/html/2401.15513v1
- EchoNet-Dynamic: https://pmc.ncbi.nlm.nih.gov/articles/PMC8979576/
- Segmentation ceiling for EF regression: https://arxiv.org/abs/2609.19730
- CAMUS original: https://arxiv.org/pdf/1908.06948 ; CAMUS architecture benchmark 2026: https://arxiv.org/abs/2601.00839
- EchoNet-LVH: https://ai.stanford.edu/~bryanhe/publications/lvh.pdf
- Carotid IMT agreement: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3918749/ ; https://arxiv.org/pdf/1706.00719
- AutoFB fetal biometry: https://arxiv.org/pdf/2107.05255 ; modular fetal biometry 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12859377/
- Bayesian linear vs neural-ODE calibration: https://arxiv.org/html/2605.09227v1
- Quantile-mapping comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC7922393/
- Van Calster shrinkage simulation: https://lirias.kuleuven.be/bitstream/123456789/655306/2/Regression%20shrinkage%20for%20clinical%20prediction%20models%202nd%20REVISION%20SUBMITTED%20TO%20SMMR.pdf ; calibration hierarchy: https://bmcmedicine.biomedcentral.com/track/pdf/10.1186/s12916-019-1466-7
- Niculescu-Mizil & Caruana 2005: https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting
- TimTrack: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265752
- UltraTimTrack: https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/
- Bizet et al. 2025 (Europe PMC record, DOI 10.1016/j.jbiomech.2025.112520): https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=AUTH:Bizet%20AND%20fascicle%20AND%20PUB_YEAR:2025&resultType=core&format=json
- Hamstring WFOV reliability: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277400
- Franchi 2020 extrapolation pitfalls: https://pubmed.ncbi.nlm.nih.gov/31403609/
- DL_Track_US README: https://github.com/PaulRitsche/DL_Track_US
- clDice: https://arxiv.org/html/2003.07311v7
- Skeleton Recall Loss: https://arxiv.org/html/2404.03010v2
- nnU-Net Revisited: https://arxiv.org/html/2404.09556v2
- Ulnar nerve architecture comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC12843675/
- Nerve diffusion + edge fusion: https://pmc.ncbi.nlm.nih.gov/articles/PMC13391844/
- Rectus femoris AutoSAM/nnU-Net/U-Net: https://pmc.ncbi.nlm.nih.gov/articles/PMC11726732/
- Ensemble aggregator comparison (J Pathol Inform 2023): https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Comparing%20ensemble%20methods%20combined%20with%20different%20aggregating%20models%22&resultType=core&format=json
- MedSAM2-Anatomy (majority vote erodes thin structures, anecdotal): https://arxiv.org/pdf/2608.00195
- Needle segmentation loss study: https://www.arxiv.org/pdf/2407.18357
- Repo baseline consulted: C:\Projects\muscle-arc\docs\literature_review.md
