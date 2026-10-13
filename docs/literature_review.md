# Literature review: automated muscle-architecture estimation from 2D B-mode ultrasound

Scope: everything a top entry in the UMUD Challenge needs to know about the field, written
for this repo. Sections 1 to 3 are the competition's own lineage (the host team's DL_Track
work, the exact conventions their tool and their manual protocol use). Sections 4 to 7
cover the wider field (classical, tracking, deep learning, measurement science). Section 8
turns all of it into design decisions for `muscle_arc`.

Terms: PA = pennation angle (deg), FL = fascicle length (mm), MT = muscle thickness (mm),
apo = aponeurosis, GM = gastrocnemius medialis, VL = vastus lateralis, TA = tibialis anterior.

Status: compiled 2026-10-13. Numbers quoted are from the cited sources; where a source does
not state a number this review says so rather than guessing.

---

## 1. The competition and its metric

- Host: Paul Ritsche, Gerardo Romeny, Oliver Faude (University of Basel). Deadline
  2026-11-15, private board 2026-11-22. 309 test images (251 TIFF, 58 PNG) incl. 27
  five-frame video clips (IMG_00056 to IMG_00195). Public board = 33 %.
- Metric (host notebook `paulritsche/umud-score`): S = (1/3)(MAE_PA/6 + MAE_FL/12 + MAE_MT/3),
  MedAE and RMSE as 1e-6 / 1e-9 tie-breakers. The divisors are "based on literature MDCs"
  (minimal detectable change), see section 7.
- Consequence: 1 mm of MT error costs as much as 2° of PA or 4 mm of FL. MT is the lever.
- Ground truth: manual analysis by two raters (Ritsche, Romeny) following the DL_Track_US
  manual protocol (section 3). Sample-submission values are made up (host, forum 2026-09-27).
- Host rulings that shape method design (forum threads, Sep to Oct 2026):
  - bottom ticks are 1 cm apart; use the vertical ruler as the single px/cm; pixels square;
  - the 512x512 training images have no recoverable scale;
  - manual measurement of the 309 test images may be used to fine-tune or calibrate an
    automated pipeline if declared as external data (ruled prize-eligible);
  - leaderboard-weighted ensembling is not eligible; manually chosen weights are. Criterion:
    "can your final submission be re-run on a new dataset where no leaderboard exists?"
  - OSF "Expert Analysed Benchmarks" (35 images, 7 raters) are the host-recommended
    calibration set.

## 2. Base papers: the DL_Track lineage (host team)

### 2.1 Cronin, Finni, Seynnes 2020 (arXiv:2009.04790), "Fully automated analysis of muscle architecture from B-mode ultrasound images with deep learning"

- Code github.com/njcronin/DL_Track (Apache-2.0, superseded). The host told participants this
  preprint is the public stand-in for the closed-access 2024 paper.
- Data: about 570 apo and 310 fascicle frames from videos and stills, four devices, GM, GL,
  VL, TA (soleus in examples); resized to 512x512; 90/10 split; heavy elastic-deformation
  augmentation; one researcher traced every apo or every visible fascicle fragment (1 to 20
  per image) in Fiji.
- Network: plain 23-layer U-Net per branch, BCE, Adam, batch 1, early stop at 18 to 21
  epochs, under 1 h per model on an RTX 2070. 0.7 s/image GPU, 4.6 s CPU.
- Geometry in `Inference_Single_Image.ipynb`: apo threshold 0.2, fascicle 0.10, fascicle
  contour >= 40 px, min apo separation 60 px, pennation gate 14 to 40°; apo edges smoothed
  (Savitzky-Golay 81, 2) and fitted with an order-2 polynomial (order 1 in later versions);
  fascicle lower edge fitted with a straight line and extrapolated; PA from the deep-apo slope
  over 50 px at the intersection; FL = straight-line distance between intersections; MT =
  shortest distance in the central third; median over fragments compared with the mean of
  three raters.
- Results on 35 unseen images: vs SMA FL 1.1 mm, PA 1.0°, MT 0.2 mm; vs the three-author
  average FL 2.1 mm, PA 0.1°, MT 0.1 mm; peak FL error about 12 mm; video ICC vs UltraTrack
  0.73 (0.45 to 0.98), mean absolute FL difference 2.9 to 6.0 mm per task.
- Author-stated limits: frames processed independently; "can easily be fooled ... by very
  subtle changes in pixel values"; averaging many fascicles adds variability.

### 2.2 Ritsche, Faude, Franchi, Finni, Seynnes, Cronin 2023 (CISS 8(2):088) and Ritsche et al. 2024 (Ultrasound Med Biol 50(2):258-267, doi 10.1016/j.ultrasmedbio.2023.10.011), "DL_Track_US"

- Networks: classic U-Net vs U-Net with VGG16 ImageNet encoder; VGG16-U-Net chosen on IoU
  and loss. Input 512x512x3, pixel /255. Loss Dice+BCE (0.5/0.5), Adam 1e-4, early stop 20,
  5-fold (from the host's public Kaggle notebook `paulritsche/dl-track-0-3-1-aponeurosis-model`).
- Training data: 310 labelled fascicle images and 570 aponeurosis images, augmented to about
  1,700 each; muscles VL, GM, TA, soleus; four devices; random 80/20 split.
- Test: 35 unseen images vs manual, SMA and UltraTrack (videos).
  - vs manual: FL -2.4 mm (95 % CI -3.7 to -1.2), PA +0.6° (-0.2 to 1.4), MT -0.6 mm
    (-1.2 to 0.002).
  - vs SMA: FL +5.2 mm, PA -1.4°, MT -0.9 mm.
  - videos vs UltraTrack: ICC 0.19 (GM passive) to 0.88 (calf raise); the authors attribute
    the variance to frame-by-frame independence.
- Stated limits: FL is the weakest parameter; each frame analysed independently; works on
  cropped B-mode frames, not console screenshots.
- Released training code (`model_training.py`): VGG16 encoder with ImageNet weights, decoder
  512/256/128/64, BCE default, Adam default LR, EarlyStopping patience 8, ReduceLROnPlateau
  0.1/10, 10 % validation split; augmentation by Keras ImageDataGenerator (rotation 5°,
  shifts 7.5 %, zoom 7.5 %, horizontal flip), three copies per image.
- Host's news page reports the VGG16-U-Net on the UMUD 35-image expert benchmark: mean
  difference MT 0.13 mm (SD 3.34), FL 3.87 mm (SD 4.58), PA -0.41° (SD 1.94). Small bias,
  large spread: the losses are occasional gross failures, not a constant offset.
- Independent test (van der Zee et al. 2025): RMSD vs three raters 3.8 mm / 3.9° on the GM
  calf-raise sequence and 8.0 mm / 4.8° on a TA sequence; "could not detect a single fascicle
  or the deep aponeurosis in some images".
- Ritsche, Schmid, Franchi, Faude 2022 (Front Physiol 13:981862), two devices analysed with
  SMA: GM FL agreement ICC 0.84, SEM 4.5 mm; RF FL ICC 0.46, SEM 8.6 mm; VL FL ICC 0.80; MT
  ICC about 0.9 everywhere. MT is the most reliable variable, RF and VL FL the least.

### 2.3 Later versions (v0.3.0, v0.3.1; GitHub PaulRitsche/DL_Track_US)

- v0.3.0 added manual scaling, Hampel + Savitzky-Golay filtering, stacked (video) mode;
  v0.3.1 added "weighted aponeurosis calculation" and frame queuing. The README reports that
  SegFormer, UNet3+ and IFSS-Net (Chanti et al. 2021, with a bidirectional LSTM for temporal
  context) were compared on 120 unseen images and a validation video; IFSS-Net gave the best
  PA/FL RMSD trade-off. The host's Kaggle benchmark CSV "DLTrack_0.3.1" scores 0.679 public.
- Shipped defaults (`gui_files/settings.json`): apo threshold 0.1, apo length threshold 600,
  fascicle threshold 0.01, fascicle length threshold 40, minimal muscle width 200,
  pennation range 10 to 30°, method `linear_extrapolation`, Hampel window 3 / 2 SD,
  stacked segmentation. The authors' documented defaults (`_backup_settings.py`) differ:
  0.2 / 400 / 0.05 / 40 / 60 / 10 to 40° / `curve_connect_linear`. (Our vendored
  `do_calculations.py` is the `linear_extrapolation` path; the curved methods exist in
  `do_calculations_curved.py` but the GUI stops with "not yet fully implemented" for them.)
- v0.3.1 video code changed MT to the minimum vertical difference in the central third and
  added "tangential extrapolation" of apo ends (changelog 2025-11-18).
- Scale-bar reader (`calibrate.py`): scans the bottom 60 % x right 15 % of the frame, marks
  rows whose max > 150, and takes the largest gap between bright rows as the bar distance.
  It never uses the spacing value in a computation. This is why stock DL_Track transfers
  badly to console screenshots: the ruler region and tick pattern differ per device.

### 2.4 The exact DL_Track geometry (from the vendored code, what the 0.45 to 0.68 public entries use)

1. Apo mask: contours > 600 points kept, merged if endpoints within 10 px, skeletonised,
   dilated 15x and eroded 10x with a 3x7 kernel (net thickening of about 5 px per edge),
   sorted top to bottom. Upper apo uses its bottom edge, lower apo its top edge; a third
   contour is used if the second is closer than `min_width`.
2. Edges smoothed with Savitzky-Golay (window 81, order 2), then fitted with a degree-1 line
   and extrapolated 700 px either side of the centre.
3. MT ("Thickness at Centre"): shortest Euclidean distance from the upper-edge point at the
   middle of the central third of the overlap to any lower-edge point in that third. Not
   vertical, not averaged over three positions.
4. Fascicles: contours > 40 points inside the band between the apos; the bottom edge of each
   contour is fitted with a straight line and extrapolated to both apo lines. FL is the
   straight-line distance between intersections (pure linear extrapolation).
5. PA per fascicle = (90 + |apo angle to horizontal near the lower intersection|) minus the
   fascicle's angle to the vertical; kept only within [min, max] pennation (10 to 40 in code,
   10 to 30 in shipped settings). Optional overlap filter removes fascicles whose x-extent
   contains a neighbour's.
6. Per image: median of fascicle lengths, median of pennation angles (after a Hampel filter
   win 4 / 1 SD), single MT. Videos: Savitzky-Golay (11, 3) over frames plus a rate limit on PA.

Implications: the public anchors inherit (a) a PA floor at 10°, (b) FL biased long by
linear extrapolation through curved fascicles, (c) MT measured between edges thickened by
the morphology step (biased thin by roughly 10 px, 0.5 to 1 mm at test-set scales).

## 3. The host's manual protocol (what the test labels are)

Kaggle Data page: test images come from Siemens Acuson Juniper, Telemed ArtUS EXT-1H and
Philips Lumify; muscles VL, GM and rectus femoris; healthy and cerebral-palsy participants,
18 to 28 y, 59 % male, disjoint from training; public/private split random by image. Stated
test ranges PA 5 to 45°, FL 30 to 200 mm, MT 10 to 50 mm. Reference annotation by two
researchers: superficial and deep apo, 3 visible fascicles, 3 pennation angles, 3 thickness
lines; FL "along the fascicle line" with "linear extrapolation techniques" when the fascicle
leaves the frame; MT "perpendicular distance ... at three locations distributed over the
muscle width"; rater discrepancies above 10 mm FL, 4° PA or 1 mm MT were re-evaluated.
Training masks (2,761 fascicle, 1,048 apo images from Juniper, ArtUS, Echo Blaster 128,
Aloka Alpha-10, Philips HD11; VL, GM, GL, TA) follow the 2024 paper's labelling.

The OSF expert benchmark README (v0.1.0, six raters, GM/soleus/VL, four devices) states the
same protocol: "3 straight lines ... at a left, middle and right location" for MT, "3
segmented lines" for fascicles, "3 angles" for PA "(not the same as for the fascicles)", mean
of each. Six-rater statistics are not yet published.

From the DL_Track_US manual-analysis documentation and the host's forum answers:

- Scale: one 1 cm line drawn on the ruler.
- Apos: drawn/extended manually first.
- MT: three straight lines "from superficial to deep aponeurosis across the muscle image",
  averaged. Host (forum): lines are perpendicular to the global image frame (vertical), not to
  either apo; the stated reason is comparability with DL_Track and a ~0.1 mm bias either way.
- FL: at least three fascicles in different regions, each drawn as three connected segments
  (so curvature is allowed), extended to the apos; raters chose fascicles that needed the
  least extrapolation (host, forum). Averaged.
- PA: at least three angles, each as two segments "along the fascicle" and "along the deep
  aponeurosis", i.e. measured at the fascicle's insertion against the local deep-apo direction.
  Averaged. Host (forum 748126): "the exact length of the part of the lower aponeurosis taken
  into account is not standardised", and "we only used the vertical scale bars" even on the
  stretched 1200x800 images.
- Ground truth = mean of the two raters. For images with more than two apos (e.g. TA with a
  central apo) the superficial compartment is the target (host, forum 2026-10-02).

Where this differs from the automated DL_Track geometry: three vertical MT lines vs one
shortest distance at the centre; curved three-segment fascicles vs straight extrapolation;
three hand-picked fascicles vs median of all; no 10° floor. Our `geometry/protocol.py`
(`host_v1`) implements the manual convention; on the OSF set it lost to the legacy median
geometry because mean-of-3 is fragile on imperfect masks, not because the convention is wrong.

## 4. Classical (non-learning) methods for single images

### 4.1 Line-transform family (Hough, Radon)

- Zhou & Zheng 2008 (Ultrasound Med Biol 34:1474, doi 10.1016/j.ultrasmedbio.2008.02.009),
  Revoting Hough Transform: take the accumulator maximum, erase its supporting edge pixels,
  re-vote, repeat; avoids one true line producing several Hough lines. Vs manual: R² 0.965,
  mean difference 0.18 ± 2.41°. About 0.14 s per line.
- Zhou, Li, Zhou, Zheng 2012 (Biomed Eng Online 11:63): Gabor + RVHT for the deep apo;
  fascicle orientation from a least-squares gradient structure tensor with a reliability gate
  (R > 0.6) and the median of reliable pixels; PA = fascicle minus apo orientation. One subject,
  qualitative agreement only.
- Ling et al. 2013 (Biomed Eng Online 12:98): Frangi vesselness, Otsu, connected components;
  elongated components (aspect > 5, width < 30 px) take their moment-ellipse angle, others fall
  back to Hough; stop after 7 lines. Vs two operators on biceps: 0.11 ± 0.92°; GM at 0 to 90 %
  MVC: 0.03 ± 2.72°. 0.036 s per line.
- Han et al. 2013 (Biomed Eng Online 12:6): enhance (Gabor or Frangi) then RVHT for the two
  apos; MT = mean line distance. Vs manual MT 1.4 mm bias, ICC 0.97, SEM 0.3 mm; the 1.4 mm is
  explained by measuring to the apo centre line instead of its inner edge (apo thickness
  1.51 mm). Without enhancement the apos were not the first two lines in 110 of 300 images.
- Zhao & Zhang 2011 (IEEE TBME 58:2094): localised Radon transform with revoting for tracking
  multiple fascicles. Liu et al. 2019 (IEEE TIP 28:3714): Gabor, revoting-constrained Radon
  ROI, Radon orientation, adaptive-fading Bayesian Kalman smoother over frames.
- Rana, Hamarneh, Wakeling 2009 (J Biomech 42:2068): multiscale vesselness then Radon (global
  orientation) or wavelets (local orientation, curvature); synthetic error < 0.06°; ten manual
  raters spread SD 1.41°; wavelet read 1.35° lower than manual. Namburete, Rana, Wakeling 2011
  extend it to curvature fields.
- Zhou, Chan, Zheng 2015 (Ultrasonics 57:72): Gabor wavelet line extraction for PA and FL,
  r = 0.94 to 0.95 vs manual on GM. Zhou & Zheng 2015 (IEEE TBME 62:2828): orientation-
  sensitive segmentation then Lucas-Kanade, CMC 0.97 vs 0.87 for earlier automatic methods.
- Bao et al. 2023 (Wearable Technologies 4:e6, PMC10936288): DBSCAN clustering of bright
  pixels into tubular clusters, orientation merge, highest-value cluster, viscosity temporal
  fallback; TA, RMSE 0.73° vs manual (UltraTrack 1.95°, ImageJ 3.6 to 4.7°); orientation only.

### 4.2 Head-to-head estimator comparison

Pohle-Fröhlich et al. 2020 (VISAPP, arXiv:1912.04134), GM walking and running, 425 frames, three
experts (spread 3.2° overall, so about 2° is the floor): ROI between vesselness-detected apos,
angle search restricted to 15 to 70°, eight overlapping sub-regions scored by gradient, best
three kept, LOESS over frames. MAE vs expert mean: Frangi-then-projection-profile 1.96° (walk)
and 1.87° (run), ICC 0.946; Radon 2.01 / 3.36°; projection profile alone 0.55° walking but
12.9° running. Deep-apo slope MAE 0.3 to 1.1°, with a roughly 1° PA offset because the
algorithm averaged the slope across the width while experts used the right end.

### 4.3 Complete classical pipelines

- SMA, Seynnes & Cronin 2020 (PLoS ONE 15:e0229034), ImageJ macro: CLAHE, Tubeness (sigma
  about 8 to 10), non-local means, FFT masking, Canny, search for two parallel bright lines;
  fascicles by OrientationJ structure tensor (sigma 4) in 2 to 3 ROIs; PA = deep-apo angle +
  fascicle angle; FL = straight line along the dominant angle between apos with linear apo
  extrapolation; MT = mean apo separation. PA bias <= 1°, FL bias up to 5.6 mm, MT < 1 mm;
  8.4 s per frame.
- MUSA, Caresio et al. 2017 (Ultrasound Med Biol 43:195): fully automatic apo detection for MT
  only; 200 images (RF, VL, TA, GM), 100 % success, 0.06 to 0.45 mm vs three operators.
- Marzilger et al. 2018 (Eur J Appl Physiol 118:291): object-based VL fascicle segments merged
  into a representative fascicle and extrapolated; between-day ICC FL 0.86, MT 0.96, PA 0.60.
  Luera et al. 2025 (Front Rehabil Sci 6) shows the convention gap in VL: semi-automatic
  bias -4.0 cm FL and +3.0° PA vs manual despite ICC 0.9 to 0.99.

## 5. Tracking methods for video

- Cronin et al. 2011 (J Appl Physiol 111:1491): Lucas-Kanade affine optical flow on a user-
  drawn fascicle; CMC 0.90 vs manual during walking. Gillett et al. 2013 (CMBBE 16:678):
  agreement 0.88, 0.94 after removing the first-frame offset; error 5 to 10 % of FL, mostly
  from manual initialisation (Aeles et al. 2017, PeerJ 5:e4164: inter-rater SEM 4.2 mm).
- UltraTrack, Farris & Lichtwark 2016 (Comput Methods Programs Biomed 128:111): KLT corner
  features plus affine transform per frame, key-frame drift correction. Drift accumulates
  (67 mm FL and 9.3° over 120 cycles); fails under non-affine deformation (Drazan 2019: RMSE
  14.3 mm / 23.0° from first-frame initialisation on maximal contractions).
- Drazan, Hullfish, Baxter 2019 (PeerJ 7:e7120): 100 min-eigenvalue points per structure,
  MATLAB point tracker (3 mm block, 4 pyramid levels, 2 mm forward-backward check), reseed when
  > 10 % lost, linear regression through each cloud, FL by line intersections, PA vs deep apo.
  Unsupervised RMSE 5.9 mm / 7.2°, supervised 3.3 mm / 4.1°; 60 fps required.
- TimTrack, van der Zee & Kuo 2022 (PLoS ONE 17:e0265752): per-frame, drift-free. Frangi at a
  thin scale for fascicles and a thick scale for apos; Hough per depth ROI, traced to the inner
  edge, constrained polynomial fit; fascicle angle = accumulator-weighted median of top-K Hough
  peaks in an elliptical ROI; MT = vertical gap times cos(beta); FL = MT / sin(phi). 339 images,
  26 subjects: RMSD vs manual MT 0.11 cm, PA 1.82°, FL 0.52 cm, essentially equal to manual-
  vs-manual (0.06 cm, 1.83°, 0.56 cm). 0.81 s per image.
- HybridTrack, Verheul & Yeo 2022 (bioRxiv 10.1101/2022.04.20.488774): Frangi + Hough with KLT
  apo motion; TA only; no manual comparison in the paper; later measured at 4.7 mm / 0.9° (TA)
  and 5.8 mm / 7.9° (GM).
- UltraTimTrack, van der Zee et al. 2025 (PeerJ CS 11:e2636): Kalman filter with UltraTrack
  flow as the process model and TimTrack detections as measurements, RTS smoother. Drift 2.1 mm
  / 0.8° over 120 cycles; RMSD vs manual 2.3 to 2.6 mm and 0.8 to 0.9° where DL_Track gave
  3.8 to 8.0 mm and 3.9 to 4.8°. Assumes straight fascicles and apos. 0.2 s per frame.
- DUSTrack 2025 (section 6.3) matches UltraTimTrack with per-video fine-tuned keypoints.
- Bizet et al. 2025 (J Biomech 180:112520): straight-line tracking in VL over-reads FL and MT
  and under-reads PA relative to 2- and 4-segment manual extrapolation, worst distally.

### 5.1 What transfers to five-frame clips

Drift is irrelevant over five frames; the useful idea is the inverse of a tracker: estimate
every frame independently (drift-free) and fuse. KLT with forward-backward checks (block
about 3 mm, 4 pyramid levels, error gate 1 to 2 mm) can propagate detections between
neighbouring frames and flag out-of-plane frames to down-weight, which is a five-frame
version of UltraTimTrack's prediction/correction or DUSTrack's overlapping-tracklet mean.
Per-frame line detection noise is about 3.5 mm / 1.4° (TimTrack cycle-to-cycle), so a
median over five frames removes roughly half of it.

## 6. Deep-learning methods 2017 to 2026

### 6.1 Direct regression and orientation fields

- Cunningham, Harding, Loram 2017 (MIUA, doi 10.1007/978-3-319-60964-5_6) and Cunningham,
  Sánchez, May, Loram 2018 (J Imaging 4:29, doi 10.3390/jimaging4020029): CNN/ResNet and an
  encoder-decoder "DCNN" predicting a dense per-pixel fibre-orientation map (incl. soleus)
  without prior segmentation; 8 volunteers, 25 Hz calf video; "approximately 6° error" for
  full-field orientation. Labels were interpolated from regional annotations. No code.
- Cunningham & Loram 2020 (J R Soc Interface 17:20190715): two-branch CNN on reference+target
  frame pairs regressing EMG, torque and angle, 32 participants, 1M pairs; not PA/FL/MT.
- Hafthorsdottir et al. 2022 (IEEE SAS) and Vostrikov et al. 2024 (IEEE TIM 73): PA regressed
  from raw RF channel data with autoencoder + XGBoost, MAE 0.43° on a small GM set whose labels
  were themselves automatic; shows PA is learnable as a scalar but does not transfer to B-mode
  screenshots.
- Gionfrida et al. 2024 (IEEE BioRob): CNN+LSTM / transformer on simulated single-element
  M-mode traces, PA RMSE about 1.1°, FL 2.54 mm; preliminary. Code
  github.com/raku-slyu/AB-Mode-Utrasound.
- Rosa et al. 2021 (PLoS ONE 16:e0246611): classical regressors on raw pixels for soleus FL,
  r = 0.70 within task but 0.22 across tasks: pixel regression does not generalise.
- No peer-reviewed 2023 to 2026 work applies a transformer end-to-end to B-mode images to
  regress PA/FL/MT.

### 6.2 Segmentation approaches

- Katakis et al. 2022 (Sensors 22:5230, doi 10.3390/s22145230), the closest single-image
  comparator: Attention U-Net at 256x256, weighted Dice, 433 longitudinal images (biceps 188,
  GM 128, TA 117) from one GE Logiq P9. Aponeuroses = two largest components, skeletonised,
  gaps bridged by a 2nd-order polynomial; fascicles NOT learned (CLAHE, ridge filter, k-means,
  line fitting, orientation clustering), FL by trig extrapolation. MT RMSE 0.40 mm (ICC 0.99),
  PA RMSE 2.22° (GM 3.0°, TA 1.75°), FL RMSE GM 7.65 mm, TA 18.4 mm, biceps 87.9 mm (parallel
  fascicles break extrapolation). 2.5 s/image. No code.
- DL_Track_US v0.3 (README, unpublished): fascicle IoU on 120 unseen images SegFormer 0.29,
  VGG16-U-Net 0.22, U-Net3+ 0.21, IFSS 0.20 (thin masks, low IoU for all). Video RMSD vs
  manual on a GM calf raise: FL IFSS 3.16 mm, SegFormer 4.67, VGG 5.89 vs UltraTimTrack 2.65;
  PA 1.2 to 1.6° vs 0.90°. Even the newest DL_Track model does not beat a Kalman-filtered
  classical tracker on video.
- Transverse CSA work gives backbone and data-scale evidence only: Marzola et al. 2021
  (3,917 images, public on Mendeley), Katakis et al. 2023 (TMUNet, DSC 0.96), Zhou et al. 2023
  MMA-Net, DeepACSA (Ritsche 2022, 1,772 panoramic images, ICC 0.97 to 0.99).
- OCRSeg (Zhang et al. 2025, PMLR 281:42): one-stage vertical contour regression (encoder + MLP
  with total-variation smoothness) for muscle boundaries, >100 FPS: the nearest published
  "boundary as a 1D function" formulation, close to our SOF surface heads.
- Keypoint heatmaps are proven for MSK points: DeepMTJ (Leitner et al. 2022, IEEE TBME, doi
  10.1109/TBME.2021.3130548): attention U-Net on 256x128, Gaussian heatmaps + weighted BCE,
  66,864 frames, 3 devices, RMSE 4.89 mm vs specialists 5.05 mm; per-device RMSE varies 1.4x
  (Telemed 3.6, Esaote 5.1, Aixplorer 5.2 mm). Nobody has published a deep-Hough or
  line-heatmap network for fascicles.

### 6.3 Tracking with learning

- DUSTrack (Namburi et al. 2025, arXiv:2507.14368; Sci Rep 2026): per-video fine-tuned
  DeepLabCut ResNet-50 keypoints (about 25 labelled frames) + Lucas-Kanade tracklet filter;
  five points (two per apo, one on a fascicle) give FL and PA by geometry. Beats zero-shot
  CoTracker3/BootsTAP/LocoTrack/PIPs++ (RMSE about 200 um); on GM plantarflexion not
  significantly different from UltraTimTrack; DL_Track and HybridTrack excluded for "severe
  failure cases". Code github.com/praneethnamburi/DUSTrack.
- UltraTimTrack (van der Zee et al. 2025, PeerJ CS 11:e2636): Kalman fusion of UltraTrack
  optical flow and TimTrack Hough detection; RMSD vs manual 2.3 to 2.6 mm FL and 0.8 to 0.9°
  PA, versus DL_Track 3.8 to 8.0 mm and 3.9 to 4.8° on the same sequences. The accuracy
  reference for video. Code github.com/timvanderzee/UltraTimTrack.
- No RAFT/FlowNet muscle paper exists; all fascicle trackers use Lucas-Kanade.

### 6.4 Foundation and self-supervised models

USFM (arXiv:2401.00153; muscle is 1,545 of 2M pretraining images), USF-MAE (arXiv:2510.22990),
OpenUS (arXiv:2511.11510), UltraSam (arXiv:2411.16222), SAMUS (arXiv:2309.06824): none applied to
fascicles or aponeuroses. MedSAM over-smooths boundaries (bad for MT). A SegFormer/ViT encoder
initialised from USF-MAE is a cheap experiment with no published evidence either way.

### 6.5 Domain shift, chrome, scale

- DeepMTJ is the only MSK work with per-device errors (1.4x spread). DL_Track_US and DeepACSA
  train on multiple devices but report no held-out-device error.
- Mikolaj et al. 2023 (arXiv:2303.13918): burned-in text and calipers act as shortcut
  features; simple masking/inpainting of chrome before training matched or beat U-Net/GAN
  removal and raised accuracy from 85.6 to 92 to 94 %. Crop or mask chrome before training.
- QIMS 2026 cervical-nerve pipeline: YOLOv11 localises the imaging region (IoU 0.99) and OCR
  reads the depth scale for mm conversion, 117,729 images.
- RulerNet (Pan et al., arXiv:2507.07077; CMIG 2026): heatmap tick detector (MobileNetV4 +
  U-Net decoder, CE 0.2 + Dice 0.8) -> Hough grouping -> 1D "DeepGP" spacing regression;
  1.19 to 1.31 px error per cm at 768 px on photographed rulers; placenta photos, not
  ultrasound. Venturini et al. 2024 (arXiv:2401.01201) and an Eramian SPIE paper use
  autocorrelation of the tick profile (94.8 % detection). No learned scale-recovery method
  exists for ultrasound screenshots; a deterministic per-vendor template (what we do) is the
  documented simplest route.

### 6.6 Error ranking across the literature (lower is better; protocols differ)

| Method | PA | FL | MT | Setting |
|---|---|---|---|---|
| UltraTimTrack (Kalman, video) | 0.8 to 0.9° | 2.3 to 2.6 mm | n/a | GM, TA video |
| DUSTrack (fine-tuned keypoints, video) | about UTT | about UTT | derivable | per-video labels |
| DL_Track_US v0.3 IFSS (video, smoothed) | 1.0 to 1.6° | 3.1 to 7.4 mm | n/a | two videos |
| Katakis 2022 Attention U-Net (single image) | 2.2° (GM 3.0°) | 7.7 mm GM, 18 mm TA | 0.40 mm | one device |
| DL_Track_US 2024 (single image, bias) | +0.6° | -2.4 mm | -0.6 mm | 35 images, 4 devices |
| Cunningham 2018 dense orientation | about 6° | n/a | n/a | 8 subjects |

Inter-rater PA spread per frame is 2 to 3° (Pohle-Fröhlich et al. 2020), so about 2° is the
single-image ceiling; the best single-image MT numbers (0.4 mm) are already well inside the
3 mm tolerance, which is why MT is where an automated entry can get closest to the raters.

## 7. Measurement science: reliability, conventions, physiology

### 7.1 How noisy are the labels?

- Kwah, Pinto, Diong, Herbert 2013 (J Appl Physiol 114:761, doi 10.1152/japplphysiol.01430.2011),
  systematic review of 36 reliability and 6 validity studies: FL ICC > 0.6 and CV < 10 %;
  PA ICC > 0.5 and CV < 14 % (PA is the noisier parameter); validity ICC > 0.7 only for large
  relaxed muscles with the joint stationary.
- May, Locke, Kingsley 2021 (PLoS ONE 16:e0258014, doi 10.1371/journal.pone.0258014), n = 87,
  GM and GL, 38 mm probe, depth 3 to 6 cm, three fascicles averaged. Definitions: FL = visible
  length plus linear extrapolation (Ando model V); PA at the deep-apo intersection; MT = mean
  vertical distance between the internal borders of the apos. Between-session MDC95 for GM:
  PA 5.7 to 6.4°, FL 9.8 to 12.1 mm, MT 3.2 mm. These are, to rounding, the UMUD tolerances
  (6° / 12 mm / 3 mm): the metric normalises by whole-protocol re-scan noise, which is 3 to 5x
  larger than pure digitisation noise (intra-rater digitisation ICC 0.99; inter-rater MT RLOA
  about ±3 to 4.5 %).
- Hagoort et al. 2022 (Biomed Eng Online 21:15, doi 10.1186/s12938-021-00967-4), three raters,
  panoramic US, young adults at rest, inter-rater SEM: GM PA 1.8°, FL 5.5 mm, MT 0.6 mm;
  TA PA 1.4°, FL 7.0 mm, MT 0.5 mm; VL PA 1.4°, FL 14.4 mm, MT 1.0 mm. Older adults: VL FL SEM
  39.5 mm (ICC 0.37). MT is the most reliable parameter in every muscle.
- Lesinski et al. 2024 (Front Sports Act Living 6:1282031): inter-rater GM MT SEM 0.3 mm,
  deep-apo PA SEM 1.4° (ICC 0.72) vs superficial-apo PA ICC 0.49; VL FL SEM 10.5 mm
  (LoA ±31.5 mm). Deep-apo PA is 1 to 2° larger than superficial-apo PA in GM and more reliable.
- Automated vs manual, same lineage: DL_Track_US 2024 FL -2.4 mm, PA +0.6°, MT -0.6 mm;
  SMA (Seynnes & Cronin 2020, PLoS ONE 15:e0229034) MT bias < 1 mm, PA bias <= 1°.
- Bolsterlee et al. 2015 (J Biomech 48:1133): fascicles lie 5.5 ± 4.1° out of the image plane;
  per-image FL differences US vs DTI about 10 mm (20 %); US PA 6° smaller than DTI on average.
- UMUD database paper (Ritsche et al. 2026, BMC Med Imaging 26:139): the "MultExpData"
  benchmark is 35 architecture images analysed by six expert operators plus 30 panoramic CSA
  images; "ExpDrawnData" is 250 VL images with fascicles drawn by one expert. GM is about 85 %
  of indexed data.

Noise floor in tolerance units (inter-rater SEM / tolerance): PA about 0.25 to 0.3, MT 0.1 to
0.3, FL 0.2 to 0.6 for GM and TA but about 1.0 for VL. A perfect estimator of the expert
consensus therefore scores roughly 0.2 to 0.3; the current public leader (0.217) is at that floor.

### 7.2 Conventions and the bias each one carries

| Choice | Evidence | Size |
|---|---|---|
| MT vertical vs perpendicular | May 2021 vertical between internal borders; Hagoort perpendicular at widest point; Sarto 2021 (Sports Med, doi 10.1007/s40279-021-01436-7) perpendicular, averaged over 3 positions. Host: vertical. | vertical = perpendicular / cos(tilt): +1.5 % at 10°, +6 % at 20°, i.e. 0.2 to 1.5 mm |
| MT inner borders vs centrelines | May 2021 inner borders; DL_Track masks are inner edges after morphology; AnatomyNet subtracts 1.4 mm from centreline distance | about one apo band, 1 to 2 mm = 0.3 to 0.7 tolerance units |
| PA reference | Deep-apo insertion in May, Hagoort, Sarto, DL_Track, host protocol. Lesinski: deep angle 1 to 2° larger than superficial in GM. Bolsterlee 2015: superficial insertion angle under-read by about 10° because probe pressure flattens the superficial apo | horizontal-reference error = deep-apo tilt, 0 to 10°; our own geometry sits 3° below the DL_Track lineage |
| FL direct vs linear extrapolation vs MT/sin(PA) | Ando et al. 2014 (J Electromyogr Kinesiol 24:214): only models combining visible length + linear extrapolation are valid (ICC > 0.75). Franchi et al. 2020 (MSSE 52:233): trig equations overestimate BFlh FL by 0.5 to 1.9 cm, manual linear extrapolation by 0.33 cm (n.s.). Noorkoiv et al. 2010 (J Appl Physiol 109:1974): VL linear extrapolation 12.6 ± 8.1 % shorter than EFOV. Pimenta et al. 2018 (MSSE 50:2584): static images show 35 % of BFlh fascicle. | trig biases FL +5 to +25 %; extrapolation bias small on average (about 3 mm) with muscle-dependent sign |
| Fascicle curvature | Muramatsu et al. 2002 (J Appl Physiol 92:129): GM curvature 0.4 /m rest to 5.2 /m at MVC, straight-line FL under-reads by about 6 % at MVC. Bolsterlee 2018 (PeerJ 6:e4610): soleus 17 /m at short length | a few % at rest; matters in contracted cine frames |
| Probe tilt | Bolsterlee, Gandevia, Herbert 2016 (PLoS ONE 11:e0157273): FL error 0.4 mm per degree of in-plane misalignment (1.1 mm/deg at 20° tilt); PA always over-read off-perpendicular. Bénard et al. 2009 (Muscle Nerve 39:652): 15° off-plane gives up to 14 % FL and 23 % angle error. Klimstra 2007: FL and MT change with orientation, PA did not. | baked into the label, so the estimator cannot correct it; widens the PA/FL noise floor |

### 7.3 Physiological ranges (healthy adults, rest, 2D US)

| Muscle | PA (°) | FL (mm) | MT (mm) | Source |
|---|---|---|---|---|
| GM | 19 (12 to 35 across joint angles) | 50 to 55 (34 to 59) | 13 to 17 | May 2021; Hagoort 2022; Narici 1996 (J Physiol 496:287); Kawakami 1998 |
| GL | 12 | 66 to 69 | 16 to 17 | May 2021 |
| Soleus | 20 to 31 | 30 to 45 | 10 to 25 | Bolsterlee 2018 (DTI); Kawakami 1998 |
| VL | 10 to 14 | 90 to 120 (extrapolated) | 20 to 27 | Hagoort 2022; Lesinski 2024; Aagaard 2001 |
| TA | 10 to 12 | 60 to 65 | 10 to 12 | Hagoort 2022 |

Contraction: GM PA +18° and FL -18 mm from rest to MVC (Narici 1996); TA PA +60 to 70 %,
FL -37 to 40 %, MT unchanged (Maganaris & Baltzopoulos 1999); GL and soleus MT +5 to 7 mm
during contraction (Maganaris et al. 1998). MT and PA co-vary positively within a muscle
(Kawakami 1993 r = 0.88 in triceps; Hagoort r >= 0.54 in VL); FL and PA weakly negative.
Training raises MT and PA (Aagaard 2001: VL PA 8.0 to 10.7°; Kawakami 1995 triceps) and
eccentric training lengthens fascicles (Franchi 2014: +12 % vs +5 %).

### 7.4 Image geometry

Probes in this literature are 38 to 60 mm wide at 3 to 6 cm depth, so on a square-pixel
console at 4 to 5 cm depth with a roughly 700 to 1000 px tall sector the spacing is 0.045
to 0.07 mm/px, and 0.1 to 0.13 mm/px for 512 px exports; this matches the exact scales we
decode on the test set (0.050 to 0.129 mm/px). Scale is always read from the on-screen ruler
in manual practice (Hagoort; DL_Track_US). Long fascicles (VL, BFlh) exceed the window, so
their FL labels are mostly extrapolated and inherit PA error; GM and TA fascicles fit.

## 8. What this means for `muscle_arc`

### 8.1 Where the score is

With tolerances of 6° / 12 mm / 3 mm and inter-rater SEM of about 1.5° / 5 mm / 0.6 mm (GM,
TA) the consensus-label floor is roughly 0.2 to 0.3 UMUD; the public leader sits there.
Our own measurements on the test set (2026-10-12) put our new dual U-Net pipeline within
1.0 mm MT, 7.8 mm FL and 2.9° PA of the public 0.451 anchor. Per parameter, in tolerance
units, that is MT 0.33, FL 0.65, PA 0.48: FL and PA are where the remaining gap is, and both
are convention problems before they are segmentation problems.

### 8.2 Conventions to adopt (each one is backed by a source above)

1. MT: inner-edge to inner-edge, measured at three positions (25/50/75 % of the overlap) as
   vertical lines (host) and averaged; reject pairs outside the muscle (skin, bone). The
   centre-line to inner-edge offset is 1 to 2 mm (Han 2013; May 2021), i.e. 0.3 to 0.7
   tolerance units, so the apo *edge* must be the regression target, not the band centre.
   Our SOF surface heads already predict inner edges; keep that formulation (OCRSeg 2025
   shows boundary-as-1D-function regression is viable at > 100 FPS).
2. PA: angle between the fascicle line and the deep-apo tangent at the fascicle's insertion
   (May 2021, Hagoort 2022, Lesinski 2024, host protocol), never against the image
   horizontal and never against the superficial apo (1 to 2° lower, less reliable). Aggregate
   with an accumulator- or length-weighted median of many fragments (TimTrack, Zhou 2012),
   restrict the search to 5 to 50°, and do not apply DL_Track's 10° floor unless the muscle
   prior says so. Our geometry currently reads 3° below the DL_Track lineage and the official
   DL_Track geometry reads 3° above it; the host's manual angle is between them, so average
   the two estimators (done in `cand_H_paMid`) until a direct check on labelled images exists.
3. FL: visible fragment plus linear extrapolation to both fitted apo curves (Ando 2014 model
   V; Franchi 2020; host protocol), choosing fragments that need the least extrapolation
   (host) and averaging about three. Keep MT / sin(PA) only as a prior for outlier rejection:
   it over-reads FL by 5 to 25 % (Franchi 2020; Ando 2014) and is ill-defined when the apos
   are not parallel (Pohle-Fröhlich 2020). Flag images where > 50 % of the FL is extrapolated
   (TimTrack warning rule) and shrink those toward the muscle prior.
4. Curvature: ignore in GM (< 1.3 % FL, Bolsterlee 2016); matters in soleus, distal VL and
   contracted frames (6 %, Muramatsu 2002; Bizet 2025). A two-segment extrapolation (follow
   the fragment, then bend toward the local apo normal) is the cheap approximation of the
   host's three-segment manual trace.

### 8.3 Architecture decisions

1. Keep segmentation-then-geometry. Every method with sub-tolerance MT (Katakis 2022 0.4 mm,
   MUSA 0.06 to 0.45 mm, DL_Track_US 0.6 mm) is an apo segmenter plus geometry; direct
   regression of PA/FL from pixels generalises poorly across tasks and subjects (Rosa 2021;
   Cunningham 2018 about 6°).
2. Apo branch is the priority model. Train the two-edge surface heads with weighted BCE
   Gaussian heatmaps (DeepMTJ recipe) or the 1D vertical-contour regression (OCRSeg) in
   addition to the mask, so MT does not depend on morphology of a thick mask.
3. Fascicle branch: masks are inherently low-IoU (0.2 to 0.3 even for SegFormer in
   DL_Track_US v0.3); evaluate the fascicle model by downstream PA/FL error, not IoU (our
   Gate1-lite does this). Add a classical orientation ensemble as a parallel estimator:
   Frangi-then-projection-profile with the angle search clamped to 15 to 70°, three best
   sub-regions, plus Radon and structure-tensor medians (Pohle-Fröhlich 2020 reached 1.9° MAE
   this way, at the expert floor). Disagreement between estimators is an uncertainty signal.
4. Exact scale and crop stay deterministic (per-vendor template); a learned tick reader
   (RulerNet recipe) is only a fallback for unseen layouts. Mask or crop console chrome
   before training (Mikolaj 2023): burned-in text is a shortcut feature.
5. Five-frame clips: estimate every frame, then median-fuse with a KLT forward-backward
   consistency weight; expect about a 30 to 50 % reduction of per-frame noise.
6. Device shift: report held-out-device error (DeepMTJ saw 1.4x across devices). The test set
   has five layouts; stratify validation by layout, not randomly.
7. Calibration: the host recommends the 35-image, six-rater OSF set for rater calibration
   (per-parameter gain and offset, muscle-specific priors from section 7.3). Manual labels on
   the test set are allowed if declared; they would pin the PA and FL conventions directly.
8. Backbones: ImageNet U-Nets at 384 to 512 px train in 1 to 3 h per branch on one T4 (our
   measurements); SegFormer-B0/B2 or a USF-MAE-initialised encoder are the only backbone
   experiments with literature support. SAM-family prompting is not worth GPU time for thin
   oriented structures.

### 8.4 Two facts from the test-set description that change priors

- Rectus femoris is a test muscle and is absent from the training set; RF fascicle length is
  the least reliable of all (Ritsche 2022: ICC 0.46, SEM 8.6 mm). Expect a device/muscle
  group in the test set that neither our models nor DL_Track have seen.
- Cerebral-palsy participants are included; CP gastrocnemius has shorter fascicles and
  smaller muscle thickness than healthy controls, so priors clipped to healthy-adult ranges
  will bias those images. Keep clip ranges at the host's stated bounds (5 to 45°, 30 to 200 mm,
  10 to 50 mm), not tighter.

### 8.5 Open questions the literature does not answer

- No published single-image deep method reports FL below about 3 mm against manual labels;
  the best video methods (UltraTimTrack, DUSTrack) reach 2.3 to 2.6 mm only with temporal
  fusion. FL is the parameter least likely to reach its tolerance floor.
- No paper reports how a model trained on cropped B-mode frames degrades on console
  screenshots; our own measurement (public 1.01 for letterboxed DL_Track on full frames vs
  0.29 on cropped OSF frames) is the only quantitative data point.
- Whether the host's two raters follow DL_Track's 10° PA floor or the physiological
  distribution is unknown; it decides whether a floor belongs in post-processing.

## 9. Reference list (first author, year)

Aagaard 2001; Aeles 2017; Ando 2014; Bao 2023; Bénard 2009; Bizet 2025; Blazevich 2006;
Bolsterlee 2015, 2016, 2018; Caresio 2017; Chanti 2021; Chow 2000; Cronin 2011, 2020;
Cunningham 2017, 2018, 2020; Drazan 2019; Farris 2016; Franchi 2014, 2018, 2020; Gillett 2013;
Gionfrida 2024; Hafthorsdottir 2022; Hagoort 2022; Han 2013; Jiao 2024 (USFM); Katakis 2022,
2023; Kawakami 1993, 1995, 1998; Kellis 2009; Klimstra 2007; Kwah 2013; Leitner 2022;
Lesinski 2024; Ling 2013; Liu 2019; Luera 2025; Maganaris 1998, 1999; Marzilger 2018;
Marzola 2021; May 2021; Mikolaj 2023; Muramatsu 2002; Namburete 2011; Namburi 2025;
Narici 1996; Noorkoiv 2010; Pan 2025 (RulerNet); Pimenta 2018; Pohle-Fröhlich 2020;
Rana 2009; Ritsche 2022, 2023, 2024, 2026; Rosa 2021; Sarto 2021; Seynnes 2007, 2020;
van der Zee 2022, 2025; Verheul 2022; Vostrikov 2024; Werkhausen 2023; Zhang 2025 (OCRSeg);
Zhao 2011; Zhou 2008, 2012, 2015. Full citations with DOIs are given inline in sections 2 to 7.
