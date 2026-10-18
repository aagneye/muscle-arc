# Top-10 strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Public score <= 0.24 (top 10) by 2026-11-14 with a GPL-3.0 reproducible pipeline.

**Architecture:** Two parallel lines. Line A (automated, no decision needed): grow the
ensemble and add an apo-edge head. Line B (needs your go): a labelling tool that follows the
host protocol exactly, your measurements on the 309 test images as a declared external file,
and a calibration layer fitted on them. Line C (merge with the leader) is a message, not code.

**Tech Stack:** PyTorch/smp on rogii-gpu; Tkinter or browser-based labelling tool (zero
install for you); NumPy calibration; Kaggle API via `scripts/vm_submit_kaggle.sh`.

**Spec:** `docs/competition_intelligence.md` sections 5 and 6.

## Global Constraints

- Final code must regenerate the submission from training data plus declared external files, with no leaderboard values (host ruling 739660).
- Any test-image measurements are declared as external data in the method description (host ruling 690868, 737782).
- 5 submissions per day; keep 2 per day in reserve for Line B checks.
- Clip to host ranges PA 5-45, FL 30-200, MT 10-50.

## Review Focus

1. A labelled image where the labeller measured a different compartment than the raters (TA central apo) -> expect the calibration to down-weight outliers (Huber), not chase them.
2. Device groups with few labels (853x1069: 11 images, 512-ish: 9) -> expect a shared global map with a shrunk per-device offset, never a per-device fit alone.
3. A test image our pipeline fails on (NaN) -> expect the clip-median fallback, then the muscle prior, never the label itself leaking into the output.
4. Calibration fitted on all 309 and evaluated on the same 309 -> expect leave-group-out (by clip) numbers reported alongside, so public-board gains are believable.
5. Submission built with a label file missing from the repo -> expect the build script to fail loudly, not to silently use uncalibrated outputs.

---

### Task 1 (Line A, now): second ensemble seed

**Files:** Create `configs/dual_v3_s2.yaml` (seed 7, same recipe); reuse `scripts/vm_train_folds.sh` with `CFG`, `SPLITS_ROOT=experiments/splits_k5_s2`, `CKPT_ROOT=experiments/checkpoints_dual_v3_k5_s2`.

- [ ] Step 1: commit the config.
- [ ] Step 2: launch the queue when >= 2 GPUs are free; expect 10 jobs, about 5 h.
- [ ] Step 3: cache 10-model probabilities with `scripts/cache_probs.py` (both seeds) and build the candidate with the 0.4427 geometry via `scripts/geometry_sweep.py --pick "pa_legacy+2" "fl_legacy|trig0" "mt_legacy"`; submit once. Expected 0.40 to 0.41.

### Task 2 (Line B, needs your go): host-protocol labelling tool

**Files:** Create `scripts/label_tool.py` (Tkinter, no extra dependencies; loads the exact
scale table so lengths are in mm), Create `tests/test_label_geometry.py`,
Create `data/external/manual_labels/README.md` (declaration text).

**Interfaces:**
- Produces `data/external/manual_labels/labels.csv` with columns `image_id, pa_deg, fl_mm, mt_mm, n_mt, n_fl, n_pa, labeller, seconds`, one row per image; raw click coordinates in `labels_raw.jsonl`.
- Protocol enforced by the tool: scale from the exact table; 3 vertical MT lines (left, middle, right) clicked top-to-bottom; 3 fascicles drawn as up to 3 connected segments and extended to the apo lines; 3 pennation angles as two segments (along fascicle, along deep apo); means written.

- [ ] Step 1: test `test_segmented_fascicle_length_mm` (3 segments of known px lengths at 0.1 mm/px -> mm sum) and `test_pennation_from_two_segments` (segments at 20° and 0° -> 20.0).
- [ ] Step 2: run, expect FAIL (module missing).
- [ ] Step 3: implement geometry helpers in `src/muscle_arc/geometry/manual.py` (`segmented_length_mm`, `angle_between_segments_deg`, `vertical_gap_mm`) and the tool.
- [ ] Step 4: run tests PASS; you label 20 images to check the pace (target <= 90 s/image, so 309 images in about 8 h over 2 days).
- [ ] Step 5: commit.

### Task 3 (Line B): calibration on manual labels

**Files:** Modify `src/muscle_arc/geometry/calibration.py` (add `fit_device_offsets(pred, gt, device, shrink)` with Huber loss and device shrinkage), Create `scripts/calibrate_to_labels.py`, Test `tests/test_calibration.py`.

**Interfaces:**
- Consumes `labels.csv` and the pipeline's uncalibrated per-image outputs (`experiments/geometry_sweep_<tag>_test_measures.csv`).
- Produces `experiments/calibration_<tag>.json` (global gain/offset per parameter + per-device offsets) and the calibrated submission; prints leave-clip-out MAE per parameter in tolerance units.

- [ ] Step 1: tests: a synthetic set with a +2° global PA bias and one device at +1 mm MT -> global and device terms recovered within 10 %; leave-clip-out MAE improves.
- [ ] Steps 2-4: fail, implement, pass.
- [ ] Step 5: build and submit; expected 0.28 to 0.33 on the first pass if labels are consistent; iterate on labelling consistency (not on the board) if leave-clip-out MAE and public score disagree.

### Task 4 (Line B): repository deliverable

- [ ] `README.md` section "External data": the declared label file, who measured it, protocol, and that the final run needs only train data + that file.
- [ ] One command `python scripts/reproduce.py` that trains (or loads committed checkpoints), caches, calibrates and writes `submission.csv`.

### Task 5 (Line C, optional): merger

- [ ] Reply to thread 749906 only if you decide to; before that, ask the leader for (a) their public/private validation method and (b) whether their code is reproducible under GPL-3.0. Merge only after seeing both.
