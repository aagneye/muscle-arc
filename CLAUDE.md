# muscle-arc: operating instructions (AI software factory mode)

## The goal

Public UMUD score **below 0.23** on the Kaggle UMUD Challenge (lower is better; top 10 is
about 0.22 as of 2026-10-22). Final deadline 2026-11-14 23:00 UTC. Work toward this goal
until it is reached; do not stop at intermediate wins.

## How to work (owner's standing instructions)

1. **Before building anything, read the current state:** what exists now, what was tried
   before and how it scored, and how far the goal is.
   - `docs/curator_climb.md` - every submission, its score and the lesson
   - `docs/competition_intelligence.md` - leaders, host rulings, discussion digest, paths
   - `docs/literature_review.md` - methods literature and design decisions
   - `docs/superpowers/plans/` - active plans
2. **Pick the work with the highest expected gain toward the goal**, based on evidence and
   research, not habit. Research (papers, discussions, notebooks) before committing to a
   direction. Write down why.
3. **Evaluate locally before Kaggle.** The official metric is published
   (`src/muscle_arc/geometry/umud_metric.py`, host notebook `paulritsche/umud-score`):
   S = (MAE_PA/6 + MAE_FL/12 + MAE_MT/3) / 3. Score candidates with
   `scripts/eval_local.py` against every available label set before spending a Kaggle
   submission. Kaggle submissions (5/day) confirm, they do not steer.
4. **Train on the free GPUs of the Azure VM** (`rogii-gpu`, 4x T4). Never stop or touch
   another project's processes; queue behind them (`scripts/vm_train_folds.sh` waits for
   free memory).
5. **Record every result** (local and public) in `docs/curator_climb.md` with the lesson.
6. Atomic commits: one logical change, `<type>(<file>): <description>`.

## Hard guardrails

- **The GitHub repo is public.** Never commit test images, manual measurements of test
  images, or anything under `data/`. `data/labeling/` is git-ignored and must stay so.
- Final pipeline must re-run on new data without any leaderboard values (host ruling,
  discussion 739660). Manual test measurements may be used only as declared external data
  for calibration/validation (host rulings 690868, 737782), never pasted as predictions.
- Do not repeat known losers (see `docs/curator_climb.md`): OSF-selected geometry
  conventions, global OSF residual blends, stretch-resizing for DL_Track.
