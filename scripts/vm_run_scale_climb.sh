#!/usr/bin/env bash
# Phase A climb: duty-cycle OCR/ticks → train strip CNN → infer → Gate2/3 → submit.
# Rollback: v8 / v10c. Residual OSF mm blend stays OFF.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
mkdir -p logs experiments/checkpoints_scale submissions

CFG="${CFG:-configs/default.yaml}"
APO="${APO:-experiments/checkpoints_phase4/apo_best.pt}"
FASC="${FASC:-experiments/checkpoints_phase4/fasc_best.pt}"
TAG="${TAG:-v14_scale}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
AUTO_SUBMIT="${AUTO_SUBMIT:-1}"

echo "=== 0) Preconditions ==="
test -f "$APO" && test -f "$FASC"
test -f submissions/submission_v8.csv
test -f ~/.kaggle/access_token && echo KAGGLE_OK

if [[ "$SKIP_AUDIT" != "1" ]]; then
  echo "=== 1) Audit depth/OCR/ticks (duty-cycle) ==="
  python -u scripts/audit_depth_scale.py \
    --folders test_images_v2 \
    --out experiments/depth_scale_table.csv \
    2>&1 | tee "logs/${TAG}_audit.log"
else
  echo "=== 1) SKIP audit (using existing depth_scale_table.csv) ==="
fi

echo "=== 2) Build scale pseudo-labels (fast: table + OSF) ==="
python - <<'PY'
import pandas as pd
from pathlib import Path
from muscle_arc.data.dataset import list_images

tbl = pd.read_csv("experiments/depth_scale_table.csv")
tbl = tbl[tbl["mm_per_pixel"].notna() & (tbl["confidence"] >= 0.55)].copy()
imgs = {p.name: p for p in list_images(Path("data/raw/test_images_v2"))}
rows = []
for _, r in tbl.iterrows():
    p = imgs.get(str(r["image_id"]))
    if p is None:
        continue
    rows.append({
        "image_id": r["image_id"],
        "path": str(p),
        "mm_per_pixel": float(r["mm_per_pixel"]),
        "source": str(r["source"]),
        "confidence": float(r["confidence"]),
        "kind": str(r.get("kind", "")),
    })
osf = Path("experiments/osf_expert_gt.csv")
if osf.exists():
    df = pd.read_csv(osf)
    for _, r in df.iterrows():
        p = Path(str(r.get("path", "")))
        mm = r.get("mm_per_pixel")
        if not p.exists():
            continue
        if pd.isna(mm) and "scale_pixel_per_cm" in df.columns and pd.notna(r.get("scale_pixel_per_cm")):
            mm = 10.0 / float(r["scale_pixel_per_cm"])
        if pd.isna(mm):
            continue
        rows.append({
            "image_id": p.name,
            "path": str(p),
            "mm_per_pixel": float(mm),
            "source": "osf_gt",
            "confidence": 1.0,
            "kind": "osf",
        })
out = pd.DataFrame(rows).drop_duplicates(subset=["path"], keep="last")
out.to_csv("experiments/scale_pseudo_labels.csv", index=False)
print(f"Wrote scale_pseudo_labels.csv n={len(out)}")
print(out["source"].value_counts().to_string())
PY

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "=== 3) Train ScaleStripDetector ==="
  python -u scripts/train_scale_detector.py \
    --labels experiments/scale_pseudo_labels.csv \
    --out experiments/checkpoints_scale/scale_detector.pt \
    --epochs 40 \
    --batch-size 16 \
    2>&1 | tee "logs/${TAG}_train_scale.log"
else
  echo "=== 3) SKIP train (using existing scale_detector.pt) ==="
  test -f experiments/checkpoints_scale/scale_detector.pt
fi

echo "=== 4) Gate1 geometry (phase4, informative) ==="
python -u scripts/eval_gate1_geometry.py \
  --config "$CFG" \
  --split holdout --split-dir experiments/splits \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate1_geometry.json \
  2>&1 | tee "logs/${TAG}_gate1.log" || true

echo "=== 5) Gate2 OSF end-to-end ==="
python -u scripts/eval_osf_pipeline.py \
  --config "$CFG" \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate2_osf_umud.json \
  2>&1 | tee "logs/${TAG}_gate2.log" || true

echo "=== 6) Infer ${TAG} (OCR+ticks+group+CNN+shape, residual off) ==="
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -u scripts/calibrate_predict.py \
  --config "$CFG" \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --out "submissions/submission_${TAG}.csv" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true \
  --sector-crop true \
  --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector.pt \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  --debug-csv "experiments/submission_${TAG}_debug.csv" \
  2>&1 | tee "logs/${TAG}_infer.log"

echo "=== 7) Gate3 dist vs v8 ==="
python -u scripts/eval_gate3_dist.py \
  --cand "submissions/submission_${TAG}.csv" \
  --ref submissions/submission_v8.csv \
  --out experiments/gate3_dist.json \
  2>&1 | tee "logs/${TAG}_gate3.log" || true

python -u scripts/eval_umud_local.py --pred "submissions/submission_${TAG}.csv" || true

python -u scripts/grow_report_card.py \
  --gate1 experiments/gate1_geometry.json \
  --gate2 experiments/gate2_osf_umud.json \
  --gate3 experiments/gate3_dist.json \
  --out "experiments/${TAG}_report_card.json" \
  2>&1 | tee "logs/${TAG}_report_card.log" || true

python3 - <<PY
import json
from pathlib import Path
import pandas as pd

tag = "${TAG}"
cand = pd.read_csv(f"submissions/submission_{tag}.csv")
ref = pd.read_csv("submissions/submission_v8.csv")
g2 = json.loads(Path("experiments/gate2_osf_umud.json").read_text()) if Path("experiments/gate2_osf_umud.json").exists() else {}
g3 = json.loads(Path("experiments/gate3_dist.json").read_text()) if Path("experiments/gate3_dist.json").exists() else {}

fl_med = float(cand.fl_mm.median())
fl_std = float(cand.fl_mm.std())
clip140 = int((cand.fl_mm >= 139.9).sum())
mt_med = float(cand.mt_mm.median())
print(tag, "n", len(cand), "FL_med", round(fl_med,1), "FL_std", round(fl_std,1),
      "clip140", clip140, "MT_med", round(mt_med,1))
print("v8 ", "FL_med", round(float(ref.fl_mm.median()),1), "FL_std", round(float(ref.fl_mm.std()),1))

# Scale-climb submit bar (anti v9/v10b):
# Gate3 hard (std/clip/FL~81) AND (Gate2 OK OR soft OSF umud<0.70 with few MT outliers)
gate3_ok = bool(g3.get("gate3_ok") or g3.get("gate4_ok"))
gate2_ok = bool(g2.get("gate2_ok"))
gate2_soft = (float(g2.get("umud", 9)) < 0.70) and (int(g2.get("n_mt_bad", 99)) <= 8)
fl_ok = 70.0 <= fl_med <= 95.0
clip_ok = clip140 <= 20
submit = bool(gate3_ok and fl_ok and clip_ok and (gate2_ok or gate2_soft))

# If Gate2 missing but Gate3+dist look good, still allow submit for scale climb
if not g2 and gate3_ok and fl_ok and clip_ok:
    submit = True

decision = "SUBMIT" if submit else "HOLD"
Path(f"experiments/{tag}_submit_decision.txt").write_text(decision + "\n")
print(
    "SUBMIT_DECISION", decision,
    "gate3", gate3_ok, "gate2", gate2_ok, "gate2_soft", gate2_soft,
    "umud", g2.get("umud"), "mt_bad", g2.get("n_mt_bad"),
)
PY

DECISION="$(tr -d '\n' < "experiments/${TAG}_submit_decision.txt")"
echo "Decision=${DECISION} AUTO_SUBMIT=${AUTO_SUBMIT}"

if [[ "$DECISION" == "SUBMIT" && "$AUTO_SUBMIT" == "1" ]]; then
  echo "=== 8) Kaggle submit ${TAG} ==="
  bash scripts/vm_submit_kaggle.sh \
    "submissions/submission_${TAG}.csv" \
    "${TAG}: duty-cycle OCR/ticks + group share + scale CNN; residual off"
else
  echo "=== 8) HOLD — not submitting (decision=${DECISION}) ==="
fi

echo "=== DONE ${TAG} ==="
ls -la "submissions/submission_${TAG}.csv" experiments/depth_scale_table.csv \
  experiments/checkpoints_scale/scale_detector.pt \
  "experiments/${TAG}_submit_decision.txt" 2>/dev/null || true
