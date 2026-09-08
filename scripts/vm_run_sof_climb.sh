#!/usr/bin/env bash
# SOF climb: Gate0 → geometry Gate2 → align retrain / SOF → scale → gated submit.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
mkdir -p logs experiments/checkpoints_sof experiments/checkpoints_scale submissions

CFG="${CFG:-configs/sof.yaml}"
APO="${APO:-experiments/checkpoints_phase4/apo_best.pt}"
FASC="${FASC:-experiments/checkpoints_phase4/fasc_best.pt}"
TAG="${TAG:-sof_v1}"
AUTO_SUBMIT="${AUTO_SUBMIT:-0}"
SKIP_SOF_TRAIN="${SKIP_SOF_TRAIN:-0}"

echo "=== Gate0: DL_Track + our geometry + GT scale ==="
python -u scripts/download_dl_track.py --out-dir data/external/dl_track \
  2>&1 | tee "logs/${TAG}_dl_track_dl.log" || true
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track --scale-mode gt \
  --sector-crop false \
  --out experiments/gate0_reference.json \
  --gate0-out experiments/gate0_reference.json \
  2>&1 | tee "logs/${TAG}_gate0.log" || true

echo "=== Gate2 ours + GT scale (geometry exam) ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --masks-from ours --scale-mode gt \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate2_osf_umud.json \
  2>&1 | tee "logs/${TAG}_gate2_gt.log" || true

echo "=== Gate2b ours + predicted scale ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --masks-from ours --scale-mode pred \
  --out experiments/gate2b_osf_pred_scale.json \
  2>&1 | tee "logs/${TAG}_gate2b.log" || true

if [[ "$SKIP_SOF_TRAIN" != "1" ]]; then
  echo "=== Train SOF multitask ==="
  python -u scripts/train_multitask.py --config "$CFG" \
    2>&1 | tee "logs/${TAG}_sof_train.log" || true
fi

echo "=== Native scale detector ==="
python - <<'PY'
import pandas as pd
from pathlib import Path
from muscle_arc.data.dataset import list_images
tbl = pd.read_csv("experiments/depth_scale_table.csv") if Path("experiments/depth_scale_table.csv").exists() else None
rows = []
if tbl is not None:
    tbl = tbl[tbl["mm_per_pixel"].notna() & (tbl["confidence"] >= 0.55)]
    imgs = {p.name: p for p in list_images(Path("data/raw/test_images_v2"))}
    for _, r in tbl.iterrows():
        p = imgs.get(str(r["image_id"]))
        if p is None: continue
        rows.append({"path": str(p), "mm_per_pixel": float(r["mm_per_pixel"]), "confidence": float(r["confidence"])})
osf = Path("experiments/osf_expert_gt.csv")
if osf.exists():
    df = pd.read_csv(osf)
    for _, r in df.iterrows():
        p = Path(str(r.get("path","")))
        mm = r.get("mm_per_pixel")
        if not p.exists(): continue
        if pd.isna(mm) and pd.notna(r.get("scale_pixel_per_cm")):
            mm = 10.0 / float(r["scale_pixel_per_cm"])
        if pd.isna(mm): continue
        rows.append({"path": str(p), "mm_per_pixel": float(mm), "confidence": 1.0})
pd.DataFrame(rows).drop_duplicates("path").to_csv("experiments/scale_pseudo_labels.csv", index=False)
print("labels", len(rows))
PY
python -u scripts/train_scale_detector.py \
  --labels experiments/scale_pseudo_labels.csv \
  --kind native \
  --out experiments/checkpoints_scale/scale_detector_native.pt \
  --epochs 30 \
  2>&1 | tee "logs/${TAG}_scale_native.log" || true

echo "=== Infer ${TAG} ==="
python -u scripts/calibrate_predict.py \
  --config configs/default.yaml \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --out "submissions/submission_${TAG}.csv" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true --sector-crop true --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector_native.pt \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  2>&1 | tee "logs/${TAG}_infer.log"

python -u scripts/eval_gate3_dist.py \
  --cand "submissions/submission_${TAG}.csv" \
  --ref submissions/submission_v8.csv \
  --out experiments/gate3_dist.json \
  2>&1 | tee "logs/${TAG}_gate3.log" || true

python -u scripts/eval_gate4_lodo.py \
  --gate2-csv experiments/gate2_osf_umud.csv \
  --out experiments/gate4_lodo.json \
  2>&1 | tee "logs/${TAG}_gate4.log" || true

python -u scripts/grow_report_card.py \
  --gate0 experiments/gate0_reference.json \
  --gate1 experiments/gate1_geometry.json \
  --gate2 experiments/gate2_osf_umud.json \
  --gate2b experiments/gate2b_osf_pred_scale.json \
  --gate3 experiments/gate3_dist.json \
  --out "experiments/${TAG}_report_card.json" \
  2>&1 | tee "logs/${TAG}_card.log" || true

DECISION="$(python -c "import json; print(json.load(open('experiments/${TAG}_report_card.json')).get('submit','HOLD'))" 2>/dev/null || echo HOLD)"
echo "Decision=${DECISION}"
if [[ "$DECISION" == "SUBMIT" && "$AUTO_SUBMIT" == "1" ]]; then
  bash scripts/vm_submit_kaggle.sh "submissions/submission_${TAG}.csv" "${TAG}-SOF-climb-gated"
fi
echo "=== DONE ${TAG} ==="
