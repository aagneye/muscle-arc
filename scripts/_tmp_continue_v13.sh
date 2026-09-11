#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate

# Stop slow apo OCR labeling
pkill -f 'build_scale_labels.py' 2>/dev/null || true
pkill -f 'vm_run_scale_climb.sh' 2>/dev/null || true
sleep 2

python - <<'PY'
import pandas as pd
from pathlib import Path
from muscle_arc.data.dataset import list_images

# Fast labels from audit table + OSF
tbl = pd.read_csv("experiments/depth_scale_table.csv")
tbl = tbl[tbl["mm_per_pixel"].notna() & (tbl["confidence"] >= 0.55)].copy()
# map image_id -> path
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

echo "=== Train ScaleStripDetector ==="
python -u scripts/train_scale_detector.py \
  --labels experiments/scale_pseudo_labels.csv \
  --out experiments/checkpoints_scale/scale_detector.pt \
  --epochs 40 \
  --batch-size 16 \
  2>&1 | tee logs/v13_scale_train_scale.log

echo "=== Gate1 ==="
python -u scripts/eval_gate1_geometry.py \
  --config configs/default.yaml \
  --split holdout --split-dir experiments/splits \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate1_geometry.json \
  2>&1 | tee logs/v13_scale_gate1.log || true

echo "=== Infer v13 ==="
python -u scripts/calibrate_predict.py \
  --config configs/default.yaml \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --out submissions/submission_v13_scale.csv \
  --thr 0.30 \
  --temporal-smooth true \
  --sector-crop true \
  --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector.pt \
  --residual-prefer off \
  --debug-csv experiments/submission_v13_scale_debug.csv \
  2>&1 | tee logs/v13_scale_infer.log

echo "=== Gate3 ==="
python -u scripts/eval_gate3_dist.py \
  --cand submissions/submission_v13_scale.csv \
  --ref submissions/submission_v8.csv \
  2>&1 | tee logs/v13_scale_gate3.log || true

python -u scripts/eval_umud_local.py --pred submissions/submission_v13_scale.csv || true

python - <<'PY'
import pandas as pd
d = pd.read_csv("submissions/submission_v13_scale.csv")
r = pd.read_csv("submissions/submission_v8.csv")
print("v13", "FL_med", round(float(d.fl_mm.median()),1), "FL_std", round(float(d.fl_mm.std()),1),
      "clip140", int((d.fl_mm>=139.9).sum()), "MT_med", round(float(d.mt_mm.median()),1))
print("v8 ", "FL_med", round(float(r.fl_mm.median()),1), "FL_std", round(float(r.fl_mm.std()),1),
      "clip140", int((r.fl_mm>=139.9).sum()), "MT_med", round(float(r.mt_mm.median()),1))
print("DONE")
PY
