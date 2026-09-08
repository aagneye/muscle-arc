#!/usr/bin/env bash
# Phase-4: letterbox + mild Tversky fasc retrain, then apo letterbox retrain, then v8 infer.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q

mkdir -p logs experiments/checkpoints_phase4 submissions

# --- fasc letterbox + tversky (warm start) ---
python3 - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
cfg["train"]["epochs"] = 30
cfg["train"]["batch_size"] = 4
cfg["train"]["lr"] = 2.0e-4
cfg["train"]["checkpoint_dir"] = "experiments/checkpoints_phase4"
Path("configs/phase4_fasc.yaml").write_text(yaml.dump(cfg))
print("wrote configs/phase4_fasc.yaml")
PY

echo "=== train fasc phase4 (GPU0) ==="
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py \
  --config configs/phase4_fasc.yaml \
  --branch fasc \
  --loss tversky \
  --warm-start experiments/checkpoints/fasc_best.pt \
  2>&1 | tee logs/train_fasc_phase4.log

echo "=== train apo phase4 letterbox (GPU0) ==="
python3 - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
cfg["train"]["epochs"] = 25
cfg["train"]["batch_size"] = 4
cfg["train"]["lr"] = 2.5e-4
cfg["train"]["checkpoint_dir"] = "experiments/checkpoints_phase4"
Path("configs/phase4_apo.yaml").write_text(yaml.dump(cfg))
print("wrote configs/phase4_apo.yaml")
PY

python scripts/train.py \
  --config configs/phase4_apo.yaml \
  --branch apo \
  --loss dice \
  --warm-start experiments/checkpoints/apo_best.pt \
  2>&1 | tee logs/train_apo_phase4.log

echo "=== geometry gate A ==="
python scripts/eval_geometry.py --max-samples 100 || true

echo "=== inference v8 ==="
python scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --out submissions/submission_v8.csv \
  --thr 0.30 \
  --temporal-smooth true \
  2>&1 | tee logs/infer_v8.log

echo "=== compare v7 vs v8 ==="
python3 - <<'PY'
import pandas as pd
from pathlib import Path
for name in ["submission_v7.csv", "submission_v8.csv", "submission_v3.csv"]:
    p = Path("submissions")/name
    if not p.exists():
        print(name, "missing"); continue
    d = pd.read_csv(p)
    print(name, "PA med", float(d.pa_deg.median()), "mean", float(d.pa_deg.mean()),
          "FL med", float(d.fl_mm.median()), "FL>=139", int((d.fl_mm>=139).sum()),
          "MT med", float(d.mt_mm.median()))
PY
