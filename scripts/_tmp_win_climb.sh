#!/usr/bin/env bash
# Win path: letterbox DL_Track + hardened chrome-mask + our geometry + OCR scales.
# Official doCalculations is Phase A (separate CSV); Gate2b bar uses our geom (≤0.42).
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q

TAG=dltrack_win
OUT=submissions/submission_${TAG}.csv
mkdir -p logs experiments submissions

echo "=== Gate2b: DL_Track + OUR geom + pred scale (shape>ticks) ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track \
  --geom ours \
  --sector-crop false \
  --chrome-mask false \
  --scale-mode pred \
  --max-images 35 \
  --out experiments/gate2b_${TAG}.json \
  2>&1 | tee logs/gate2b_${TAG}.log || true

GATE2B=$(python3 -c "import json; print(json.load(open('experiments/gate2b_${TAG}.json')).get('umud', 9))")
echo "Gate2b UMUD=${GATE2B}"

echo "=== Test infer (calibrate_predict letterbox + chrome-mask + our geom) ==="
python -u scripts/calibrate_predict.py \
  --config configs/default.yaml \
  --masks-from dl_track \
  --geom ours \
  --dl-track-dir data/external/dl_track \
  --out "$OUT" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true --sector-crop false --chrome-mask true \
  --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector_native.pt \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  2>&1 | tee logs/${TAG}_infer.log

python -u scripts/eval_gate3_dist.py \
  --cand "$OUT" --ref submissions/submission_v8.csv \
  --out experiments/${TAG}_gate3.json || true

python - <<'PY'
import json
from pathlib import Path
import pandas as pd

tag = "dltrack_win"
g2 = json.load(open(f"experiments/gate2b_{tag}.json"))
g3 = json.load(open(f"experiments/{tag}_gate3.json")) if Path(f"experiments/{tag}_gate3.json").exists() else {}
s = pd.read_csv(f"submissions/submission_{tag}.csv")
umud = float(g2.get("umud", 9))
fl = float(s.fl_mm.median())
mt = float(s.mt_mm.median())
pa = float(s.pa_deg.median())
fl_std = float(s.fl_mm.std())
g3_ok = bool(g3.get("gate3_ok") or g3.get("ok"))
# Plan: Gate2b ≤0.42 and Gate3 green; FL med ~70–95
ok = umud <= 0.42 and g3_ok and (70 <= fl <= 95) and (15 <= mt <= 28) and fl_std > 5
print({
    "gate2b": umud,
    "gate3_ok": g3_ok,
    "fl_med": fl,
    "mt_med": mt,
    "pa_med": pa,
    "fl_std": fl_std,
    "SUBMIT": ok,
})
Path(f"experiments/{tag}_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
Path(f"experiments/{tag}_gate2b.txt").write_text(f"{umud}\n")
# Soft public mid-checkpoint candidate even if Gate3 soft — still useful for ≤0.50
soft = umud <= 0.42 and (70 <= fl <= 95) and fl_std > 5
Path(f"experiments/{tag}_soft_submit.txt").write_text("SUBMIT\n" if soft else "HOLD\n")
PY

echo "=== Apo-pair overlays (20+20) ==="
python -u scripts/debug_apo_pair_overlays.py --n-each 20 --chrome-mask true \
  --out-dir experiments/overlays_apo_pair \
  2>&1 | tee logs/${TAG}_overlays.log | tail -5 || true

echo "=== Chrome-mask coverage on test ==="
python - <<'PY'
from pathlib import Path
import yaml
from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry.sector_crop import mask_chrome, classify_kind, find_sector
cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
paths = DataPaths.from_config(cfg["data"])
n_con = n_mask = n = 0
for p in list_images(paths.test_images):
    g = read_gray(p)
    kind, chrome = classify_kind(g, find_sector(g))
    n += 1
    if kind == "console":
        n_con += 1
    m, info = mask_chrome(g)
    if info.applied:
        n_mask += 1
print({"n": n, "console": n_con, "chrome_masked": n_mask})
Path("experiments/chrome_mask_coverage.json").write_text(
    __import__("json").dumps({"n": n, "console": n_con, "chrome_masked": n_mask}, indent=2)
)
PY

echo DONE_WIN_BUILD
