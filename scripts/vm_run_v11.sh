#!/usr/bin/env bash
# v11 gated climb: audit → gates → infer → Gate4 → conditional submit
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
pip install pytesseract openpyxl requests -q || true
sudo apt-get install -y tesseract-ocr >/dev/null 2>&1 || true
mkdir -p logs experiments submissions

APO=experiments/checkpoints_phase4/apo_best.pt
FASC=experiments/checkpoints_phase4/fasc_best.pt
APO2=""
FASC2=""
if [[ -f experiments/checkpoints_seed2/fasc_best.pt ]]; then
  FASC2="--fasc-ckpt2 experiments/checkpoints_seed2/fasc_best.pt"
  echo "Using fasc seed2 ensemble"
fi
if [[ -f experiments/checkpoints_seed2/apo_best.pt ]]; then
  APO2="--apo-ckpt2 experiments/checkpoints_seed2/apo_best.pt"
fi

echo "=== Phase0 Gate1 (geometry) ==="
python -u scripts/eval_gate1_geometry.py \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --max-samples 160 \
  --out experiments/gate1_geometry.json \
  2>&1 | tee logs/gate1_v11.log || true
ROUTE=$(python3 -c "import json; print(json.load(open('experiments/gate1_geometry.json')).get('route','unknown'))" 2>/dev/null || echo unknown)
echo "GATE1_ROUTE=$ROUTE"

echo "=== Phase0 Gate2 (OSF e2e) ==="
python -u scripts/eval_osf_pipeline.py \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate2_osf_umud.json \
  2>&1 | tee logs/gate2_v11.log || true

echo "=== Phase3 optional DL_Track listing ==="
python scripts/download_dl_track.py 2>&1 | tee logs/dl_track_v11.log || true

echo "=== re-audit depth OCR/ticks ==="
python -u scripts/audit_depth_scale.py \
  --out experiments/depth_scale_table.csv \
  2>&1 | tee logs/depth_scale_audit_v11.log

echo "=== v11 infer ==="
export CUDA_VISIBLE_DEVICES=0
# shellcheck disable=SC2086
python -u scripts/calibrate_predict.py \
  --apo-ckpt "$APO" \
  --fasc-ckpt "$FASC" \
  $APO2 $FASC2 \
  --out submissions/submission_v11.csv \
  --apo-thr 0.35 \
  --fasc-thr 0.10 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  --depth-scale-table experiments/depth_scale_table.csv \
  --sector-crop true \
  --live-depth-scale true \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  --debug-csv experiments/submission_v11_debug.csv \
  2>&1 | tee logs/infer_v11.log

echo "=== local eval + Gate4 ==="
python scripts/eval_umud_local.py --pred submissions/submission_v8.csv --save-baseline || true
python scripts/eval_umud_local.py --pred submissions/submission_v11.csv || true
python scripts/eval_umud_local.py --pred submissions/submission_v10c.csv || true
python scripts/eval_gate4_dist.py \
  --cand submissions/submission_v11.csv \
  --ref submissions/submission_v8.csv \
  --out experiments/gate4_v11.json || true

python3 - <<'PY'
import json
from pathlib import Path
g1 = json.loads(Path("experiments/gate1_geometry.json").read_text()) if Path("experiments/gate1_geometry.json").exists() else {}
g2 = json.loads(Path("experiments/gate2_osf_umud.json").read_text()) if Path("experiments/gate2_osf_umud.json").exists() else {}
g4 = json.loads(Path("experiments/gate4_v11.json").read_text()) if Path("experiments/gate4_v11.json").exists() else {}
depth_ok = False
log = Path("logs/depth_scale_audit_v11.log")
if log.exists():
    t = log.read_text()
    depth_ok = "DEPTH_SCALE_OK" in t or "with_scale=" in t
print("SUMMARY", {"gate1_route": g1.get("route"), "gate1_geom_ok": g1.get("geom_ok"),
                  "gate2_ok": g2.get("gate2_ok"), "gate2_umud": g2.get("umud"),
                  "gate4_ok": g4.get("gate4_ok"), "gate4": g4})
# Submit gate: Gate4 required; Gate2 preferred; never submit on n=2 alone
submit = bool(g4.get("gate4_ok"))
# Soft: if gate2 exists and failed badly (>0.70), block
if g2.get("umud") is not None and g2["umud"] > 0.70:
    submit = False
    print("BLOCK: Gate2 UMUD too high")
Path("experiments/v11_submit_decision.txt").write_text("SUBMIT\n" if submit else "HOLD\n")
print("DECISION", "SUBMIT" if submit else "HOLD")
PY

if grep -q SUBMIT experiments/v11_submit_decision.txt; then
  echo "=== submitting v11 ==="
  bash /tmp/vm_submit_v8.sh submissions/submission_v11.csv \
    "v11 gated: sector+OCR/ticks+5frame+splitThr+apoPair+edgeFL; residual off" \
    2>&1 | tee logs/submit_v11.log || true
else
  echo "=== HOLD submit (gates failed) — keeping v8/v10c rollback ==="
fi
echo DONE
