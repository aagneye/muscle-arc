#!/usr/bin/env bash
# Finish win build after Gate2b: infer without broken scale-detector pickle.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate

TAG=dltrack_win
OUT=submissions/submission_${TAG}.csv

# Wait for Gate2b if still running
while pgrep -f 'eval_osf_pipeline.py.*gate2b_dltrack_win' >/dev/null; do
  echo "waiting Gate2b..."
  sleep 20
done

python3 -c "import json; m=json.load(open('experiments/gate2b_dltrack_win.json')); print('Gate2b', m.get('umud')); open('experiments/dltrack_win_gate2b.txt','w').write(str(m.get('umud'))+'\n')"

echo "=== Test infer (no scale-detector — OCR table + group share) ==="
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
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  2>&1 | tee logs/${TAG}_infer.log

python -u scripts/eval_gate3_dist.py \
  --cand "$OUT" --ref submissions/submission_v8.csv \
  --out experiments/${TAG}_gate3.json || true

python3 <<'PY'
import json
from pathlib import Path
import pandas as pd
tag="dltrack_win"
g2=json.load(open(f"experiments/gate2b_{tag}.json"))
g3=json.load(open(f"experiments/{tag}_gate3.json")) if Path(f"experiments/{tag}_gate3.json").exists() else {}
s=pd.read_csv(f"submissions/submission_{tag}.csv")
umud=float(g2.get("umud",9))
fl=float(s.fl_mm.median()); mt=float(s.mt_mm.median()); pa=float(s.pa_deg.median())
fl_std=float(s.fl_mm.std())
g3_ok=bool(g3.get("gate3_ok") or g3.get("ok"))
ok = umud<=0.42 and g3_ok and (70<=fl<=95) and (15<=mt<=28) and fl_std>5
soft = umud<=0.42 and (70<=fl<=95) and fl_std>5
print({"gate2b":umud,"gate3_ok":g3_ok,"fl":fl,"mt":mt,"pa":pa,"fl_std":fl_std,"SUBMIT":ok,"SOFT":soft})
Path(f"experiments/{tag}_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
Path(f"experiments/{tag}_soft_submit.txt").write_text("SUBMIT\n" if soft else "HOLD\n")
PY

# coverage
python3 <<'PY'
import json
from pathlib import Path
import yaml
from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry.sector_crop import mask_chrome, classify_kind, find_sector
cfg=yaml.safe_load(Path("configs/default.yaml").read_text())
paths=DataPaths.from_config(cfg["data"])
n=n_con=n_mask=0
for p in list_images(paths.test_images):
    g=read_gray(p); n+=1
    kind,_=classify_kind(g, find_sector(g))
    if kind=="console": n_con+=1
    if mask_chrome(g)[1].applied: n_mask+=1
print({"n":n,"console":n_con,"chrome_masked":n_mask})
Path("experiments/chrome_mask_coverage.json").write_text(json.dumps({"n":n,"console":n_con,"chrome_masked":n_mask},indent=2))
PY

echo DONE_WIN_FINISH
