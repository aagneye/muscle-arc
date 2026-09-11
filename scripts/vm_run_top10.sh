#!/usr/bin/env bash
# Top-10 climb: scale clusters + local gates + 2nd fasc seed ensemble + v9/v10/v11 submits
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
mkdir -p logs experiments submissions data/external

echo "=== re-audit scales (reject 72dpi) ==="
python scripts/audit_tiff_scales.py --out experiments/scale_table.csv | tee logs/scale_audit2.log
python3 - <<'PY'
from muscle_arc.geometry.scale import load_scale_table
df=load_scale_table('experiments/scale_table.csv')
print('usable rows', int(df['scale_usable'].sum()) if len(df) else 0, '/', len(df))
PY

echo "=== download OSF expert benchmarks (best effort) ==="
python scripts/download_osf_benchmarks.py 2>&1 | tee logs/osf_download.log | tail -40 || true

echo "=== train fasc seed2 for ensemble (GPU0, 18 epochs) ==="
python3 - <<'PY'
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path('configs/default.yaml').read_text())
cfg['seed']=123
cfg['train']['epochs']=18
cfg['train']['lr']=2.0e-4
cfg['train']['checkpoint_dir']='experiments/checkpoints_seed2'
Path('configs/seed2_fasc.yaml').write_text(yaml.dump(cfg))
print('wrote configs/seed2_fasc.yaml')
PY
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py \
  --config configs/seed2_fasc.yaml \
  --branch fasc \
  --loss tversky \
  --warm-start experiments/checkpoints_phase4/fasc_best.pt \
  2>&1 | tee logs/train_fasc_seed2.log

echo "=== v9: residual scale (Phase A') + geometry polish + multiscale ==="
python scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --out submissions/submission_v9.csv \
  --thr 0.30 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  --residual-model-dir experiments/residual_models \
  --residual-prefer scale \
  2>&1 | tee logs/infer_v9.log

python scripts/eval_umud_local.py --pred submissions/submission_v8.csv --save-baseline || true
python scripts/eval_umud_local.py --pred submissions/submission_v9.csv; V9_RC=$? || true

echo "=== v10: + seed2 ensemble ==="
python scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --fasc-ckpt2 experiments/checkpoints_seed2/fasc_best.pt \
  --out submissions/submission_v10.csv \
  --thr 0.30 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  2>&1 | tee logs/infer_v10.log

python scripts/eval_umud_local.py --pred submissions/submission_v10.csv; V10_RC=$? || true

# Prefer best local combo among v9/v10 vs v8 for submit; also enable frangi v11 pass
echo "=== v11: ensemble + frangi fasc enhance ==="
# temporarily set frangi in config copy
python3 - <<'PY'
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path('configs/default.yaml').read_text())
cfg['infer']['frangi_enhance_fasc']=True
Path('configs/v11.yaml').write_text(yaml.dump(cfg))
PY
python scripts/calibrate_predict.py \
  --config configs/v11.yaml \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --fasc-ckpt2 experiments/checkpoints_seed2/fasc_best.pt \
  --out submissions/submission_v11.csv \
  --thr 0.28 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  2>&1 | tee logs/infer_v11.log

python scripts/eval_umud_local.py --pred submissions/submission_v11.csv; V11_RC=$? || true

python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd
import subprocess, sys

def combo(path):
    r=subprocess.run([sys.executable,'scripts/eval_umud_local.py','--pred',path,'--baseline','experiments/local_mae_baseline.json'], capture_output=True, text=True)
    # parse last json-ish from stdout by re-running metrics inline
    import yaml
    from muscle_arc.infer.predict import load_sample_submission
    from pathlib import Path as P
    cfg=yaml.safe_load(P('configs/default.yaml').read_text())
    pred=pd.read_csv(path)
    pred['stem']=pred['image_id'].map(lambda x: P(str(x)).stem)
    sample=load_sample_submission(P(cfg['data']['root'])/cfg['data']['sample_submission'], sep=cfg['data'].get('csv_sep',';'))
    id_col=[c for c in sample.columns if 'id' in c.lower()][0]
    pa_e=[]; fl_e=[]; mt_e=[]
    for _,s in sample.iterrows():
        rid=str(s[id_col]); stem=P(rid).stem
        m=pred[pred['image_id']==rid]
        if m.empty: m=pred[pred['stem']==stem]
        if m.empty: continue
        pa_e.append(abs(float(m.iloc[0].pa_deg)-float(s.pa_deg)))
        fl_e.append(abs(float(m.iloc[0].fl_mm)-float(s.fl_mm)))
        mt_e.append(abs(float(m.iloc[0].mt_mm)-float(s.mt_mm)))
    if not pa_e: return 9e9, None
    import numpy as np
    c=float(np.mean([np.mean(pa_e)/5, np.mean(fl_e)/10, np.mean(mt_e)/3]))
    return c, (float(np.mean(pa_e)), float(np.mean(fl_e)), float(np.mean(mt_e)))

cands=[]
for name in ['submission_v8.csv','submission_v9.csv','submission_v10.csv','submission_v11.csv']:
    p=Path('submissions')/name
    if not p.exists():
        continue
    c, detail=combo(str(p))
    print(name, 'combo', c, 'detail', detail)
    cands.append((c, name))
cands.sort()
best=cands[0][1]
print('BEST_LOCAL', best, 'combo', cands[0][0])
Path('experiments/best_local_submission.txt').write_text(best)
PY

BEST=$(cat experiments/best_local_submission.txt)
echo "Submitting best local: $BEST"
# Only submit if not v8 (must improve) — still submit best among v9+ if better than v8
python3 - <<'PY'
from pathlib import Path
best=Path('experiments/best_local_submission.txt').read_text().strip()
raise SystemExit(0 if best != 'submission_v8.csv' else 3)
PY
MSG="top10 climb ${BEST%.csv}: cluster-scale + local apo tangent + MS-TTA/ensemble/frangi"
bash /tmp/vm_submit_v8.sh "submissions/$BEST" "$MSG" || true
# reuse submit helper pattern
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
export KAGGLE_API_TOKEN="$TOKEN"
cp "submissions/$BEST" submissions/submission.csv
python - <<'PY'
import json
from pathlib import Path
home=Path.home()/'.kaggle'
token=(home/'access_token').read_text().strip()
(home/'kaggle.json').write_text(json.dumps({'username':'aagneye','key':token}))
(home/'kaggle.json').chmod(0o600)
PY
kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data \
  -f submissions/submission.csv -m "$MSG" || true
sleep 30
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/umud-challenge-muscle-architecture-in-ultrasound-data?pageSize=6" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); items=d if isinstance(d,list) else d.get('submissions') or d.get('data') or [];
[print(x.get('date') or x.get('dateSubmitted'),'|',x.get('publicScore'),'|',x.get('status'),'|',(x.get('description') or '')[:100]) for x in items[:6]]"
