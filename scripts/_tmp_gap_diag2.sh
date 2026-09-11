#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python - <<'PY'
import json
from pathlib import Path
import yaml

for name in ["gate2_grow.json", "gate2_grow_v2.json", "gate2_osf_umud.json", "gate2b_osf_pred_scale.json"]:
    p = Path("experiments") / name
    d = json.loads(p.read_text())
    pa, fl, mt = d["pa_mae"], d["fl_mae"], d["mt_mae"]
    print(
        f"{name}: umud={d['umud']:.3f} PA={pa:.2f} FL={fl:.2f} MT={mt:.2f} "
        f"terms PA={pa/6:.2f} FL={fl/12:.2f} MT={mt/3:.2f} "
        f"scale={d.get('scale_mode')} masks={d.get('masks_from')} "
        f"fl_ok={d.get('n_fl_ok')} mt_ok={d.get('n_mt_ok')} "
        f"mt_ratio={d.get('mt_ratio_median')} fl_ratio={d.get('fl_ratio_median')}"
    )

# Target terms for umud=0.40 if equal
print("target equal terms for S=0.40: each component avg 0.40 -> MAE_PA=2.4, MAE_FL=4.8, MAE_MT=1.2")
print("DL_Track-ish: PA~1.45 FL~3.74 MT~1.31 -> S~0.33")
print("Thwaits best OSF: ~0.36")

for c in ["configs/sof.yaml", "configs/default.yaml", "configs/phase4.yaml", "configs/grow.yaml"]:
    p = Path(c)
    if p.exists():
        y = yaml.safe_load(p.read_text())
        print(c, "img_size=", y.get("img_size"), "encoder=", y.get("encoder") or y.get("backbone"))

# Infer log scale assignment from sof
print("---")
sof = Path("logs/sof_v1b_infer.log")
if sof.exists():
    for line in sof.read_text().splitlines():
        if "Scale assignment" in line or "geom_stats" in line or "clip hits" in line:
            print(line)
PY
