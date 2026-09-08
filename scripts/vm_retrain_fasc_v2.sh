#!/usr/bin/env bash
# Retrain fascicle branch with Tversky loss for sparse masks on GPU 1.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${1:-1}"
mkdir -p logs experiments/checkpoints

# Patch train loop temporarily via env-driven script
python <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
cfg["train"]["epochs"] = 60
cfg["train"]["batch_size"] = 4
cfg["train"]["lr"] = 2.0e-4
cfg["train"]["checkpoint_dir"] = "experiments/checkpoints_fasc_v2"
Path("configs/fasc_v2.yaml").write_text(yaml.dump(cfg))
print("wrote configs/fasc_v2.yaml")
PY

# Create training entry that uses Tversky+BCE for fasc
cat > /tmp/train_fasc_tversky.py <<'PY'
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from muscle_arc.data.dataset import UltrasoundSegDataset, pair_images_masks
from muscle_arc.data.paths import DataPaths
from muscle_arc.models.segmentation import build_segmentation_model
from muscle_arc.train.loop import save_checkpoint, validate

class TverskyBCE(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha, self.beta, self.eps = alpha, beta, eps
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (2,3)
        tp = (probs*targets).sum(dims)
        fp = (probs*(1-targets)).sum(dims)
        fn = ((1-probs)*targets).sum(dims)
        tversky = (tp+self.eps)/(tp+self.alpha*fp+self.beta*fn+self.eps)
        return bce + (1-tversky).mean()

cfg=yaml.safe_load(open('configs/fasc_v2.yaml'))
random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paths=DataPaths.from_config(cfg['data']); paths.assert_train_present()
pairs=pair_images_masks(paths.fasc_imgs, paths.fasc_masks)
print('fasc pairs', len(pairs), 'device', device)
ds=UltrasoundSegDataset(pairs, img_size=int(cfg['img_size']))
n_val=max(1,int(len(ds)*cfg['train']['val_fraction'])); n_train=len(ds)-n_val
train_ds,val_ds=random_split(ds,[n_train,n_val], generator=torch.Generator().manual_seed(cfg['seed']))
tl=DataLoader(train_ds,batch_size=cfg['train']['batch_size'],shuffle=True,num_workers=4,pin_memory=True)
vl=DataLoader(val_ds,batch_size=cfg['train']['batch_size'],shuffle=False,num_workers=4,pin_memory=True)
model=build_segmentation_model(cfg['model']).to(device)
# warm start from previous fasc checkpoint if present
prev=Path('experiments/checkpoints/fasc_best.pt')
if prev.exists():
    payload=torch.load(prev,map_location=device,weights_only=False)
    model.load_state_dict(payload['model'])
    print('warm-started from', prev)
loss_fn=TverskyBCE()
opt=torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=float(cfg['train']['weight_decay']))
scaler=torch.cuda.amp.GradScaler() if device.type=='cuda' else None
best=float('inf'); out=Path(cfg['train']['checkpoint_dir'])/'fasc_best.pt'
from tqdm import tqdm
for epoch in range(1, int(cfg['train']['epochs'])+1):
    model.train(); total=0.0
    for batch in tqdm(tl, desc=f'train{epoch}', leave=False):
        imgs=batch['image'].to(device); masks=batch['mask'].to(device)
        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                loss=loss_fn(model(imgs), masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss=loss_fn(model(imgs), masks); loss.backward(); opt.step()
        total+=float(loss.item())*imgs.size(0)
    tr=total/max(1,len(train_ds))
    # val with same loss
    model.eval(); vtot=0.0
    with torch.no_grad():
        for batch in vl:
            imgs=batch['image'].to(device); masks=batch['mask'].to(device)
            vtot+=float(loss_fn(model(imgs), masks).item())*imgs.size(0)
    va=vtot/max(1,len(val_ds))
    print(f'[fasc_v2] epoch {epoch}: train={tr:.4f} val={va:.4f}', flush=True)
    if va<best:
        best=va; save_checkpoint(model, out, meta={'branch':'fasc_v2','epoch':epoch,'val':va})
        print('saved', out, flush=True)
print('done best', best)
PY

nohup python /tmp/train_fasc_tversky.py > logs/train_fasc_v2_gpu${CUDA_VISIBLE_DEVICES}.log 2>&1 &
echo $! > logs/train_fasc_v2.pid
echo "Started fasc_v2 PID=$(cat logs/train_fasc_v2.pid) on GPU $CUDA_VISIBLE_DEVICES"
