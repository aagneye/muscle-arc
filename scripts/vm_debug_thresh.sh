#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python <<'PY'
import torch, yaml
from pathlib import Path
import numpy as np
from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.infer.predict import predict_mask
from muscle_arc.models.segmentation import build_segmentation_model

cfg=yaml.safe_load(open('configs/default.yaml'))
device=torch.device('cuda')
def load(ckpt):
    m=build_segmentation_model(cfg['model']).to(device)
    p=torch.load(ckpt,map_location=device,weights_only=False)
    m.load_state_dict(p['model']); m.eval(); return m
apo=load('experiments/checkpoints/apo_best.pt')
fasc=load('experiments/checkpoints/fasc_best.pt')
paths=list_images(Path('data/raw/test_images_v2'))[:5]
for p in paths:
    g=read_gray(p)
    for thr in [0.2,0.3,0.4,0.5,75]:
        a=predict_mask(apo,g,512,device,True,thr)
        f=predict_mask(fasc,g,512,device,True,thr)
        print(p.name, 'thr', thr, 'apo', int(a.sum()), 'fasc', int(f.sum()))
PY
