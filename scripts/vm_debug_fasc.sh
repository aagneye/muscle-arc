#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python <<'PY'
import torch, yaml, numpy as np
from pathlib import Path
from muscle_arc.data.dataset import list_images, read_gray, pair_images_masks, read_mask
from muscle_arc.models.segmentation import build_segmentation_model

cfg=yaml.safe_load(open('configs/default.yaml'))
device=torch.device('cuda')

def load(ckpt):
    m=build_segmentation_model(cfg['model']).to(device)
    p=torch.load(ckpt,map_location=device,weights_only=False)
    print(ckpt, 'meta', p.get('meta'))
    m.load_state_dict(p['model']); m.eval(); return m

fasc=load('experiments/checkpoints/fasc_best.pt')
apo=load('experiments/checkpoints/apo_best.pt')

# Compare logits on a TRAIN fasc image vs TEST image
train_pairs=pair_images_masks(Path('data/raw/fasc_imgs_v1'), Path('data/raw/fasc_masks_v1'))
ip, mp = train_pairs[10]
g=read_gray(ip); gt=read_mask(mp)
print('train fasc img', ip.name, g.shape, 'gt sum', gt.sum())

def probs(model, gray, size=512):
    import cv2
    r=cv2.resize(gray,(size,size))
    rgb=np.stack([r,r,r],-1)
    t=torch.from_numpy(rgb).permute(2,0,1).float()[None]/255.0
    with torch.no_grad():
        logits=model(t.to(device))
        p=torch.sigmoid(logits)[0,0].cpu().numpy()
    return p

p_train=probs(fasc,g)
print('fasc on TRAIN: min/mean/max/p50/p90/p99', p_train.min(), p_train.mean(), p_train.max(), np.percentile(p_train,[50,90,99]))

test=list_images(Path('data/raw/test_images_v2'))[0]
gt2=read_gray(test)
p_test=probs(fasc,gt2)
print('fasc on TEST', test.name, ': min/mean/max/p50/p90/p99', p_test.min(), p_test.mean(), p_test.max(), np.percentile(p_test,[50,90,99]))

p_apo_test=probs(apo,gt2)
print('apo on TEST: min/mean/max/p50/p90/p99', p_apo_test.min(), p_apo_test.mean(), p_apo_test.max(), np.percentile(p_apo_test,[50,90,99]))

# Also try apo model on fasc train (sanity)
p_wrong=probs(apo,g)
print('apo model on fasc TRAIN: mean/max', p_wrong.mean(), p_wrong.max())
PY
