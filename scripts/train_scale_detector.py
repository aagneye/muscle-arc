"""Train ScaleStripDetector on pseudo-labels / OSF GT.

Example:
  python scripts/build_scale_labels.py \\
    --folders data/raw/test_images_v2 data/raw/apo_imgs_v1 \\
    --out experiments/scale_pseudo_labels.csv
  python scripts/train_scale_detector.py \\
    --labels experiments/scale_pseudo_labels.csv \\
    --out experiments/checkpoints_scale/scale_detector.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from muscle_arc.models.scale_detector import (
    MM_HI,
    MM_LO,
    ScaleStripDetector,
    extract_border_strips,
    huber_mm_loss,
)


class ScaleLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.df.iloc[idx]
        gray = cv2.imread(str(r["path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(r["path"])
        x = extract_border_strips(gray)
        y = float(r["mm_per_pixel"])
        y = float(np.clip(y, MM_LO, MM_HI))
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/checkpoints_scale/scale_detector.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-conf", type=float, default=0.55)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.labels)
    df = df[df["mm_per_pixel"].notna()].copy()
    df = df[df["confidence"] >= args.min_conf]
    df = df[(df["mm_per_pixel"] >= MM_LO) & (df["mm_per_pixel"] <= MM_HI)]
    df = df[df["path"].map(lambda p: Path(p).exists())]
    if len(df) < 20:
        raise SystemExit(f"Need >=20 labels, got {len(df)}")

    idx = list(range(len(df)))
    random.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx = set(idx[:n_val])
    train_df = df.iloc[[i for i in idx if i not in val_idx]]
    val_df = df.iloc[list(val_idx)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScaleStripDetector().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(
        ScaleLabelDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ScaleLabelDataset(val_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    best_mae = 1e9
    args.out.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = huber_mm_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(y)
            n_tr += len(y)

        model.eval()
        errs = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                errs.extend((pred - y).abs().cpu().tolist())
        mae = float(np.mean(errs)) if errs else 1e9
        row = {"epoch": epoch, "train_loss": tr_loss / max(n_tr, 1), "val_mae": mae}
        history.append(row)
        print(f"epoch {epoch} train_loss={row['train_loss']:.5f} val_mae={mae:.5f}", flush=True)
        if mae < best_mae:
            best_mae = mae
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_mae": mae,
                    "mm_lo": MM_LO,
                    "mm_hi": MM_HI,
                    "n_train": len(train_df),
                    "n_val": len(val_df),
                },
                args.out,
            )
            print(f"  saved {args.out} (best val_mae={mae:.5f})", flush=True)

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"best_val_mae": best_mae, "history": history}, indent=2))
    print(f"Done best_val_mae={best_mae:.5f} metrics={metrics_path}")


if __name__ == "__main__":
    main()
