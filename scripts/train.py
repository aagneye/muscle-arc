#!/usr/bin/env python3
"""Train aponeurosis and fascicle segmentation models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from muscle_arc.data.dataset import UltrasoundSegDataset, pair_images_masks
from muscle_arc.data.paths import DataPaths
from muscle_arc.models.segmentation import build_segmentation_model
from muscle_arc.train.loop import DiceBCELoss, save_checkpoint, train_one_epoch, validate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_branch(
    name: str,
    pairs: list,
    cfg: dict,
    device: torch.device,
) -> Path:
    img_size = int(cfg["img_size"])
    train_cfg = cfg["train"]
    dataset = UltrasoundSegDataset(pairs, img_size=img_size)
    val_frac = float(train_cfg["val_fraction"])
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
    )

    model = build_segmentation_model(cfg["model"]).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    best_path = ckpt_dir / f"{name}_best.pt"
    best_val = float("inf")

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        va = validate(model, val_loader, loss_fn, device)
        print(f"[{name}] epoch {epoch}: train={tr:.4f} val={va:.4f}")
        if va < best_val:
            best_val = va
            save_checkpoint(model, best_path, meta={"branch": name, "epoch": epoch, "val": va})
            print(f"[{name}] saved {best_path}")

    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--branch",
        choices=("apo", "fasc", "both"),
        default="both",
        help="Which segmentation branch to train",
    )
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    device_name = cfg.get("device", "cuda")
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )

    paths = DataPaths.from_config(cfg["data"])
    paths.assert_train_present()

    if args.branch in ("apo", "both"):
        apo_pairs = pair_images_masks(paths.apo_imgs, paths.apo_masks)
        print(f"Apo pairs: {len(apo_pairs)}")
        train_branch("apo", apo_pairs, cfg, device)

    if args.branch in ("fasc", "both"):
        fasc_pairs = pair_images_masks(paths.fasc_imgs, paths.fasc_masks)
        print(f"Fasc pairs: {len(fasc_pairs)}")
        train_branch("fasc", fasc_pairs, cfg, device)


if __name__ == "__main__":
    main()
