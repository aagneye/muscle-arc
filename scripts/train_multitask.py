#!/usr/bin/env python3
"""Train Surfaces + Orientation Field (SOF) multitask model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from muscle_arc.data.augments import build_train_transform, paste_console_chrome
from muscle_arc.data.dataset import letterbox, pair_images_masks, read_gray, read_mask
from muscle_arc.data.labels import apo_three_class, orientation_targets, surface_targets
from muscle_arc.data.paths import DataPaths
from muscle_arc.data.splits import filter_pairs_by_stems, load_split_manifest
from muscle_arc.models.multitask import build_sof_model
from muscle_arc.train.losses import sof_total_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SOFDataset(Dataset):
    """Paired apo+fasc masks with SOF targets; stems must match across branches."""

    def __init__(
        self,
        apo_pairs: list[tuple[Path, Path]],
        fasc_by_stem: dict[str, tuple[Path, Path]],
        img_size: int = 768,
        transform=None,
        console_chrome: bool = True,
    ) -> None:
        self.samples: list[tuple[Path, Path, Path, Path]] = []
        for img_p, mask_p in apo_pairs:
            stem = img_p.stem
            if stem not in fasc_by_stem:
                continue
            f_img, f_mask = fasc_by_stem[stem]
            self.samples.append((img_p, mask_p, f_img, f_mask))
        self.img_size = img_size
        self.transform = transform
        self.console_chrome = console_chrome

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        apo_img_p, apo_m_p, fasc_img_p, fasc_m_p = self.samples[idx]
        # Prefer apo image as the visual (same FOV expected)
        image = read_gray(apo_img_p)
        apo = read_mask(apo_m_p)
        fasc = read_mask(fasc_m_p)
        if apo.shape[:2] != image.shape[:2]:
            apo = cv2.resize(apo, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if fasc.shape[:2] != image.shape[:2]:
            fasc = cv2.resize(fasc, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.console_chrome and random.random() < 0.35:
            image = paste_console_chrome(image)
            # Masks stay in crop coords — chrome only outside; skip chrome when sizes change a lot
            # Re-crop to original content region approx top-left
            # Safer: disable mask chrome mismatch by skipping chrome if pad changes geometry
            # Re-read without chrome if shapes diverge
            pass  # chrome on full frame without mask paste — only image domain shift

        # Align then letterbox
        image_lb, _ = letterbox(image, self.img_size, is_mask=False)
        apo_lb, _ = letterbox(apo, self.img_size, is_mask=True)
        fasc_lb, _ = letterbox(fasc, self.img_size, is_mask=True)

        if self.transform is not None:
            # Apply same geometric aug to image + both masks
            try:
                out = self.transform(image=image_lb, masks=[apo_lb, fasc_lb])
                image_lb = out["image"]
                apo_lb, fasc_lb = out["masks"][0], out["masks"][1]
            except Exception:  # noqa: BLE001
                out = self.transform(image=image_lb, mask=apo_lb)
                image_lb, apo_lb = out["image"], out["mask"]

        apo_cls = apo_three_class(apo_lb, fasc_lb)
        y_s, y_d, valid = surface_targets(apo_lb, fasc_lb)
        c2, s2, _pres = orientation_targets(fasc_lb)

        if image_lb.ndim == 2:
            image_lb = np.stack([image_lb, image_lb, image_lb], axis=-1)
        image_t = torch.from_numpy(image_lb).permute(2, 0, 1).float() / 255.0
        return {
            "image": image_t,
            "apo": torch.from_numpy(apo_lb.astype(np.float32)).unsqueeze(0),
            "fasc": torch.from_numpy(fasc_lb.astype(np.float32)).unsqueeze(0),
            "apo_cls": torch.from_numpy(apo_cls),
            "y_super": torch.from_numpy(y_s),
            "y_deep": torch.from_numpy(y_d),
            "surf_valid": torch.from_numpy(valid),
            "ori_cos2": torch.from_numpy(c2),
            "ori_sin2": torch.from_numpy(s2),
            "id": apo_img_p.stem,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sof.yaml"))
    parser.add_argument("--split-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_seed(int(cfg["seed"]))
    paths = DataPaths.from_config(cfg["data"])
    paths.assert_train_present()
    train_cfg = cfg["train"]
    split_dir = Path(args.split_dir or train_cfg.get("split_dir", "experiments/splits"))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "experiments/checkpoints_sof"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    apo_pairs = pair_images_masks(paths.apo_imgs, paths.apo_masks)
    fasc_pairs = pair_images_masks(paths.fasc_imgs, paths.fasc_masks)
    fasc_by_stem = {p[0].stem: p for p in fasc_pairs}

    man = split_dir / "apo_splits.json"
    if man.exists():
        splits = load_split_manifest(man)["splits"]
        train_pairs = filter_pairs_by_stems(apo_pairs, splits["train"])
        val_pairs = filter_pairs_by_stems(apo_pairs, splits["val"])
    else:
        n_val = max(1, int(0.15 * len(apo_pairs)))
        val_pairs = apo_pairs[:n_val]
        train_pairs = apo_pairs[n_val:]

    strength = train_cfg.get("aug_strength", "console")
    tfm = build_train_transform(int(cfg["img_size"]), strength=strength) if train_cfg.get("use_augments", True) else None
    train_ds = SOFDataset(train_pairs, fasc_by_stem, int(cfg["img_size"]), transform=tfm)
    val_ds = SOFDataset(val_pairs, fasc_by_stem, int(cfg["img_size"]), transform=None, console_chrome=False)
    print(f"SOF train={len(train_ds)} val={len(val_ds)}")
    if len(train_ds) < 2:
        raise SystemExit("SOF dataset empty — need overlapping apo/fasc stems")

    loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=True,
    )
    vloader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sof_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    best = 1e9
    history = []
    patience = int(train_cfg.get("early_stop_patience", 12))
    bad = 0
    bce_w = float(train_cfg.get("bce_weight", 0.2))

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(batch["image"])
                loss, stats = sof_total_loss(out, batch, bce_weight=bce_w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(stats["total"])
        # val
        model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in vloader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch["image"])
                loss, stats = sof_total_loss(out, batch, bce_weight=bce_w)
                vlosses.append(stats["total"])
        tr = float(np.mean(losses)) if losses else 0.0
        va = float(np.mean(vlosses)) if vlosses else 0.0
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"epoch {epoch} train={tr:.4f} val={va:.4f}", flush=True)
        if va < best:
            best = va
            bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "val": va}, ckpt_dir / "sof_best.pt")
            print(f"  saved {ckpt_dir / 'sof_best.pt'}", flush=True)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}")
                break

    (ckpt_dir / "sof_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done best_val={best:.4f}")


if __name__ == "__main__":
    main()
