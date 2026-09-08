#!/usr/bin/env python3
"""Train aponeurosis and fascicle segmentation models (sequence-safe splits)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from muscle_arc.data.augments import build_train_transform
from muscle_arc.data.dataset import UltrasoundSegDataset, pair_images_masks
from muscle_arc.data.paths import DataPaths
from muscle_arc.data.splits import filter_pairs_by_stems, load_split_manifest
from muscle_arc.models.segmentation import build_segmentation_model
from muscle_arc.train.gate1_lite import gate1_lite_score
from muscle_arc.train.loop import (
    DiceBCELoss,
    TverskyBCELoss,
    save_checkpoint,
    train_one_epoch,
    validate_metrics,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_pairs(
    name: str,
    pairs: list,
    cfg: dict,
    split_dir: Path | None,
) -> tuple[list, list]:
    """Return (train_pairs, val_pairs). Holdout is never returned for training."""
    train_cfg = cfg["train"]
    manifest_path = None
    if split_dir is not None:
        cand = split_dir / f"{name}_splits.json"
        if cand.exists():
            manifest_path = cand

    if manifest_path is not None:
        man = load_split_manifest(manifest_path)
        splits = man["splits"]
        train_pairs = filter_pairs_by_stems(pairs, splits["train"])
        val_pairs = filter_pairs_by_stems(pairs, splits["val"])
        hold_n = len(splits.get("holdout", []))
        print(
            f"[{name}] splits from {manifest_path}: "
            f"train={len(train_pairs)} val={len(val_pairs)} holdout={hold_n} (excluded)"
        )
        if not train_pairs or not val_pairs:
            raise SystemExit(f"[{name}] empty train/val after split filter")
        return train_pairs, val_pairs

    dataset = UltrasoundSegDataset(pairs, img_size=int(cfg["img_size"]))
    val_frac = float(train_cfg.get("val_fraction", 0.15))
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )
    train_idx = set(train_ds.indices)
    val_idx = set(val_ds.indices)
    train_pairs = [pairs[i] for i in sorted(train_idx)]
    val_pairs = [pairs[i] for i in sorted(val_idx)]
    print(
        f"[{name}] WARN no split manifest — random_split "
        f"train={len(train_pairs)} val={len(val_pairs)}"
    )
    return train_pairs, val_pairs


def train_branch(
    name: str,
    pairs: list,
    cfg: dict,
    device: torch.device,
    loss_name: str = "dice",
    letterbox_resize: bool = True,
    warm_start: Path | None = None,
    split_dir: Path | None = None,
    use_augments: bool | None = None,
    apo_pairs_all: list | None = None,
    fasc_pairs_all: list | None = None,
) -> Path:
    img_size = int(cfg["img_size"])
    train_cfg = cfg["train"]
    train_pairs, val_pairs = _resolve_pairs(name, pairs, cfg, split_dir)

    if use_augments is None:
        use_augments = bool(train_cfg.get("use_augments", True))
    aug_strength = str(train_cfg.get("aug_strength", "mild"))
    if name == "fasc" and aug_strength == "fasc":
        strength = "fasc"
    else:
        strength = "mild" if name == "apo" else aug_strength
    transform = build_train_transform(img_size, strength=strength) if use_augments else None

    train_ds = UltrasoundSegDataset(
        train_pairs,
        img_size=img_size,
        transform=transform,
        letterbox_resize=letterbox_resize,
    )
    val_ds = UltrasoundSegDataset(
        val_pairs,
        img_size=img_size,
        transform=None,
        letterbox_resize=letterbox_resize,
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
    if warm_start is not None and warm_start.exists():
        payload = torch.load(warm_start, map_location=device, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
        print(f"[{name}] warm-started from {warm_start}")

    if loss_name == "tversky":
        loss_fn = TverskyBCELoss(
            alpha=float(train_cfg.get("tversky_alpha", 0.4)),
            beta=float(train_cfg.get("tversky_beta", 0.6)),
        )
    else:
        loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    best_path = ckpt_dir / f"{name}_best.pt"
    patience = int(train_cfg.get("early_stop_patience", 12))
    bad_epochs = 0
    epochs = int(train_cfg["epochs"])
    lite_every = int(train_cfg.get("gate1_lite_every", 0) or 0)
    lite_max = int(train_cfg.get("gate1_lite_max_samples", 24))
    apo_thr = float(train_cfg.get("apo_thr", cfg.get("infer", {}).get("apo_thr", 0.35)))
    fasc_thr = float(train_cfg.get("fasc_thr", cfg.get("infer", {}).get("fasc_thr", 0.10)))

    # Partner masks for Gate1-lite (matched stems on val)
    apo_by_stem = {p[0].stem: p for p in (apo_pairs_all or [])}
    fasc_by_stem = {p[0].stem: p for p in (fasc_pairs_all or [])}
    val_stems = sorted({p[0].stem for p in val_pairs})
    use_lite = lite_every > 0 and bool(apo_by_stem) and bool(fasc_by_stem)
    if use_lite:
        print(
            f"[{name}] Gate1-lite every {lite_every} epochs "
            f"(max_samples={lite_max}, patience counts lite steps)"
        )

    best_score = float("-inf")  # maximize geom score or IoU
    select_mode = "gate1_lite" if use_lite else "iou"

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        va = validate_metrics(model, val_loader, loss_fn, device)
        scheduler.step(va["loss"])
        print(
            f"[{name}] epoch {epoch}: train={tr:.4f} "
            f"val_loss={va['loss']:.4f} val_iou={va['iou']:.4f} val_dice={va['dice']:.4f}"
        )

        improved = False
        meta = {
            "branch": name,
            "epoch": epoch,
            "val_loss": va["loss"],
            "val_iou": va["iou"],
            "val_dice": va["dice"],
            "loss": loss_name,
            "letterbox": letterbox_resize,
            "augments": use_augments,
            "aug_strength": strength if use_augments else None,
            "select_mode": select_mode,
        }

        if use_lite and epoch % lite_every == 0:
            g1 = gate1_lite_score(
                branch=name,
                model=model,
                fasc_by_stem=fasc_by_stem,
                apo_by_stem=apo_by_stem,
                stems=val_stems,
                img_size=img_size,
                device=device,
                apo_thr=apo_thr,
                fasc_thr=fasc_thr,
                max_samples=lite_max,
            )
            meta.update(
                {
                    "gate1_lite_pa": g1["pa_mae_deg"],
                    "gate1_lite_fl": g1["fl_rel_mae"],
                    "gate1_lite_mt": g1["mt_rel_mae"],
                    "gate1_lite_score": g1["score"],
                    "gate1_lite_n": g1["n"],
                }
            )
            print(
                f"[{name}] gate1_lite: n={int(g1['n'])} "
                f"pa={g1['pa_mae_deg']:.2f} fl_rel={g1['fl_rel_mae']:.3f} "
                f"mt_rel={g1['mt_rel_mae']:.3f} score={g1['score']:.4f}"
            )
            if g1["score"] > best_score:
                best_score = g1["score"]
                improved = True
                save_checkpoint(model, best_path, meta=meta)
                print(f"[{name}] saved {best_path} (gate1_lite={best_score:.4f})")
            bad_epochs = 0 if improved else bad_epochs + 1
            if bad_epochs >= patience:
                print(
                    f"[{name}] early stop at epoch {epoch} "
                    f"(lite_patience={patience}, best_score={best_score:.4f})"
                )
                break
        elif not use_lite:
            if va["iou"] > best_score:
                best_score = va["iou"]
                improved = True
                save_checkpoint(model, best_path, meta=meta)
                print(f"[{name}] saved {best_path} (iou={best_score:.4f})")
            bad_epochs = 0 if improved else bad_epochs + 1
            if bad_epochs >= patience:
                print(
                    f"[{name}] early stop at epoch {epoch} "
                    f"(patience={patience}, best_iou={best_score:.4f})"
                )
                break

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
    parser.add_argument(
        "--loss",
        choices=("dice", "tversky"),
        default=None,
        help="Fascicle loss (default: tversky). Apo always uses dice.",
    )
    parser.add_argument(
        "--no-letterbox",
        action="store_true",
        help="Disable aspect-preserving letterbox resize",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start (used for --branch apo|fasc; for both use --warm-start-dir)",
    )
    parser.add_argument(
        "--warm-start-dir",
        type=Path,
        default=None,
        help="Directory with {apo,fasc}_best.pt to warm-start each branch",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Dir with {apo,fasc}_splits.json (default: train.split_dir in config)",
    )
    parser.add_argument(
        "--no-augments",
        action="store_true",
        help="Disable train-time albumentations",
    )
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    device_name = cfg.get("device", "cuda")
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    letterbox_resize = not args.no_letterbox
    split_dir = args.split_dir
    if split_dir is None and cfg.get("train", {}).get("split_dir"):
        split_dir = Path(cfg["train"]["split_dir"])
    use_augments = False if args.no_augments else None

    paths = DataPaths.from_config(cfg["data"])
    paths.assert_train_present()

    apo_pairs = pair_images_masks(paths.apo_imgs, paths.apo_masks)
    fasc_pairs = pair_images_masks(paths.fasc_imgs, paths.fasc_masks)

    def _warm(branch: str) -> Path | None:
        if args.warm_start is not None:
            return args.warm_start
        if args.warm_start_dir is not None:
            p = args.warm_start_dir / f"{branch}_best.pt"
            return p if p.exists() else None
        return None

    if args.branch in ("apo", "both"):
        print(f"Apo pairs: {len(apo_pairs)}")
        train_branch(
            "apo",
            apo_pairs,
            cfg,
            device,
            loss_name="dice",
            letterbox_resize=letterbox_resize,
            warm_start=_warm("apo"),
            split_dir=split_dir,
            use_augments=use_augments,
            apo_pairs_all=apo_pairs,
            fasc_pairs_all=fasc_pairs,
        )

    if args.branch in ("fasc", "both"):
        print(f"Fasc pairs: {len(fasc_pairs)}")
        fasc_loss = args.loss or "tversky"
        train_branch(
            "fasc",
            fasc_pairs,
            cfg,
            device,
            loss_name=fasc_loss,
            letterbox_resize=letterbox_resize,
            warm_start=_warm("fasc"),
            split_dir=split_dir,
            use_augments=use_augments,
            apo_pairs_all=apo_pairs,
            fasc_pairs_all=fasc_pairs,
        )


if __name__ == "__main__":
    main()
