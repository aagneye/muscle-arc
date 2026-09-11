#!/usr/bin/env python3
"""Write sequence-safe train / val / holdout manifests for apo and fasc."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from muscle_arc.data.dataset import pair_images_masks
from muscle_arc.data.paths import DataPaths
from muscle_arc.data.splits import (
    assign_groups_to_splits,
    group_ids_for_stems,
    write_split_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/splits"))
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["seed"])
    paths = DataPaths.from_config(cfg["data"])
    paths.assert_train_present()

    for branch, img_d, mask_d in (
        ("apo", paths.apo_imgs, paths.apo_masks),
        ("fasc", paths.fasc_imgs, paths.fasc_masks),
    ):
        pairs = pair_images_masks(img_d, mask_d)
        stems = [p[0].stem for p in pairs]
        splits = assign_groups_to_splits(
            stems,
            seed=seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            holdout_frac=args.holdout_frac,
        )
        # Sanity: no group leakage
        for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
            ga = group_ids_for_stems(splits[a])
            gb = group_ids_for_stems(splits[b])
            leak = ga & gb
            if leak:
                raise SystemExit(f"{branch} group leak {a}/{b}: {sorted(leak)[:5]}")

        path = write_split_manifest(
            args.out_dir,
            branch,
            splits,
            seed=seed,
            meta={
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "holdout_frac": args.holdout_frac,
                "n_pairs": len(pairs),
            },
        )
        print(
            f"{branch}: wrote {path} "
            f"train={len(splits['train'])} val={len(splits['val'])} "
            f"holdout={len(splits['holdout'])} groups="
            f"{len(group_ids_for_stems(stems))}"
        )


if __name__ == "__main__":
    main()
