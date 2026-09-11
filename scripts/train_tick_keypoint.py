"""Train TickKeypointNet on synthetic tick strips stamped onto real US images.

No manual labels required: ground truth tick centers / mm_per_pixel are exact
by construction (see src/muscle_arc/geometry/synth_ticks.py). Backgrounds are
drawn from real train/test images so the model sees authentic speckle,
console text, and colour-bar clutter, per RulerNet's synthetic-data ablation
(arXiv:2507.07077v2 Table 4/5): synthetic volume is the primary lever, and
using more diverse backgrounds/imperfections improves generalization to
unseen (real) tick appearances.

Example:
  python scripts/train_tick_keypoint.py \\
    --image-dirs data/raw/test_images_v2 data/raw/apo_imgs_v1 \\
    --out experiments/checkpoints_tick/tick_keypoint.pt \\
    --steps 2000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from muscle_arc.geometry.synth_ticks import synthetic_batch
from muscle_arc.models.tick_keypoint import (
    TickKeypointNet,
    extract_peaks,
    robust_median_pitch,
    save_checkpoint,
    tick_heatmap_loss,
)


def list_backgrounds(dirs: list[Path], limit: int | None = None) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in exts:
                paths.append(p)
    if limit is not None:
        random.Random(0).shuffle(paths)
        paths = paths[:limit]
    return paths


class SyntheticTickDataset(Dataset):
    """Wraps synth_ticks.synthetic_batch as a torch Dataset.

    Regenerates a fresh synthetic batch from a fixed background pool each
    epoch (on-the-fly generation, RulerNet §3.2), so the model never sees the
    exact same tick placement twice even though backgrounds repeat.
    """

    def __init__(
        self,
        backgrounds: list[np.ndarray],
        *,
        n_per_image: int = 4,
        strip_px: int = 32,
        heatmap_len: int = 256,
        seed: int = 0,
    ) -> None:
        self.backgrounds = backgrounds
        self.n_per_image = n_per_image
        self.strip_px = strip_px
        self.heatmap_len = heatmap_len
        self.seed = seed
        self._regen(seed)

    def _regen(self, seed: int) -> None:
        self.samples = synthetic_batch(
            self.backgrounds,
            n_per_image=self.n_per_image,
            strip_px=self.strip_px,
            heatmap_len=self.heatmap_len,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        profile, heat, _mm = self.samples[idx]
        x = torch.from_numpy(profile.astype(np.float32)).view(1, -1)
        y = torch.from_numpy(heat.astype(np.float32)).view(1, -1)
        return x, y


def load_grays(paths: list[Path], max_side: int = 512) -> list[np.ndarray]:
    out = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        h, w = im.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        out.append(im)
    return out


def evaluate_pitch_recovery(model: TickKeypointNet, backgrounds: list[np.ndarray], device: torch.device, seed: int = 999) -> float:
    """Sanity metric: median relative pitch error on a fresh synthetic batch
    using the true generator pitch as ground truth (offline analogue of
    RulerNet's mAPE/cm — see docs/research_scale_reader.md)."""
    from muscle_arc.geometry.synth_ticks import generate_synthetic_tick_strip

    rng = np.random.default_rng(seed)
    errs = []
    model.eval()
    with torch.no_grad():
        for _ in range(40):
            bg = backgrounds[int(rng.integers(0, len(backgrounds)))]
            sample = generate_synthetic_tick_strip(bg, axis="y", side="left", rng=rng)
            h, w = sample.image.shape[:2]
            s = max(8, min(32, h // 2, w // 2))
            region = sample.image[:, :s]
            profile = region.max(axis=1).astype(np.float32)
            profile_resized = cv2.resize(profile[None, :], (256, 1), interpolation=cv2.INTER_LINEAR)[0]
            x = torch.from_numpy(profile_resized / 255.0).view(1, 1, -1).to(device)
            heat = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            peaks = extract_peaks(heat, threshold=0.5)
            scale = 256 / max(h, 1)
            pitch_pred, _conf = robust_median_pitch(peaks)
            if pitch_pred is None:
                continue
            pitch_true_resized = sample.pitch_px * scale
            errs.append(abs(pitch_pred - pitch_true_resized) / max(pitch_true_resized, 1e-6))
    return float(np.median(errs)) if errs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/checkpoints_tick/tick_keypoint.pt"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--n-backgrounds", type=int, default=200)
    parser.add_argument("--n-per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bg_paths = list_backgrounds(args.image_dirs, limit=args.n_backgrounds)
    if not bg_paths:
        raise SystemExit(f"No background images found under {args.image_dirs}")
    backgrounds = load_grays(bg_paths)
    print(f"Loaded {len(backgrounds)} background images for synthetic ticks")

    model = TickKeypointNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    step = 0
    epoch = 0
    best_err = float("inf")
    while step < args.steps:
        ds = SyntheticTickDataset(
            backgrounds, n_per_image=args.n_per_image, seed=args.seed + epoch
        )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        model.train()
        for x, y in dl:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = tick_heatmap_loss(logits, y)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss={loss.item():.4f}")
            if step % args.eval_every == 0:
                err = evaluate_pitch_recovery(model, backgrounds, device)
                print(f"  [eval] median relative pitch error = {err:.4f}")
                if err < best_err:
                    best_err = err
                    save_checkpoint(
                        model, args.out, meta={"step": step, "median_rel_pitch_err": err}
                    )
                    print(f"  saved best checkpoint -> {args.out} (err={err:.4f})")
        epoch += 1

    final_err = evaluate_pitch_recovery(model, backgrounds, device)
    print(f"Final median relative pitch error: {final_err:.4f}")
    if final_err <= best_err:
        save_checkpoint(model, args.out, meta={"step": step, "median_rel_pitch_err": final_err})
        print(f"Saved final checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
