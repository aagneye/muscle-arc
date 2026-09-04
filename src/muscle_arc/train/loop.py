from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class DiceBCELoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - ((2 * inter + self.eps) / (den + self.eps)).mean()
        return bce + dice


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * images.size(0)
    return total / max(1, len(loader.dataset))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        total += float(loss.item()) * images.size(0)
    return total / max(1, len(loader.dataset))


def save_checkpoint(model: nn.Module, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "meta": meta or {}}
    torch.save(payload, path)
