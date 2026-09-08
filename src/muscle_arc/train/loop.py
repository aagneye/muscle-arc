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


class TverskyBCELoss(nn.Module):
    """BCE + Tversky — beta>alpha emphasizes recall on thin fascicle strands."""

    def __init__(self, alpha: float = 0.4, beta: float = 0.6, eps: float = 1e-6) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return bce + (1.0 - tversky).mean()


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


@torch.no_grad()
def validate_metrics(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    thr: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Val loss + soft Dice + hard IoU (for checkpoint selection)."""
    model.eval()
    total_loss = 0.0
    inter = 0.0
    union = 0.0
    soft_inter = 0.0
    soft_den = 0.0
    n = 0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        probs = torch.sigmoid(logits)
        pred = (probs > thr).float()
        inter += float((pred * masks).sum().item())
        union += float(((pred + masks) > 0).float().sum().item())
        soft_inter += float((probs * masks).sum().item())
        soft_den += float(probs.sum().item() + masks.sum().item())
        n += bs
    return {
        "loss": total_loss / max(1, n),
        "iou": inter / (union + eps),
        "dice": (2.0 * soft_inter + eps) / (soft_den + eps),
    }

def save_checkpoint(model: nn.Module, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "meta": meta or {}}
    torch.save(payload, path)
