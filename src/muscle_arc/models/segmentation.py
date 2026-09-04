from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_segmentation_model(model_cfg: dict) -> nn.Module:
    """Build an SMP segmentation model from config."""
    arch = model_cfg.get("architecture", "Unet")
    builder = getattr(smp, arch, None)
    if builder is None:
        raise ValueError(f"Unknown SMP architecture: {arch}")

    return builder(
        encoder_name=model_cfg.get("encoder", "tu-efficientnet_b4"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=int(model_cfg.get("in_channels", 3)),
        classes=int(model_cfg.get("classes", 1)),
    )
