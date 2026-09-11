"""Surfaces + Orientation Field (SOF) multitask segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SoftArgmax1d(nn.Module):
    """Soft-argmax over height for each column → normalized y in [0,1]."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (N, H, W)
        prob = F.softmax(logits, dim=1)
        n, h, w = logits.shape
        yy = torch.linspace(0, 1, h, device=logits.device, dtype=logits.dtype).view(1, h, 1)
        return (prob * yy).sum(dim=1)  # (N, W)


class SOFNet(nn.Module):
    """Shared encoder; apo 3-class, surface heatmaps, fasc presence, orientation."""

    def __init__(
        self,
        encoder: str = "tu-efficientnet_b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,  # unused; we take encoder features via decoder path
        )
        # Reuse Unet decoder channels by attaching heads on the final decoder feature.
        # Simpler approach: separate lightweight heads from RGB via shared Unet-like outs.
        # We run the Unet once per head group by branching from encoder.
        enc_channels = self.backbone.encoder.out_channels[-1]
        self.apo_head = nn.Sequential(
            nn.Conv2d(enc_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1),
        )
        self.surf_super = nn.Sequential(
            nn.Conv2d(enc_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.surf_deep = nn.Sequential(
            nn.Conv2d(enc_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.fasc_head = nn.Sequential(
            nn.Conv2d(enc_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.ori_head = nn.Sequential(
            nn.Conv2d(enc_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )
        self.soft_argmax = SoftArgmax1d()
        # Also keep a standard binary apo/fasc decode via SMP for stable IoU
        self.apo_bin = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.fasc_bin = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    def _enc_feat(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.encoder(x)
        # deepest feature map
        f = feats[-1]
        return f

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Binary seg (full resolution via Unet)
        apo_bin = self.apo_bin(x)
        fasc_bin = self.fasc_bin(x)
        # Multihead from encoder bottleneck — upsample to input size
        feat = self._enc_feat(x)
        apo_logits = F.interpolate(self.apo_head(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
        ss = F.interpolate(self.surf_super(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
        sd = F.interpolate(self.surf_deep(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
        fasc_log = F.interpolate(self.fasc_head(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
        ori = F.interpolate(self.ori_head(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
        # Surface soft-argmax over height from 1-channel heatmaps treated as logits HxW
        y_s = self.soft_argmax(ss.squeeze(1))
        y_d = self.soft_argmax(sd.squeeze(1))
        return {
            "apo_bin": apo_bin,
            "fasc_bin": fasc_bin,
            "apo_logits": apo_logits,
            "fasc_logits": fasc_log,
            "ori": ori,  # (N,2,H,W) cos2,sin2 (unnormalized)
            "y_super": y_s,
            "y_deep": y_d,
            "surf_super_map": ss,
            "surf_deep_map": sd,
        }


def build_sof_model(cfg: dict) -> SOFNet:
    m = cfg.get("model", cfg)
    return SOFNet(
        encoder=m.get("encoder", "tu-efficientnet_b4"),
        encoder_weights=m.get("encoder_weights", "imagenet"),
        in_channels=int(m.get("in_channels", 3)),
    )
