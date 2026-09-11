from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from muscle_arc.data.dataset import letterbox, list_images, read_gray, stem_key, unletterbox
from muscle_arc.geometry.metrics import ArchitectureParams, clip_params, estimate_architecture


def enhance_fascicle_gray(gray: np.ndarray, blend: float = 0.35) -> np.ndarray:
    """Blend Frangi vesselness into gray to emphasize thin fascicle lines."""
    try:
        from skimage.filters import frangi
    except Exception:  # noqa: BLE001
        return gray
    g = gray.astype(np.float64)
    if g.max() > 1.0:
        g = g / 255.0
    v = frangi(g, sigmas=range(1, 4), black_ridges=False)
    v = v / (v.max() + 1e-8)
    out = (1.0 - blend) * g + blend * v
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


@torch.no_grad()
def predict_prob(
    model: nn.Module,
    image_gray: np.ndarray,
    img_size: int,
    device: torch.device,
    tta_hflip: bool = True,
    use_letterbox: bool = True,
    multiscale: bool = False,
    frangi_enhance: bool = False,
) -> np.ndarray:
    """Return soft probability map resized to the original image size."""
    h, w = image_gray.shape[:2]
    src = enhance_fascicle_gray(image_gray) if frangi_enhance else image_gray

    def _one_scale(scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-6:
            img = src
        else:
            nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
            img = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        if use_letterbox:
            canvas, meta = letterbox(img, img_size, is_mask=False)
        else:
            canvas = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            meta = None

        def _infer(arr: np.ndarray) -> np.ndarray:
            rgb = np.stack([arr, arr, arr], axis=-1)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            logits = model(tensor)
            return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        pred = _infer(canvas)
        if tta_hflip:
            pred = 0.5 * (pred + np.fliplr(_infer(np.fliplr(canvas).copy())))
        if meta is not None:
            pred_full = unletterbox(pred.astype(np.float32), meta)
        else:
            pred_full = cv2.resize(pred.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if pred_full.shape[:2] != (h, w):
            pred_full = cv2.resize(pred_full, (w, h), interpolation=cv2.INTER_LINEAR)
        return pred_full

    if not multiscale:
        return _one_scale(1.0)
    preds = [_one_scale(s) for s in (0.85, 1.0, 1.15)]
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    image_gray: np.ndarray,
    img_size: int,
    device: torch.device,
    tta_hflip: bool = True,
    mask_percentile: float = 75.0,
) -> np.ndarray:
    h, w = image_gray.shape[:2]
    pred = predict_prob(model, image_gray, img_size, device, tta_hflip)
    # predict_prob already at full res; threshold in place
    if mask_percentile and 0 < mask_percentile < 100:
        thresh = (
            float(np.percentile(pred, mask_percentile))
            if mask_percentile > 1
            else float(mask_percentile)
        )
    else:
        thresh = 0.5
    mask = (pred > thresh).astype(np.uint8)
    # Avoid median blur on sparse fascicle predictions — it erases thin lines.
    if int(mask.sum()) > 5000:
        mask = cv2.medianBlur(mask, 5)
    else:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def load_sample_submission(path: Path, sep: str = ";") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    # Fallback if separator was wrong
    if df.shape[1] == 1:
        df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def build_submission(
    image_ids: list[str],
    params: list[ArchitectureParams],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": image_ids,
            "pa_deg": [p.pa_deg for p in params],
            "fl_mm": [p.fl_mm for p in params],
            "mt_mm": [p.mt_mm for p in params],
        }
    )


def temporal_smooth(
    df: pd.DataFrame,
    window: int = 5,
    cols: tuple[str, ...] = ("pa_deg", "fl_mm", "mt_mm"),
) -> pd.DataFrame:
    """Median-smooth consecutive rows in groups of ``window`` (video snippets)."""
    out = df.copy()
    n = len(out)
    for start in range(0, n, window):
        end = min(start + window, n)
        if end - start < 2:
            continue
        for col in cols:
            out.loc[out.index[start:end], col] = float(
                np.median(out.loc[out.index[start:end], col].to_numpy())
            )
    return out


def run_folder_inference(
    apo_model: nn.Module,
    fasc_model: nn.Module,
    image_dir: Path,
    img_size: int,
    device: torch.device,
    mm_per_pixel: float,
    clip: dict,
    tta_hflip: bool = True,
    mask_percentile: float = 75.0,
) -> pd.DataFrame:
    apo_model.eval()
    fasc_model.eval()
    paths = list_images(image_dir)
    ids: list[str] = []
    params: list[ArchitectureParams] = []
    for path in paths:
        gray = read_gray(path)
        apo = predict_mask(
            apo_model, gray, img_size, device, tta_hflip, mask_percentile
        )
        fasc_prob = predict_prob(
            fasc_model, gray, img_size, device, tta_hflip, True, True, False
        )
        fasc = (fasc_prob > (mask_percentile if mask_percentile <= 1 else 0.30)).astype(np.uint8)
        est = estimate_architecture(
            apo, fasc, mm_per_pixel=mm_per_pixel, fasc_prob=fasc_prob, gray=gray
        )
        est = clip_params(
            est,
            pa_range=tuple(clip["pa_deg"]),
            fl_range=tuple(clip["fl_mm"]),
            mt_range=tuple(clip["mt_mm"]),
        )
        # Kaggle sample uses filenames with extension (e.g. IMG_00001.tif)
        ids.append(path.name)
        params.append(est)
    return build_submission(ids, params)
