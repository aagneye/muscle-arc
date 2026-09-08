"""DL_Track_US VGG16-UNet mask inference (Gate0 reference).

Loads Keras .h5 weights when available. Returns float32 HxW probability maps
compatible with ``predict_prob`` / geometry (same spatial size as input gray).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def find_dl_track_models(root: Path | str = Path("data/external/dl_track")) -> dict[str, Path]:
    """Return {'apo': path, 'fasc': path} when both can be guessed from filenames."""
    root = Path(root)
    if not root.exists():
        return {}
    h5s = list(root.rglob("*.h5")) + list(root.rglob("*.hdf5"))
    apo = fasc = None
    for p in h5s:
        low = p.name.lower()
        if apo is None and ("apo" in low or "aponeuros" in low):
            apo = p
        if fasc is None and ("fasc" in low or "fascicle" in low):
            fasc = p
    out: dict[str, Path] = {}
    if apo is not None:
        out["apo"] = apo
    if fasc is not None:
        out["fasc"] = fasc
    return out


class DLTrackSegmenter:
    """Thin wrapper around two Keras U-Nets (apo + fasc)."""

    def __init__(
        self,
        apo_path: Path | str,
        fasc_path: Path | str,
        img_size: int = 512,
    ) -> None:
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "tensorflow required for DL_Track weights. pip install tensorflow"
            ) from exc
        self.img_size = int(img_size)
        self.apo = keras.models.load_model(str(apo_path), compile=False)
        self.fasc = keras.models.load_model(str(fasc_path), compile=False)

    def _prep(self, gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = gray.shape[:2]
        g = gray.astype(np.float32)
        if g.max() > 1.5:
            g = g / 255.0
        resized = cv2.resize(g, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        # DL_Track often expects 3-channel or 1-channel — match model input
        inp_shape = self.apo.input_shape
        if inp_shape is not None and len(inp_shape) >= 4 and inp_shape[-1] == 3:
            x = np.stack([resized, resized, resized], axis=-1)
        else:
            x = resized[..., None]
        return x[None].astype(np.float32), (h, w)

    def predict_prob(self, gray: np.ndarray, branch: str = "apo") -> np.ndarray:
        x, (h, w) = self._prep(gray)
        model = self.apo if branch == "apo" else self.fasc
        pred = model.predict(x, verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        prob = np.asarray(pred).squeeze()
        if prob.ndim == 3:
            prob = prob[..., 0]
        prob = prob.astype(np.float32)
        if float(prob.max()) > 1.5:  # logits
            prob = 1.0 / (1.0 + np.exp(-np.clip(prob, -20, 20)))
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    def predict_masks(
        self,
        gray: np.ndarray,
        apo_thr: float = 0.35,
        fasc_thr: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        apo_p = self.predict_prob(gray, "apo")
        fasc_p = self.predict_prob(gray, "fasc")
        apo = (apo_p > apo_thr).astype(np.uint8)
        fasc = (fasc_p > fasc_thr).astype(np.uint8)
        return apo, fasc, apo_p, fasc_p


def load_dl_track(
    root: Path | str = Path("data/external/dl_track"),
    apo_path: Path | str | None = None,
    fasc_path: Path | str | None = None,
    img_size: int = 512,
) -> DLTrackSegmenter:
    if apo_path is None or fasc_path is None:
        found = find_dl_track_models(root)
        apo_path = apo_path or found.get("apo")
        fasc_path = fasc_path or found.get("fasc")
    if apo_path is None or fasc_path is None:
        raise FileNotFoundError(
            f"DL_Track .h5 not found under {root}. Run scripts/download_dl_track.py"
        )
    return DLTrackSegmenter(apo_path, fasc_path, img_size=img_size)
