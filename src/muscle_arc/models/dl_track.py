"""DL_Track_US VGG16-UNet mask inference (Gate0 reference).

Loads Keras .h5 weights when available. Returns float32 HxW probability maps
compatible with ``predict_prob`` / geometry (same spatial size as input gray).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _load_legacy_keras():
    """Load Keras API that can deserialize DL_Track TF2/Keras2 .h5 weights.

    Standalone Keras 3 rejects Conv2DTranspose(groups=...) from older exports.
    Prefer tf_keras (TF 2.16+ companion) then tensorflow.keras under legacy mode.
    """
    try:
        import tf_keras as keras  # type: ignore

        return keras
    except ImportError:
        pass
    try:
        import os

        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tensorflow (+ tf_keras for Keras 3) required for DL_Track weights. "
            "pip install tensorflow tf_keras"
        ) from exc
    return keras


def find_dl_track_models(root: Path | str = Path("data/external/dl_track")) -> dict[str, Path]:
    """Return {'apo': path, 'fasc': path} when both can be guessed from filenames.

    Prefers 2D VGG16 U-Nets (Gate0 / published DL_Track) over 3D IFSS fasc weights.
    """
    root = Path(root)
    if not root.exists():
        return {}
    h5s = list(root.rglob("*.h5")) + list(root.rglob("*.hdf5"))

    def _score_apo(p: Path) -> tuple[int, str]:
        low = p.name.lower()
        if "apo" not in low and "aponeuros" not in low:
            return (-1, low)
        score = 0
        if "vgg16" in low:
            score += 10
        if "3d" in low or "ifss" in low:
            score -= 5
        return (score, low)

    def _score_fasc(p: Path) -> tuple[int, str]:
        low = p.name.lower()
        if "fasc" not in low and "fascicle" not in low:
            return (-1, low)
        score = 0
        if "vgg16" in low:
            score += 10
        if "3d" in low or "ifss" in low:
            score -= 20  # temporal model; wrong input rank for still frames
        return (score, low)

    apo_cands = sorted(((_score_apo(p), p) for p in h5s), key=lambda t: t[0], reverse=True)
    fasc_cands = sorted(((_score_fasc(p), p) for p in h5s), key=lambda t: t[0], reverse=True)
    out: dict[str, Path] = {}
    if apo_cands and apo_cands[0][0][0] >= 0:
        out["apo"] = apo_cands[0][1]
    if fasc_cands and fasc_cands[0][0][0] >= 0:
        out["fasc"] = fasc_cands[0][1]
    return out


class DLTrackSegmenter:
    """Thin wrapper around two Keras U-Nets (apo + fasc)."""

    def __init__(
        self,
        apo_path: Path | str,
        fasc_path: Path | str,
        img_size: int = 512,
    ) -> None:
        # DL_Track .h5 files are TF/Keras-2 era. Standalone Keras 3 rejects
        # Conv2DTranspose(groups=...). Prefer tf_keras / legacy keras.
        keras = _load_legacy_keras()
        self.img_size = int(img_size)
        self.apo = keras.models.load_model(str(apo_path), compile=False)
        self.fasc = keras.models.load_model(str(fasc_path), compile=False)

    def _prep(self, gray: np.ndarray, model) -> tuple[np.ndarray, tuple[int, int], dict]:
        """Letterbox to square (aspect-preserving) then feed the net.

        Stretch-to-512 crushed console screenshots (sector ≪ frame) and was a
        likely cause of public ~1.0 vs Gate0 ~0.34. OSF crops are near-square
        so letterbox ≈ stretch there.
        """
        from muscle_arc.data.dataset import letterbox

        h, w = gray.shape[:2]
        g = gray.astype(np.float32)
        if g.max() > 1.5:
            g = g / 255.0
        # letterbox expects uint-like 0..255 for consistency with train pipeline
        g8 = (np.clip(g, 0, 1) * 255.0).astype(np.uint8)
        canvas, meta = letterbox(g8, self.img_size, is_mask=False)
        resized = canvas.astype(np.float32) / 255.0
        inp_shape = getattr(model, "input_shape", None)
        channels = 1
        temporal = False
        if inp_shape is not None and len(inp_shape) == 5:
            temporal = True
            channels = int(inp_shape[-1] or 1)
        elif inp_shape is not None and len(inp_shape) >= 4:
            channels = int(inp_shape[-1] or 1)
        if channels == 3:
            x = np.stack([resized, resized, resized], axis=-1)
        else:
            x = resized[..., None]
        x = x[None].astype(np.float32)
        if temporal:
            x = x[:, None, ...]
        return x, (h, w), meta

    def predict_prob(self, gray: np.ndarray, branch: str = "apo") -> np.ndarray:
        from muscle_arc.data.dataset import unletterbox

        model = self.apo if branch == "apo" else self.fasc
        x, (h, w), meta = self._prep(gray, model)
        pred = model.predict(x, verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        prob = np.asarray(pred).squeeze()
        if prob.ndim == 3:
            prob = prob[..., 0]
        elif prob.ndim == 4:
            prob = prob[0, ..., 0] if prob.shape[-1] <= 3 else prob[0]
        prob = prob.astype(np.float32)
        if float(prob.max()) > 1.5:
            prob = 1.0 / (1.0 + np.exp(-np.clip(prob, -20, 20)))
        # Map letterboxed probs back to original HxW
        out = unletterbox(prob, meta)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)

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
    print(f"DL_Track apo={apo_path}")
    print(f"DL_Track fasc={fasc_path}")
    return DLTrackSegmenter(apo_path, fasc_path, img_size=img_size)
