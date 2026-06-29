"""Grad-CAM explainability for PyroSense AI (EigenCAM).

This module generates heatmaps for a given image and model, and returns a
side-by-side visualization: Original | Heatmap | Overlay.

For YOLOv8, model internals differ across versions. This implementation is
robust: it attempts to attach EigenCAM to a reasonable backbone layer and
falls back to a simple saliency-like heatmap if CAM cannot be produced.

Example:
    >>> import numpy as np
    >>> from inference.gradcam_explainer import GradCamExplainer
    >>> expl = GradCamExplainer()
    >>> img = np.zeros((128,128,3), dtype=np.uint8)
    >>> out = expl.generate_heatmap(img)
    >>> out.shape[1] >= img.shape[1] * 3
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.image_utils import ensure_bgr_uint8
from utils.logger import logger
from utils.visualization import concat_horiz


@dataclass
class HeatmapResult:
    """Heatmap images for UI rendering."""

    original: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    three_panel: np.ndarray


class GradCamExplainer:
    """Generate EigenCAM heatmaps for a PyTorch model."""

    def __init__(self, yolo_model: Optional[object] = None) -> None:
        """Create explainer. Optionally pass an Ultralytics YOLO model object.

        Example:
            >>> from inference.gradcam_explainer import GradCamExplainer
            >>> _ = GradCamExplainer()
        """

        self.yolo_model = yolo_model

    def generate_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Generate three-panel heatmap visualization (BGR): Original|Heatmap|Overlay.

        Example:
            >>> import numpy as np
            >>> from inference.gradcam_explainer import GradCamExplainer
            >>> out = GradCamExplainer().generate_heatmap(np.zeros((64,64,3), dtype=np.uint8))
            >>> out.shape[0]
            64
        """

        res = self.generate(image)
        return res.three_panel

    def generate(self, image: np.ndarray) -> HeatmapResult:
        """Generate heatmap assets.

        Returns:
          HeatmapResult with original, heatmap (jet), overlay, and three_panel.
        """

        img = ensure_bgr_uint8(image)
        try:
            heat = self._eigen_cam(img)
        except Exception as e:
            logger.warning(f"EigenCAM failed, falling back to heuristic heatmap: {e}")
            heat = self._fallback_heatmap(img)

        overlay = self._overlay(img, heat, alpha=0.45)
        three = concat_horiz([img, heat, overlay])
        return HeatmapResult(original=img, heatmap=heat, overlay=overlay, three_panel=three)

    def _eigen_cam(self, image_bgr: np.ndarray) -> np.ndarray:
        """Attempt to produce EigenCAM heatmap for YOLO model backbone."""

        if self.yolo_model is None:
            raise RuntimeError("No model provided for EigenCAM")
        try:
            import cv2
            import torch
            from pytorch_grad_cam import EigenCAM
        except Exception as e:
            raise RuntimeError(f"Grad-CAM dependencies not available: {e}") from e

        # Ultralytics model wrappers: yolo_model.model is a torch.nn.Module
        model = getattr(self.yolo_model, "model", None) or getattr(self.yolo_model, "model", None)
        if model is None:
            raise RuntimeError("Unable to access underlying torch model")

        # Choose a target layer: last convolution-like layer in backbone if possible
        target_layer = None
        for m in reversed(list(model.modules())):
            name = m.__class__.__name__.lower()
            if "conv" in name or "c2f" in name or "bottleneck" in name:
                target_layer = m
                break
        if target_layer is None:
            raise RuntimeError("No suitable target layer found for CAM")

        rgb = image_bgr[:, :, ::-1]
        img = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = torch.from_numpy(np.transpose(img, (2, 0, 1))[None, ...])
        x = x.to(next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu")

        cam = EigenCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=x, targets=None)[0]
        grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)

        # Resize CAM back to original size
        hm = cv2.resize(grayscale_cam, (image_bgr.shape[1], image_bgr.shape[0]))
        hm = (hm * 255.0).astype(np.uint8)
        jet = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        return jet

    def _fallback_heatmap(self, image_bgr: np.ndarray) -> np.ndarray:
        """Create a simple edge/intensity-based heatmap as a fallback."""

        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f"OpenCV required for fallback heatmap: {e}") from e

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 120)
        blur = cv2.GaussianBlur(edges, (0, 0), sigmaX=7, sigmaY=7)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def _overlay(self, image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f"OpenCV required for overlay: {e}") from e

        a = float(np.clip(alpha, 0.0, 1.0))
        return cv2.addWeighted(heatmap_bgr, a, image_bgr, 1.0 - a, 0.0)

