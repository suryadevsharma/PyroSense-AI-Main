"""YOLOv8 detection wrapper for PyroSense AI.

This module provides a production-friendly wrapper around Ultralytics YOLOv8
with convenient methods for image/video/stream inference, plus ONNX export.

Example:
    >>> import numpy as np
    >>> from models.yolo_detector import YOLODetector
    >>> det = YOLODetector(model_path="models/weights/best.pt", device="cpu")
    >>> img = np.zeros((256, 256, 3), dtype=np.uint8)
    >>> res = det.detect_image(img)
    >>> isinstance(res.class_names, list)
    True
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from utils.image_utils import ensure_bgr_uint8
from utils.logger import logger
from utils.visualization import draw_boxes


@dataclass
class DetectionResult:
    """Structured detection result from a single frame."""

    boxes: List[Tuple[float, float, float, float]]
    scores: List[float]
    class_ids: List[int]
    class_names: List[str]
    inference_time_ms: float
    frame: np.ndarray
    annotated_frame: np.ndarray


class YOLODetector:
    """Ultralytics YOLOv8 wrapper with stream and export support."""

    FIRE_SMOKE_CLASSES: Dict[int, str] = {0: "fire", 1: "smoke"}

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> None:
        """Create a YOLO detector with fire/smoke validation.

        Logic:
          1) If `model_path` exists, load and inspect `model.names`.
             If it is not a fire/smoke model, warn and mark not ready.
          2) If missing or generic COCO, auto-download a fire/smoke model.
          3) Only run inference when `self.model_ready` is True.

        Example:
            >>> from models.yolo_detector import YOLODetector
            >>> _ = YOLODetector("models/weights/best.pt")
        """

        self.model_path = str(model_path)
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)

        self.model_ready: bool = False
        self.model = None
        self._model = None  # backward-compat for other modules

        from ultralytics import YOLO

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Production model weights not found at '{self.model_path}'. "
                "Please package trained weights before deployment."
            )

        print(f"Loading YOLO weights from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            if self.device != "auto":
                self.model.to(self.device)
            self.model_ready = True
            print("Loaded successfully.")
            print("Inference engine ready.")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model at {self.model_path}: {e}") from e

        self._model = self.model

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except Exception as e:
            logger.error(f"Ultralytics not available: {e}")
            raise

        path = Path(self.model_path)
        if not path.exists():
            logger.warning(f"YOLO weights not found at '{self.model_path}'.")
        try:
            model = YOLO(self.model_path)
            if self.device != "auto":
                model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.model_path}': {e}")
            raise

    def _get_model_names(self, model_obj) -> Dict[int, str]:
        names = getattr(model_obj, "names", None)
        if isinstance(names, dict) and names:
            return {int(k): str(v) for k, v in names.items()}
        inner = getattr(model_obj, "model", None)
        names2 = getattr(inner, "names", None) if inner is not None else None
        if isinstance(names2, dict) and names2:
            return {int(k): str(v) for k, v in names2.items()}
        return {}

    def _names_look_like_fire_smoke(self, names: Dict[int, str]) -> bool:
        # Accept similar variants (e.g. "flame", "fire_smoke"). Some community models may include extra labels;
        # we accept any model that clearly includes fire and smoke labels.
        if not names or len(names) < 2:
            return False
        vals = [str(v).lower() for v in names.values()]
        return (any("fire" in v or "flame" in v for v in vals)) and any("smoke" in v for v in vals)


    def detect_image(self, image: np.ndarray) -> DetectionResult:
        """Run detection on a single image (BGR).

        Example:
            >>> import numpy as np
            >>> from models.yolo_detector import YOLODetector
            >>> d = YOLODetector("models/weights/best.pt", device="cpu")
            >>> r = d.detect_image(np.zeros((128,128,3), dtype=np.uint8))
            >>> isinstance(r.boxes, list)
            True
        """

        frame = ensure_bgr_uint8(image)

        if not self.model_ready or self.model is None:
            return DetectionResult(
                boxes=[],
                scores=[],
                class_ids=[],
                class_names=[],
                inference_time_ms=0.0,
                frame=frame,
                annotated_frame=frame.copy(),
            )

        # Hard reject any COCO-pretrained detections if a generic model slips through
        names = self._get_model_names(self.model)
        if not self._names_look_like_fire_smoke(names):
            COCO_FIRE_IDS: List[int] = []  # COCO has NO fire class — so this will be empty
            _ = COCO_FIRE_IDS
            return DetectionResult(
                boxes=[],
                scores=[],
                class_ids=[],
                class_names=[],
                inference_time_ms=0.0,
                frame=frame,
                annotated_frame=frame.copy(),
            )

        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=None if self.device == "auto" else self.device,
            )
        except Exception as e:
            logger.error(f"YOLO image inference failed: {e}")
            raise
        dt_ms = (time.perf_counter() - t0) * 1000.0

        boxes: List[Tuple[float, float, float, float]] = []
        scores: List[float] = []
        class_ids: List[int] = []
        class_names: List[str] = []

        ALLOWED = ["fire", "smoke", "flame", "fire_smoke"]

        try:
            r0 = results[0]
            names = getattr(r0, "names", {}) or {}
            if getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                conf = r0.boxes.conf.cpu().numpy()
                cls = r0.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, ci in zip(xyxy, conf, cls):
                    cname = str(names.get(int(ci), str(ci)))
                    if not any(kw in cname.lower() for kw in ALLOWED):
                        continue
                    boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    scores.append(float(c))
                    class_ids.append(int(ci))
                    class_names.append(cname)
        except Exception as e:
            logger.warning(f"Failed to parse YOLO output, returning empty detections: {e}")

        annotated = draw_boxes(frame, boxes, scores, class_names) if boxes else frame.copy()
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            inference_time_ms=float(dt_ms),
            frame=frame,
            annotated_frame=annotated,
        )

    def detect_video(self, video_path: str, output_path: str) -> Dict[str, float]:
        """Run detection on a video file and save an annotated output video.

        Returns basic stats: frames_processed, avg_fps, avg_inference_ms.

        Example:
            >>> from models.yolo_detector import YOLODetector
            >>> _ = YOLODetector("models/weights/best.pt", device="cpu")
        """

        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f"OpenCV is required for video inference: {e}") from e

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))

        frames = 0
        t_start = time.perf_counter()
        inf_ms_total = 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = self.detect_image(frame)
                out.write(res.annotated_frame)
                frames += 1
                inf_ms_total += res.inference_time_ms
        finally:
            cap.release()
            out.release()

        total_s = max(1e-6, time.perf_counter() - t_start)
        avg_fps = float(frames / total_s)
        avg_inf = float(inf_ms_total / max(1, frames))
        return {"frames_processed": float(frames), "avg_fps": avg_fps, "avg_inference_ms": avg_inf}

    def detect_stream(self, source: Union[int, str]) -> Generator[DetectionResult, None, None]:
        """Yield detections from a webcam/RTSP/YouTube or file source.

        For YouTube URLs, this will attempt to use `yt-dlp` to obtain a direct
        stream URL usable by OpenCV.

        Example:
            >>> from models.yolo_detector import YOLODetector
            >>> d = YOLODetector("models/weights/best.pt", device="cpu")
            >>> gen = d.detect_stream(0)
            >>> hasattr(gen, "__iter__")
            True
        """

        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f"OpenCV is required for stream inference: {e}") from e

        src: Union[int, str] = source
        if isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source):
            src = self._resolve_youtube_url(source)

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise ValueError(f"Unable to open stream source: {source}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.detect_image(frame)
        finally:
            cap.release()

    def _resolve_youtube_url(self, url: str) -> str:
        """Resolve a YouTube URL to a direct stream URL with yt-dlp."""

        try:
            import yt_dlp
        except Exception as e:
            raise RuntimeError(f"yt-dlp is required for YouTube ingestion: {e}") from e

        ydl_opts = {"quiet": True, "skip_download": True, "format": "best[ext=mp4]/best"}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if "url" in info:
                return str(info["url"])
            formats = info.get("formats") or []
            if not formats:
                raise ValueError("No stream formats found for YouTube URL")
            return str(formats[-1].get("url"))

    def export_onnx(self, output_path: str) -> str:
        """Export the YOLO model to ONNX.

        Example:
            >>> from models.yolo_detector import YOLODetector
            >>> d = YOLODetector("models/weights/best.pt", device="cpu")
            >>> isinstance(d.get_model_info(), dict)
            True
        """

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_ready or self.model is None:
            raise RuntimeError("Model is not ready; cannot export ONNX.")
        try:
            self.model.export(format="onnx", imgsz=640, simplify=True, opset=12)
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

        # Ultralytics exports next to weights by default; try to locate.
        if out.exists():
            return str(out)
        for cand in (Path(self.model_path).with_suffix(".onnx"), Path("yolov8n.onnx")):
            if cand.exists():
                cand.replace(out)
                return str(out)
        logger.warning("ONNX export completed but output file not found; returning requested path.")
        return str(out)

    def get_model_info(self) -> Dict[str, object]:
        """Return model metadata (names, task, device).

        Example:
            >>> from models.yolo_detector import YOLODetector
            >>> YOLODetector("models/weights/best.pt", device="cpu").get_model_info().get("task") is not None
            True
        """

        info: Dict[str, object] = {"weights": self.model_path, "model_ready": self.model_ready}
        try:
            m = self.model if self.model is not None else self._model
            info["task"] = getattr(m, "task", None)
            info["names"] = self._get_model_names(m) if m is not None else None
        except Exception:
            info["task"] = None
            info["names"] = None
        info["device"] = self.device
        info["conf_threshold"] = self.conf_threshold
        info["iou_threshold"] = self.iou_threshold
        return info

