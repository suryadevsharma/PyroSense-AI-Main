"""Real-time stream processing for webcam/RTSP/YouTube sources.

This provides a thin wrapper around OpenCV VideoCapture with demo-mode support.

Example:
    >>> from inference.stream_processor import StreamProcessor
    >>> _ = StreamProcessor()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np

from utils.logger import logger


@dataclass
class StreamFrame:
    """A single frame from a stream."""

    frame_bgr: np.ndarray
    source: str


class StreamProcessor:
    """Read frames from a source; supports demo fallback via `data/samples/`."""

    def __init__(self, samples_dir: str = "data/samples") -> None:
        self.samples_dir = str(samples_dir)

    def frames(self, source: Union[int, str], *, demo_mode: bool = False) -> Generator[StreamFrame, None, None]:
        """Yield frames from camera/RTSP/YouTube, or demo samples if needed."""

        if demo_mode:
            yield from self._demo_frames()
            return

        try:
            import cv2
        except Exception as e:
            logger.warning(f"OpenCV unavailable; switching to demo mode: {e}")
            yield from self._demo_frames()
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.warning("Unable to open camera/stream; switching to demo mode.")
            yield from self._demo_frames()
            return

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield StreamFrame(frame_bgr=frame, source=str(source))
        finally:
            cap.release()

    def _demo_frames(self) -> Generator[StreamFrame, None, None]:
        """Loop through sample images for demo live feed."""

        from utils.image_utils import pil_to_bgr
        from PIL import Image

        candidates: list[Path] = []
        # Prefer realistic assets in data/samples/ first
        p = Path(self.samples_dir)
        if p.exists():
            candidates.extend(sorted([x for x in p.glob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png"}]))

        # Only fall back to synthetic validation images if realistic ones are missing
        if not candidates:
            synth_val = Path("data/processed/dfire/images/val")
            if synth_val.exists():
                candidates.extend(sorted([x for x in synth_val.glob("*.jpg")]))

        # De-duplicate while preserving order
        seen = set()
        imgs: list[Path] = []
        for c in candidates:
            key = str(c.resolve())
            if key in seen:
                continue
            seen.add(key)
            imgs.append(c)
        if not imgs:
            # Generate synthetic frames if samples are absent.
            logger.warning("No sample images found; generating synthetic demo frames.")
            for i in range(50):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, 2] = min(255, i * 5)  # ramp red channel
                yield StreamFrame(frame_bgr=frame, source="demo-synthetic")
            return

        logger.info(f"Demo mode: looping through {len(imgs)} sample images.")
        while True:
            for img_path in imgs:
                try:
                    pil = Image.open(img_path).convert("RGB")
                    yield StreamFrame(frame_bgr=pil_to_bgr(pil), source=f"demo:{img_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load sample '{img_path}': {e}")

