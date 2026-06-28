"""Image utility helpers for PyroSense AI.

Example:
    >>> import numpy as np
    >>> from utils.image_utils import ensure_bgr_uint8
    >>> img = np.zeros((10, 10, 3), dtype=np.uint8)
    >>> ensure_bgr_uint8(img).shape
    (10, 10, 3)
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

from utils.logger import logger


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to BGR uint8 NumPy array.

    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        >>> arr = pil_to_bgr(img)
        >>> arr.shape
        (8, 8, 3)
    """

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert BGR NumPy array to PIL RGB image.

    Example:
        >>> import numpy as np
        >>> img = np.zeros((8, 8, 3), dtype=np.uint8)
        >>> p = bgr_to_pil(img)
        >>> p.size
        (8, 8)
    """

    image_bgr = ensure_bgr_uint8(image_bgr)
    rgb = image_bgr[:, :, ::-1]
    return Image.fromarray(rgb, mode="RGB")


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is HxWx3 BGR uint8.

    Example:
        >>> import numpy as np
        >>> x = np.zeros((5, 5, 3), dtype=np.float32)
        >>> ensure_bgr_uint8(x).dtype.name
        'uint8'
    """

    if image is None:
        raise ValueError("Image is None")
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={getattr(image, 'shape', None)}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_keep_aspect(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """Resize image keeping aspect ratio so max(H,W)=max_size.

    Example:
        >>> import numpy as np
        >>> img = np.zeros((2000, 1000, 3), dtype=np.uint8)
        >>> out = resize_keep_aspect(img, 1000)
        >>> max(out.shape[:2])
        1000
    """

    image = ensure_bgr_uint8(image)
    h, w = image.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return image
    scale = max_size / float(m)
    new_w, new_h = int(w * scale), int(h * scale)
    try:
        import cv2

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.warning(f"OpenCV resize failed, falling back to PIL: {e}")
        pil = bgr_to_pil(image).resize((new_w, new_h))
        return pil_to_bgr(pil)


def encode_image_base64_jpeg(image_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode BGR image to base64 JPEG (no data URI prefix).

    Example:
        >>> import numpy as np
        >>> s = encode_image_base64_jpeg(np.zeros((10,10,3), dtype=np.uint8))
        >>> isinstance(s, str)
        True
    """

    image_bgr = ensure_bgr_uint8(image_bgr)
    pil = bgr_to_pil(image_bgr)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_base64_image(data_b64: str) -> np.ndarray:
    """Decode a base64 image string (optionally data URI) into BGR image.

    Example:
        >>> import numpy as np
        >>> b = encode_image_base64_jpeg(np.zeros((8,8,3), dtype=np.uint8))
        >>> img = decode_base64_image(b)
        >>> img.shape
        (8, 8, 3)
    """

    if "," in data_b64 and data_b64.strip().startswith("data:"):
        data_b64 = data_b64.split(",", 1)[1]
    raw = base64.b64decode(data_b64)
    pil = Image.open(BytesIO(raw)).convert("RGB")
    return pil_to_bgr(pil)


def bbox_area_ratio(bbox_xyxy: Tuple[float, float, float, float], frame_shape: Tuple[int, int, int]) -> float:
    """Compute bbox area ratio w.r.t. frame area.

    Example:
        >>> bbox_area_ratio((0,0,10,10), (100,100,3))
        0.01
    """

    x1, y1, x2, y2 = bbox_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    fh, fw = frame_shape[:2]
    denom = float(max(1, fw * fh))
    return float(area / denom)


LAST_PROCESSED_FRAME = None


def get_last_processed_frame():
    """Retrieve the last processed image frame (BGR numpy array or None)."""
    return LAST_PROCESSED_FRAME


def set_last_processed_frame(frame: np.ndarray) -> None:
    """Store the last processed image frame globally."""
    global LAST_PROCESSED_FRAME
    LAST_PROCESSED_FRAME = frame

