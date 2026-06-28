"""FAISS similarity search for historical detections using CLIP embeddings.

This module maintains a persistent FAISS index and associated metadata.
It embeds frames via CLIP (transformers) and returns the top-k similar past
incidents.

If optional dependencies are missing, operations degrade gracefully and return
empty results.

Example:
    >>> from llm.faiss_history import FaissHistory
    >>> h = FaissHistory()
    >>> h.search_similar(None, top_k=3)  # gracefully handles missing image
    []
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import get_settings
from utils.image_utils import bgr_to_pil, ensure_bgr_uint8
from utils.logger import logger


@dataclass
class SimilarItem:
    """A similar detection from history."""

    detection_id: int
    score: float
    frame_path: Optional[str]
    class_name: Optional[str]
    timestamp: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "detection_id": int(self.detection_id),
            "score": float(self.score),
            "frame_path": self.frame_path,
            "class_name": self.class_name,
            "timestamp": self.timestamp,
        }


_SHARED_FAISS: Optional[FaissHistory] = None


class FaissHistory:
    """Persistent FAISS index with CLIP embeddings."""

    @classmethod
    def get_shared_instance(cls) -> FaissHistory:
        global _SHARED_FAISS
        if _SHARED_FAISS is None:
            _SHARED_FAISS = cls()
        return _SHARED_FAISS

    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.settings.faiss_dir / "index.faiss"
        self.meta_path = self.settings.faiss_dir / "meta.jsonl"
        self._clip = None
        self._tokenizer = None
        self._processor = None
        self._index = None
        self._dim = 512
        self._meta: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        self._meta = []
        if self.meta_path.exists():
            for line in self.meta_path.read_text(encoding="utf-8").splitlines():
                try:
                    self._meta.append(json.loads(line))
                except Exception:
                    continue

        try:
            import faiss
        except Exception as e:
            logger.warning(f"FAISS not available: {e}")
            return

        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._dim = int(self._index.d)
            logger.info(f"Loaded FAISS index dim={self._dim} size={self._index.ntotal}")
        else:
            self._index = faiss.IndexFlatIP(self._dim)
            logger.info("Initialized new FAISS index (cosine via inner product on normalized vectors).")

    def _load_clip(self) -> None:
        if self._clip is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            raise RuntimeError(f"transformers required for CLIP embeddings: {e}") from e

        model_id = "openai/clip-vit-base-patch32"
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._clip = CLIPModel.from_pretrained(model_id)
        self._clip.eval()

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Embed an image into a normalized float32 vector."""

        self._load_clip()
        pil = bgr_to_pil(ensure_bgr_uint8(image_bgr))
        inputs = self._processor(images=pil, return_tensors="pt")
        import torch

        with torch.inference_mode():
            feats = self._clip.get_image_features(**inputs)
        vec = feats.detach().cpu().numpy().astype(np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
        return vec

    def add_detection(
        self,
        *,
        detection_id: int,
        image_bgr: np.ndarray,
        frame_path: Optional[str],
        class_name: Optional[str],
        timestamp: Optional[str],
    ) -> None:
        """Add a detection frame to FAISS index."""

        if self._index is None:
            return
        try:
            vec = self.embed(image_bgr)
        except Exception as e:
            logger.warning(f"Embedding unavailable; skipping FAISS add: {e}")
            return

        try:
            import faiss
        except Exception:
            return

        self._index.add(vec)
        meta = {"detection_id": int(detection_id), "frame_path": frame_path, "class_name": class_name, "timestamp": timestamp}
        self._meta.append(meta)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        with self.meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        faiss.write_index(self._index, str(self.index_path))

    def search_similar(self, image_bgr: Optional[np.ndarray], top_k: int = 3) -> List[Dict[str, object]]:
        """Return top-k similar historical detections.

        Returns list of dicts compatible with Streamlit rendering.
        """

        if image_bgr is None or self._index is None or not self._meta:
            return []

        try:
            vec = self.embed(image_bgr)
        except Exception as e:
            logger.warning(f"Embedding unavailable; skipping FAISS search: {e}")
            return []

        k = int(max(1, top_k))
        scores, idx = self._index.search(vec, k)
        out: List[Dict[str, object]] = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i < 0 or i >= len(self._meta):
                continue
            m = self._meta[i]
            out.append(
                SimilarItem(
                    detection_id=int(m.get("detection_id", -1)),
                    score=float(s),
                    frame_path=m.get("frame_path"),
                    class_name=m.get("class_name"),
                    timestamp=m.get("timestamp"),
                ).as_dict()
            )
        return out

