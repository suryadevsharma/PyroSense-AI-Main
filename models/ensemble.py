"""Ensemble logic: YOLOv8 + EfficientNetV2 with risk scoring.

Example:
    >>> from models.ensemble import weighted_score
    >>> weighted_score(0.7, 0.85, 0.65, 0.35)
    0.7525
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from utils.image_utils import bbox_area_ratio
from utils.visualization import severity_from_risk


def weighted_score(yolo_score: float, clf_score: float, yolo_weight: float = 0.65, clf_weight: float = 0.35) -> float:
    """Compute weighted average of two confidence scores."""

    y = float(yolo_score)
    c = float(clf_score)
    yw = float(yolo_weight)
    cw = float(clf_weight)
    denom = max(1e-9, (yw + cw))
    return float((y * yw + c * cw) / denom)


@dataclass
class RiskScore:
    """Risk score (0-100) and severity band."""

    score: float
    severity: str

    def as_dict(self) -> Dict[str, object]:
        return {"score": float(self.score), "severity": self.severity}


def compute_risk_score(
    *,
    confidence: float,
    bbox_xyxy: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int, int],
    growth_rate: float,
    smoke_presence: float,
) -> RiskScore:
    """Compute composite risk score (0-100).

    Formula:
      score = (confidence * 0.4) + (area_ratio * 0.3) + (growth_rate * 0.2) + (smoke_presence * 0.1)
    Inputs should be in [0,1]. Output scaled to 0-100.

    Example:
        >>> r = compute_risk_score(confidence=0.9, bbox_xyxy=(0,0,10,10), frame_shape=(100,100,3), growth_rate=0.1, smoke_presence=0.2)
        >>> 0 <= r.score <= 100
        True
    """

    conf = float(np.clip(confidence, 0.0, 1.0))
    area = float(np.clip(bbox_area_ratio(bbox_xyxy, frame_shape), 0.0, 1.0))
    growth = float(np.clip(growth_rate, 0.0, 1.0))
    smoke = float(np.clip(smoke_presence, 0.0, 1.0))

    # A fire covering > 20% of the frame is incredibly massive; 
    # we apply a multiplier so area has a much stronger impact on the final score.
    area_factor = float(np.clip(area * 3.5, 0.0, 1.0))

    from config.settings import get_settings
    settings = get_settings()
    w_conf = float(settings.risk_w_conf)
    w_area = float(settings.risk_w_area)
    w_growth = float(settings.risk_w_growth)
    w_smoke = float(settings.risk_w_smoke)

    # Normalize weights so they sum to 1.0
    total = w_conf + w_area + w_growth + w_smoke
    if total > 0.0:
        w_conf /= total
        w_area /= total
        w_growth /= total
        w_smoke /= total
    else:
        w_conf, w_area, w_growth, w_smoke = 0.40, 0.30, 0.20, 0.10

    raw = conf * w_conf + area_factor * w_area + growth * w_growth + smoke * w_smoke
    score = float(np.clip(raw * 100.0, 0.0, 100.0))

    # Non-linear boost for immediate threats (high confidence + large area)
    if conf > 0.65 and area > 0.05:
        score = float(np.clip(score * 1.5, 0.0, 100.0))

    return RiskScore(score=score, severity=severity_from_risk(score))


def smoke_presence_from_classes(class_names: List[str]) -> float:
    """Estimate smoke presence from detected class names.

    Example:
        >>> smoke_presence_from_classes(["fire"])
        0.0
    """

    names = [n.lower() for n in class_names]
    return 1.0 if any("smoke" in n for n in names) else 0.0

