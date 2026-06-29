"""Image upload detection endpoint for PyroSense AI API."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from alerts.alert_manager import AlertManager
from api.dependencies import get_db
from api.schemas import DetectResponse, DetectionItem, RiskInfo
from config.settings import get_settings
from database import crud
from inference.detector import InferenceEngine
from inference.gradcam_explainer import GradCamExplainer
from llm.faiss_history import FaissHistory
from llm.incident_summarizer import IncidentSummarizer
from utils.image_utils import pil_to_bgr
from utils.logger import logger

router = APIRouter(tags=["detection"])


def _save_image(path: Path, image_bgr: np.ndarray) -> None:
    try:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image_bgr)
    except Exception as e:
        logger.warning(f"Failed to save image {path}: {e}")


@router.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    location: str = Form(default="Unknown"),
    source: str = Form(default="upload"),
    db: Session = Depends(get_db),
) -> DetectResponse:
    """Run detection on an uploaded image and return JSON with artifacts."""

    settings = get_settings()
    engine = InferenceEngine.get_shared_instance()
    summarizer = IncidentSummarizer()
    faiss_hist = FaissHistory.get_shared_instance()

    raw = await file.read()
    from PIL import Image
    from io import BytesIO

    pil = Image.open(BytesIO(raw)).convert("RGB")
    frame = pil_to_bgr(pil)
    payload = engine.detect_image(frame)

    # Grad-CAM
    expl = GradCamExplainer(yolo_model=engine.yolo._model)  # best-effort
    heat_three = expl.generate_heatmap(frame)

    # Persist artifacts
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    snap_path = Path(settings.snapshots_dir) / f"{ts}.jpg"
    heat_path = Path(settings.heatmaps_dir) / f"{ts}.jpg"
    _save_image(snap_path, frame)
    _save_image(heat_path, heat_three)

    # Save to DB first with a placeholder summary, so we get the detection ID
    boxes = [tuple(d["bbox_xyxy"]) for d in payload.get("detections", [])]
    primary_class = str(payload.get("primary_class", "none"))
    conf = float(payload.get("ensemble_conf", 0.0))
    risk = payload.get("risk") or {}
    det_row = crud.create_detection(
        db,
        timestamp=datetime.utcnow(),
        class_name=primary_class,
        confidence=conf,
        boxes_xyxy=boxes,
        frame_path=str(snap_path),
        heatmap_path=str(heat_path),
        llm_summary="Agent analysis in progress...",
        source=source,
        risk_score=float(risk.get("score", 0.0)),
    )

    # Compute embedding once
    try:
        embedding = faiss_hist.embed(frame)
    except Exception as e:
        logger.warning(f"Failed to generate embedding: {e}")
        embedding = None

    # FAISS add + search using precomputed embedding
    faiss_hist.add_detection(
        detection_id=det_row.id,
        image_bgr=frame,
        frame_path=str(snap_path),
        class_name=primary_class,
        timestamp=payload.get("timestamp"),
        embedding=embedding,
    )
    similar = faiss_hist.search_similar(frame, top_k=3, embedding=embedding)

    # Agent or Fallback execution flow
    run_agent_flow = bool(settings.llm_provider == "groq" and settings.groq_api_key)
    agent_success = False
    llm_summary = ""

    if run_agent_flow:
        try:
            agent_context = {
                "risk_score": float(risk.get("score", 0.0)),
                "severity": str(risk.get("severity", "LOW")),
                "bbox": boxes,
                "confidence": conf,
                "timestamp": str(payload.get("timestamp")),
                "location": location,
            }
            agent_res = await summarizer.run_agent(
                context=agent_context,
                frame=frame,
                db=db,
                detection_id=det_row.id,
                payload=payload,
            )
            llm_summary = agent_res.get("summary", "")
            det_row.llm_summary = llm_summary
            db.commit()
            payload["llm_summary"] = llm_summary
            agent_success = True
            logger.info(f"Agent successfully processed detection. Actions: {agent_res.get('action_taken')}")
        except Exception as e:
            logger.warning(f"Agent flow failed: {e}. Falling back to standard pipeline.")

    if not agent_success:
        # Fallback: run original rule-based pipeline
        from models.yolo_detector import DetectionResult
        det_res = engine.yolo.detect_image(frame)
        llm_summary = summarizer.summarize(det_res, location=location)
        det_row.llm_summary = llm_summary
        db.commit()
        payload["llm_summary"] = llm_summary

        try:
            am = AlertManager()
            await am.trigger_alert(payload, detection_id=det_row.id, location=location, source=source)
        except Exception as e:
            logger.warning(f"Fallback alert dispatch failed: {e}")

    # Build response
    base = "/artifacts"
    snapshot_url = f"{base}/snapshots/{snap_path.name}"
    heatmap_url = f"{base}/heatmaps/{heat_path.name}"

    return DetectResponse(
        timestamp=str(payload.get("timestamp")),
        primary_class=str(payload.get("primary_class")),
        yolo_conf=float(payload.get("yolo_conf", 0.0)),
        clf_conf=float(payload.get("clf_conf", 0.0)),
        ensemble_conf=float(payload.get("ensemble_conf", 0.0)),
        risk=RiskInfo(score=float(risk.get("score", 0.0)), severity=str(risk.get("severity", "LOW"))),
        detections=[
            DetectionItem(
                bbox_xyxy=list(d["bbox_xyxy"]),
                score=float(d["score"]),
                class_id=int(d["class_id"]),
                class_name=str(d["class_name"]),
            )
            for d in payload.get("detections", [])
        ],
        inference_time_ms=float(payload.get("inference_time_ms", 0.0)),
        snapshot_url=snapshot_url,
        heatmap_url=heatmap_url,
        llm_summary=llm_summary,
        faiss_similar=similar,
        detection_id=det_row.id,
    )

