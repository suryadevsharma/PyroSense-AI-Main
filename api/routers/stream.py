"""WebSocket streaming inference endpoint for PyroSense AI API."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from inference.detector import InferenceEngine
from utils.image_utils import decode_base64_image
from utils.logger import logger

router = APIRouter(tags=["stream"])


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """Accept base64 frames and return JSON detections in real-time.

    Client sends JSON: {"frame_b64": "...", "location": "..." }
    Server responds JSON with `InferenceEngine.detect_image()` payload.
    """

    await websocket.accept()
    engine = InferenceEngine.get_shared_instance()
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            frame_b64 = str(data.get("frame_b64", ""))
            frame = decode_base64_image(frame_b64)
            payload = engine.detect_image(frame)
            payload["location"] = data.get("location", "Unknown")
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.warning(f"WebSocket stream error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

