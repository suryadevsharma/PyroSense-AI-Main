"""FastAPI app entrypoint for PyroSense AI.

Run:
  uvicorn api.main:app --reload --port 8000

The API provides:
  - POST /api/v1/detect: multipart image upload detection
  - WS  /api/v1/ws/stream: base64 frame streaming inference
  - GET /api/v1/history: paginated history
  - GET /api/v1/health: service health info
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import detection, health, history, stream
from config.settings import get_settings
from database.migrations.init_db import init_db
from inference.detector import InferenceEngine
from utils.logger import logger

detector: InferenceEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    global detector
    settings = get_settings()
    try:
        init_db()
    except Exception as e:
        logger.warning(f"DB init failed on startup: {e}")

    try:
        from alerts.telegram_alert import start_telegram_bot
        start_telegram_bot()
    except Exception as e:
        logger.warning(f"Failed to start telegram bot background thread: {e}")

    try:
        detector = InferenceEngine()
    except Exception as e:
        logger.warning(f"Failed to pre-load detector on startup: {e}")

    logger.info("PyroSense API starting up.")
    yield
    logger.info("PyroSense API shutting down.")


app = FastAPI(title="PyroSense AI API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health_endpoint():
    return {"status": "ok", "model_loaded": detector is not None}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(detection.router, prefix="/api/v1")
app.include_router(stream.router, prefix="/api/v1")

# Static artifacts: snapshots + heatmaps
settings = get_settings()
artifacts_root = Path(settings.data_dir) / "processed"
(artifacts_root / "snapshots").mkdir(parents=True, exist_ok=True)
(artifacts_root / "heatmaps").mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=str(artifacts_root)), name="artifacts")

