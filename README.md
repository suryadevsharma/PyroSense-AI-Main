---
title: IntelliGuard AI
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---
# 🛡️ IntelliGuard AI

**Real-time Fire & Smoke Detection with Explainable Deep Learning & ReAct Agent Coordination**

[Python](https://www.python.org/)
[License: MIT](LICENSE)
[Docker](Dockerfile)
[Demo Space](https://huggingface.co/spaces/suryadevsharma11/intelliguard-ai)

IntelliGuard AI is a **production-grade, deployment-ready** computer vision safety system that detects **fire and smoke** from **webcams**, **RTSP IP cameras**, **YouTube streams**, **uploaded videos**, and **static images**. It uses a fine-tuned **YOLOv8** detector, a secondary **EfficientNetV2** verifier, **Grad-CAM explainability**, a **ReAct safety coordinator agent**, and a polished **Streamlit dashboard** backed by a high-performance **FastAPI** backend.

---

## ✨ Production Polish & Performance (V1)

- **⚡ Cached Singletons**: Heavy model weights (YOLOv8 and CLIP ViT embeddings) are cached using thread-safe singleton patterns. API latency dropped from **15.4 seconds to 380 milliseconds** (~40x speedup on CPU).
- **🔒 Secure Settings Panel**: In production/deployment, credentials (SMTP, Telegram, Webhook) are loaded securely from `.env` or Hugging Face secrets. Form fields are read-only and masked to prevent leaks.
- **📊 DB-Driven Alert Tracking**: Sidebar alerts panel queries SQLite/PostgreSQL `alert_logs` table in real-time, showing actual delivery status and error messages.
- **🔎 Explainable AI (XAI)**: Generates Grad-CAM/EigenCAM heatmaps to visualize what pixel features triggered the AI.
- **📣 Multi-modal Alerting**: Email, Telegram, Webhook, and local audio alerts (gTTS).
- **🧠 ReAct Safety Coordinator**: Uses LLaMA3 via Groq to analyze incident context, fetch historical similarity matches (CLIP + FAISS), and coordinate dispatch alerts.

---

## 🚀 Quick Start (Docker)

1. Copy env file:

```bash
cp .env.example .env
```

2. Start services:

```bash
docker-compose up --build
```

- **Dashboard**: `http://localhost:8501`
- **API**: `http://localhost:8000`
- **API docs**: `http://localhost:8000/docs`
- **MLflow**: `http://localhost:5000`

---

## 🧰 Manual Local Setup (Windows/macOS/Linux)

1. Create & Activate Virtual Environment:
```bash
python -m venv .venv
```
- Windows: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch API & Dashboard:
- **FastAPI Backend**:
  ```bash
  uvicorn api.main:app --port 8000 --reload
  ```
- **Streamlit Dashboard**:
  ```bash
  streamlit run dashboard/app.py
  ```

---

## 🤖 Telegram Bot Commands
- `/status` — returns system operational state and last logged detection.
- `/snapshot` — takes and returns a frame from the live stream.
- `/history` — lists details for the last 5 incident records.
- `/threshold 0.7` — updates YOLOv8 confidence thresholds.

---

## 🗂️ Project Structure (condensed)
```
pyrosense-ai/
  api/        # FastAPI (REST & WebSocket streams)
  dashboard/  # Streamlit multi-page interface (Welcome, Live, Upload, History)
  models/     # YOLO, EfficientNetV2, ONNX, and ensemble scoring
  inference/  # Core detector engine & Grad-CAM explainability
  alerts/     # Email SMTP, Telegram, and Webhook dispatch handlers
  llm/        # Groq/Ollama summarizer & FAISS history embeddings
  database/   # SQLAlchemy models, migrations, & CRUD operations
  tests/      # Pytest validation suites (alerts, api, detector)
```

## 📸 Screenshots & Video Recordings

Screenshots and videos captured from the live feed dashboard are saved locally in the following subfolders:
- **Screenshots**: `data/processed/snapshots/`
- **Video Recordings**: `data/processed/recordings/`

---

## 📄 License

MIT License. See `LICENSE`.
