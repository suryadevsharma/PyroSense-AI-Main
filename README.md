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

## ✨ Production Polish & Performance

### 🛡️ Stateful Incident Management & Temporal Verification
- **Incident State Machine**: Implemented per-source state tracking (`SAFE` ➔ `ACTIVE` ➔ `RESOLVED`). Prevents duplicate notifications during an ongoing incident.
- **Temporal Verification**: Ignores single false-positive frames; requires continuous detection across $N$ consecutive frames (`MIN_CONSECUTIVE_DETECTIONS`) before declaring an `ACTIVE` incident.
- **Auto-Resolution Alerts**: Detects when a threat has cleared for $T$ seconds (`INCIDENT_CLEAR_SECONDS`) and automatically dispatches a resolution confirmation message.
- **Immediate Escalation**: Instantly triggers new notification cycles if the incident risk level escalates (e.g., from `MEDIUM` to `CRITICAL`), bypassing default channel cooldowns.

### 🔒 Enterprise-Grade Settings Security
- **Masked Read-Only Panel**: Prevents exposing secrets (SMTP password, Telegram Bot token, Webhook URLs) in the frontend. Replaced editable fields with read-only status cards showing `Loaded from Environment`.
- **Zero Front-End Exposure**: Removed password toggle eye icons, preventing any browser page source inspections or inspector leaks of server variables.
- **Server-Side Testing**: Refactored the test alert triggers ("Send Test Email", "Send Test Message", "Test Webhook") to consume server-side configuration variables directly.

### ⚡ Performance & Resource Optimizations
- **CLIP Embedding Caching/Reuse**: Computes the CLIP embedding once per frame and shares the vector between FAISS indexing and historical similarity search, cutting LLM/search processing time by **~80-150ms** on CPU.
- **FAISS Singleton Pattern**: Replaces repeated FAISS disk reads with a thread-safe singleton, significantly reducing memory and RAM spikes during dashboard interaction.
- **Database Query Aggregation**: Optimized dashboard calculations to query direct SQL `COUNT` aggregates rather than reading all database rows into memory. Saves database I/O and changes time complexity from $O(N)$ to $O(1)$.
- **Auto-Sanitization of Credentials**: Automatic validation strips standard single/double quotes, trailing carriage returns (`\r`), newlines (`\n`), and carriage spaces from loaded secrets, preventing `InvalidURL` failures.

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

---

## 📸 Screenshots & Video Recordings

Screenshots and videos captured from the live feed dashboard are saved locally in the following subfolders:
- **Screenshots**: `data/processed/snapshots/`
- **Video Recordings**: `data/processed/recordings/`

---

## 📄 License

MIT License. See `LICENSE`.
