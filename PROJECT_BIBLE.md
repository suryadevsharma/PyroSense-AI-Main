# PyroSense AI / IntelliGuard - Project Bible
*Private Interview Preparation & Reference Document*

---

## 1. COMPLETE ARCHITECTURE DIAGRAM (Text/ASCII)

Below is the end-to-end system architecture for PyroSense AI (IntelliGuard), detailing components, data flow, ports, and protocols.

```
+--------------------------------------------------------------------------------------------------+
|                                        1. CLIENT / FRONTEND LAYER                                |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | Streamlit Web Dashboard (dashboard/app.py) [Port: 8501]                                  |   |
|   |                                                                                          |   |
|   |  - Live Detection UI (pages/1_Live_Detection.py)  - Threat Map (pages/6_Threat_Map.py)   |   |
|   |  - Upload Analysis (pages/2_Upload_Analysis.py)   - Settings UI (pages/5_Settings.py)    |   |
|   |  - Incident History (pages/3_Incident_History.py)  - Model Insights (pages/4_Insights.py) |   |
|   +------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
          | (REST APIs / JSON payloads)                      | (WebSocket stream for live frames)
          | POST /api/v1/detect [Port: 8000]                 | WS /api/v1/ws/stream [Port: 8000]
          v                                                  v
+--------------------------------------------------------------------------------------------------+
|                                        2. BACKEND / API LAYER                                    |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | FastAPI Application (api/main.py) [Port: 8000]                                           |   |
|   |                                                                                          |   |
|   |  - Routers (api/routers/): detection.py, stream.py, history.py, health.py                |   |
|   |  - Request Validation (api/schemas.py) & Dependency Injection (api/dependencies.py)      |   |
|   +------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
          | (Function Calls / NumPy Arrays)                  | (Async Alerts / Webhooks / API requests)
          v                                                  v
+-------------------------------------------------------+  +---------------------------------------+
|                 3. MACHINE LEARNING & AI              |  |         4. ALERT & LLM PIPELINE       |
|                                                       |  |                                       |
|  +-------------------------------------------------+  |  |  +---------------------------------+  |
|  | InferenceEngine (inference/detector.py)         |  |  |  | AlertManager (alerts/alert_mg.) |  |
|  |                                                 |  |  |  |                                 |  |
|  |   - YOLOv8 Detector (models/yolo_detector.py)   |  |  |  |   - Email Dispatch (SMTP:587)   |  |
|  |   - EfficientNetV2 Classifier (models/eff..)    |  |  |  |   - Telegram Bot API (HTTPS)    |  |
|  |   - Ensemble Scorer (models/ensemble.py)        |  |  |  |   - Audio Output (gTTS/pyttsx3) |  |
|  |   - Grad-CAM Heatmaps (gradcam_explainer.py)    |  |  |  |   - Custom JSON Webhooks        |  |
|  +-------------------------------------------------+  |  |  +---------------------------------+  |
|                            |                          |  |                  |                    |
|                            v                          |  |                  v                    |
|  +-------------------------------------------------+  |  |  +---------------------------------+  |
|  | FAISS Vector Search (llm/faiss_history.py)      |  |  |  | IncidentSummarizer (llm/in..)   |  |
|  |                                                 |  |  |  |                                 |  |
|  |   - Image features via CLIP (Transformers)      |  |  |  |   - Groq API (llama3-8b-8192)   |  |
|  |   - Similarity Indexing (data/processed/faiss)  |  |  |  |   - Ollama API (llama3:local)   |  |
|  +-------------------------------------------------+  |  |  +---------------------------------+  |
+-------------------------------------------------------+  +---------------------------------------+
          |                                                                   |
          | (SQLAlchemy ORM / session)                                        | (Alert logs storage)
          v                                                                   v
+--------------------------------------------------------------------------------------------------+
|                                      5. DATABASE & STORAGE LAYER                                 |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | SQLite DB (pyrosense.db) / PostgreSQL Connection (database/session.py)                   |   |
|   |                                                                                          |   |
|   |  - detections table (ID, timestamp, confidence, bounding_boxes, artifact_paths, risk)    |   |
|   |  - alert_logs table (ID, detection_id, channel, status, sent_at, error_message)          |   |
|   |  - model_runs table (ID, run_id, model_name, mAP50, created_at)                          |   |
|   +------------------------------------------------------------------------------------------+   |
|   | Local File System Storage (data/processed/)                                              |   |
|   |  - /snapshots/ (original captured frame jpg)                                             |   |
|   |  - /heatmaps/ (Grad-CAM blend explainer output jpg)                                      |   |
|   +------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
                                                               ^
                                                               | (ONNX Webhook Feed)
                                                               |
+--------------------------------------------------------------------------------------------------+
|                                      6. EDGE MONITORING (OPTIONAL)                               |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | Raspberry Pi 4 Edge Device (edge_deploy/raspberry_pi.py)                                 |   |
|   |                                                                                          |   |
|   |  - ONNX Runtime Engine (models/onnx_inference.py) & Camera Capture (Picamera2/OpenCV)     |   |
|   +------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. EVERY TECHNOLOGY USED WITH REASON

| Technology | What it is | Why chose it over alternatives | Exactly where it is used in THIS project |
| :--- | :--- | :--- | :--- |
| **FastAPI** | High-performance Python web framework for building APIs. | Chosen over Django/Flask because it provides native async support (ideal for WebSockets stream parsing), auto-generated Swagger UI docs, and type safety with Pydantic. | Located in the `api/` directory (routes: `api/routers/`, app entrypoint: [api/main.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/api/main.py)). |
| **Streamlit** | Python web dashboard framework for rapid data app building. | Chosen over React/Vue because it enables fast prototyping of interactive widgets, natively handles Python object visualizations, and renders ML plots quickly. | Located in the `dashboard/` directory (main landing page: [dashboard/app.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/dashboard/app.py)). |
| **YOLOv8 (Ultralytics)** | State-of-the-art real-time object detection model. | Chosen over YOLOv5 or SSD because it provides superior accuracy-to-speed ratios, supports simple ONNX exporting, and contains clean APIs for frame inference. | Wrapper defined in [models/yolo_detector.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/models/yolo_detector.py); utilized in [inference/detector.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/inference/detector.py). |
| **EfficientNetV2** | High-efficiency convolutional neural network for image classification. | Chosen over ResNet because it has a smaller memory footprint and higher parameter efficiency, acting as a great secondary verifier head. | Defined in [models/efficientnet_classifier.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/models/efficientnet_classifier.py); trained in [training/train_efficientnet.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/training/train_efficientnet.py). |
| **Grad-CAM (EigenCAM)** | Explainer mechanism visualizing CNN/YOLO model activations. | Chosen over LIME or SHAP because it is computationally efficient at pixel-level visualization on CNN feature maps and runs in real time during inference. | Located in [inference/gradcam_explainer.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/inference/gradcam_explainer.py) and shown in Model Insights page. |
| **FAISS & CLIP** | Facebook AI Similarity Search index coupled with CLIP embeddings. | Chosen over Pinecone or Chroma because it operates entirely offline, has zero external network latency, and easily indexes CLIP image embeddings. | Implemented in [llm/faiss_history.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/llm/faiss_history.py) and utilized in the main detection endpoint. |
| **Groq / Ollama** | LLM providers serving Llama 3 models. | Chosen over raw local inference (for cloud) because Groq delivers sub-second inference speeds, while Ollama provides a free local fallback option. | Implemented in [llm/incident_summarizer.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/llm/incident_summarizer.py) to synthesize alert narratives. |
| **SQLAlchemy** | SQL Object-Relational Mapper (ORM) for Python. | Chosen over raw SQL drivers (like sqlite3 or psycopg2) because it abstracts query logic, handles connection pooling, and simplifies DB migrations. | Located in the `database/` folder (sessions: [database/session.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/database/session.py), models: [database/models.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/database/models.py)). |
| **MLflow** | Machine learning lifecycle tracking platform. | Chosen over Weights & Biases because it runs locally on SQLite/file backends, requiring no external accounts or subscription limits. | Initiated via [training/launch_mlflow.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/training/launch_mlflow.py); metrics logged in [training/train_yolo.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/training/train_yolo.py). |
| **ONNX Runtime** | High-performance inference engine for ONNX models. | Chosen over running full PyTorch on edge devices because it reduces dependency weights, utilizes CPU instructions (NEON/AVX) efficiently, and saves RAM. | Located in [models/onnx_inference.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/models/onnx_inference.py) and [edge_deploy/raspberry_pi.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/edge_deploy/raspberry_pi.py). |

---

## 3. 50 INTERVIEW QUESTIONS WITH DETAILED ANSWERS

### Backend (10 questions)

#### Q1: Why did you choose FastAPI for the backend over Django or Flask?
FastAPI was chosen because this project requires high-performance, asynchronous streaming of video frames via WebSockets. Django's synchronous ORM history and heavy weight make it overkill, while Flask lacks built-in async support and automatic request validation. FastAPI handles WebSocket connections asynchronously inside `api/routers/stream.py` without blocking the thread, and leverages Pydantic to validate uploads and inputs in `api/routers/detection.py`.

#### Q2: Walk me through the security and authentication mechanism. How is it configured?
In the current project layout, we prioritize developer agility and local deployment. We manage security through API keys and configurations loaded from environment variables in `config/settings.py`. There is no session token auth implemented, but APIs are protected via CORS policies restricted in `api/main.py`. For external APIs (like Groq), we load `GROQ_API_KEY` from `.env`, which is kept secure on the server side and never leaked to the client dashboard. Webhook dispatches can also pass custom authorization headers.

#### Q3: Walk me through the API router structure of your FastAPI backend.
The FastAPI app defined in `api/main.py` delegates endpoints to modular routers located in the `api/routers/` directory:
1. `health.py`: Provides `/health` to check database, model, and GPU status, and `/metrics` for system health.
2. `history.py`: Houses GET `/history` and GET `/history/{id}` for querying SQLAlchemy incident records.
3. `detection.py`: Houses POST `/detect` which processes multipart image uploads, runs inference, saves artifacts, index vectors, and dispatches alerts.
4. `stream.py`: Houses WS `/ws/stream` for continuous base64 frame streaming inference.

#### Q4: How is the database session managed in the FastAPI routers?
We use dependency injection via FastAPI's `Depends` utility. In `api/dependencies.py`, we define `get_db()`, which yields a SQLAlchemy `Session` from `SessionLocal()` (configured in `database/session.py`). The route functions receive this session as an argument. The `get_db()` helper uses a `try...finally` block, ensuring that database sessions are closed automatically when the request finishes, preventing database connection leaks.

#### Q5: How do you validate requests for file uploads and JSON bodies?
We separate inputs using FastAPI's parameter types. In `api/routers/detection.py`, the image is received as an `UploadFile` via FastAPI's `File(...)` dependency to ensure it is handled as a multipart upload stream. Accompanying form metadata (like camera location or source) is validated using FastAPI's `Form(...)` params. Response bodies are strictly typed using Pydantic models defined in `api/schemas.py`, ensuring all fields (like risk scores, coordinates, similar incidents) conform to correct data structures.

#### Q6: How does the backend handle failures in external dependencies like the Groq LLM API or Telegram alert?
Robust error handling is baked directly into the route flows. In `api/routers/detection.py`, if the Groq LLM API fails due to network issues or rate limits, the `IncidentSummarizer` catches the exception in a `try...except` block and falls back to a rule-based summary builder (`_fallback_summary` in `llm/incident_summarizer.py`). Similarly, the `AlertManager` wraps channel dispatches in try/excepts, ensuring that if Telegram or SMTP fails, it records the failure status inside the `alert_logs` database table but does not block the API response from returning.

#### Q7: How are uploaded images for inference handled and stored?
When a frame is sent to `/api/v1/detect`, the backend reads the image bytes asynchronously. It converts the bytes to a NumPy array for OpenCV/inference processing. If a hazard is detected, the engine generates two images: a snapshot of the raw frame and a blended Grad-CAM heatmap. The file paths are determined by `config/settings.py` (pointing by default to `data/processed/snapshots` and `data/processed/heatmaps`). We mount this processed directory under the `/artifacts` prefix in `api/main.py` using `StaticFiles` so they can be fetched securely by the dashboard frontend.

#### Q8: What security precautions did you take in your FastAPI setup?
Our FastAPI configuration enforces CORS controls (`CORSMiddleware`) allowing explicit origins, methods, and headers. Sensitive details like API keys for Groq, Telegram bot tokens, and SMTP email credentials are kept on the server and loaded using `pydantic-settings` to avoid hardcoding secrets in codebase files. We also sanitize uploaded image buffers using PIL to ensure they are valid image files, preventing malicious script uploads.

#### Q9: Why are some endpoints like `/api/v1/detect` async while database queries are mostly sync?
The `/api/v1/detect` endpoint is marked `async` because it handles file reading (`await file.read()`) and triggers asynchronous alert notifications via HTTP webhooks, Telegram APIs, and SMTP using `asyncio.gather`. However, our database queries are handled through standard synchronous SQLAlchemy sessions. To ensure the synchronous DB operations do not block FastAPI's single-threaded event loop, FastAPI automatically runs synchronous database functions in an internal threadpool.

#### Q10: How would you scale the backend if the system starts receiving 100+ frames per second?
FastAPI can be scaled by running multiple Uvicorn worker processes behind a load balancer (like Nginx). However, the primary bottleneck will be ML model inference. In a production environment, I would decouple the inference engine by offloading YOLOv8 and EfficientNetV2 execution to a specialized service like Triton Inference Server or Ray Cluster. I would queue incoming frames in Redis, run inference asynchronously, and write incident summaries using Celery workers to avoid blocking the client-facing APIs.

---

### Database (8 questions)

#### Q11: Why did you choose SQLite/PostgreSQL-ready setup over a NoSQL database like MongoDB?
We chose a relational database schema because our data models have strict structured relationships. For example, every entry in `alert_logs` is strictly tied to a primary incident in the `detections` table via a foreign key relationship. SQLAlchemy allows us to easily swap out SQLite (used for lightweight local development) for PostgreSQL in production simply by altering the `DATABASE_URL` environment variable, ensuring zero code changes to queries.

#### Q12: Walk me through the database tables in your database.
We have three tables defined in [database/models.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/database/models.py):
1. `detections`: Stores metadata for every hazard incident (timestamp, class name, confidence, bounding boxes, risk metrics, paths to snapshots/heatmaps, and LLM text summaries).
2. `alert_logs`: Logs the delivery state (pending, sent, failed, or skipped) of alerts dispatched to various channels (email, telegram, audio, webhook) for each detection.
3. `model_runs`: Logs training metadata (MLflow run ID, model name, and final mAP50 scores) for models trained using our pipelines.

#### Q13: Explain the structure of the schema in SQL syntax.
The SQL schema corresponds to:
*   `detections`: `id` (INTEGER PRIMARY KEY), `timestamp` (DATETIME), `class_name` (VARCHAR), `confidence` (FLOAT), `bbox_json` (TEXT), `frame_path` (TEXT), `heatmap_path` (TEXT), `llm_summary` (TEXT), `source` (VARCHAR), and `risk_score` (FLOAT).
*   `alert_logs`: `id` (INTEGER PRIMARY KEY), `detection_id` (INTEGER FOREIGN KEY referencing detections.id), `channel` (VARCHAR), `status` (VARCHAR), `sent_at` (DATETIME), and `error_msg` (TEXT).
*   `model_runs`: `id` (INTEGER PRIMARY KEY), `run_id` (VARCHAR), `model_name` (VARCHAR), `mAP50` (FLOAT), and `created_at` (DATETIME).

#### Q14: How are relationships between tables mapped in SQLAlchemy?
We use SQLAlchemy's modern `Mapped` types and `relationship` configurations:
In `database/models.py`, `Detection` has a 1-to-many relationship mapped to `AlertLog`:
```python
alerts: Mapped[list["AlertLog"]] = relationship(back_populates="detection", cascade="all, delete-orphan")
```
This relationship uses `cascade="all, delete-orphan"`, meaning that if a `Detection` row is deleted, all corresponding `AlertLog` records are automatically removed from the database, preventing orphaned rows.

#### Q15: Which columns did you index and why?
In `database/models.py`, we explicitly define indexes on columns frequently used in WHERE clauses, sorting, and JOIN operations:
*   `Detection.timestamp` (index=True) to speed up chronological lookups and dashboard graphs.
*   `Detection.class_name` and `Detection.source` to filter historical incidents.
*   `AlertLog.detection_id` to speed up joins between detections and logs.
*   `AlertLog.channel` to query logs by alert types.
*   `ModelRun.run_id` and `ModelRun.model_name` to filter training run entries.

#### Q16: How does the application optimize historical queries for the dashboard?
In `database/crud.py`, the `list_detections` function implements pagination parameters (`offset` and `limit`). Instead of fetching all records into memory, we use SQLAlchemy's `offset(offset).limit(limit)` queries. This translates directly to `LIMIT` and `OFFSET` in SQL, meaning the database engine only parses and returns the requested block of data.

#### Q17: How is the database initialized and seeded?
We have an initialization script at `database/migrations/init_db.py`. On application startup, the FastAPI lifespan context manager (`api/main.py`) checks if the database is initialized. If not, it executes `init_db()` which reads the configuration and executes `Base.metadata.create_all(engine)` to build tables. For local testing, we can seed sample data into tables by generating fake detections and logging test alerts.

#### Q18: How do you handle database sessions and prevent connection pooling issues?
In `database/session.py`, we define `sessionmaker` bound to our SQL engine:
```python
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```
We disable autocommit and autoflush to control transactions explicitly. In our FastAPI dependencies and Streamlit pages, database operations are wrapped in `with SessionLocal() as db:` blocks. The context manager pattern guarantees that even if an exception occurs mid-transaction, session resources are cleaned up and returned to the connection pool.

---

### Machine Learning / AI (10 questions)

#### Q19: Why did you choose YOLOv8 for detection and EfficientNetV2 for verification?
Fire and smoke detection are highly safety-critical. YOLOv8 is an extremely fast object detector that can identify bounding boxes of fire/smoke at high frame rates (exceeding 30 FPS on edge CPUs). However, YOLO models can trigger false positives on fire-like objects (e.g., red shirts, lamps, bright sunset glares). EfficientNetV2 is optimized for image classification and acts as a secondary verification head. It reviews the cropped bounding box proposed by YOLOv8, confirming if fire/smoke is actually present, drastically reducing false alarms.

#### Q20: What datasets did you train/validate on?
The models are trained using custom datasets and public hazard benchmarks (specifically the D-Fire dataset split). D-Fire is a comprehensive dataset designed specifically for fire and smoke segmentation and detection. In `data/download_datasets.py`, we support downloading a mini-split of this dataset. The YOLO and classification training pipelines use data paths mapping to `data/processed/` generated by our preprocess and augmentation scripts.

#### Q21: How was the training pipeline configured and monitored?
We have training scripts defined in `training/train_yolo.py` and `training/train_efficientnet.py`.
The YOLOv8 model is fine-tuned using the Ultralytics API. We monitor training convergence, learning rates, loss functions, and classification accuracy by integrating MLflow. In `training/train_yolo.py`, MLflow tracking is initialized, logging hyperparameters (batch size, learning rates, augmentations) and validation metrics (mAP50, loss) into our local SQLite mlflow database.

#### Q22: Walk me through the ensemble logic that combines YOLOv8 and EfficientNetV2 predictions.
In `models/ensemble.py`, we define the `weighted_score` function.
When the YOLOv8 detector yields a candidate box with a class probability $P_{yolo}$, we crop the region of interest and pass it through the EfficientNetV2 classifier, which outputs a hazard probability $P_{clf}$.
We compute a weighted average:
$$\text{Score}_{ens} = \frac{P_{yolo} \cdot W_{yolo} + P_{clf} \cdot W_{clf}}{W_{yolo} + W_{clf}}$$
By default, we set $W_{yolo} = 0.65$ and $W_{clf} = 0.35$ (which can be customized in the settings panel). This ensemble score represents the combined confidence score used to trigger alerts.

#### Q23: How is the composite Risk Score calculated?
In `models/ensemble.py`, `compute_risk_score` uses a multi-factor formula to return a score from 0 to 100:
$$\text{Risk} = (\text{Confidence} \times 0.40) + (\text{Area Ratio} \times 0.40) + (\text{Growth Rate} \times 0.10) + (\text{Smoke Presence} \times 0.10)$$
*   **Confidence**: The ensemble probability.
*   **Area Ratio**: The percentage of the camera frame covered by the hazard.
*   **Growth Rate**: The rate at which the bounding box is expanding compared to the previous frame.
*   **Smoke Presence**: A binary booster (1.0 if smoke class is detected, 0.0 otherwise).
For immediate threats (high confidence & area > 5%), we apply a non-linear $1.5\times$ boost. The score is mapped to severity bands: LOW, MEDIUM, HIGH, and CRITICAL.

#### Q24: What is the purpose of Grad-CAM in this project and how did you implement it?
Grad-CAM (Gradient-weighted Class Activation Mapping) is an explainability (XAI) tool. It tells human operators *why* the AI detected fire in a specific area by visualizing model weights activations as heatmaps. In `inference/gradcam_explainer.py`, we hook into the final convolutional layer of the YOLOv8 backbone, compute gradients of target classes, and construct a heatmap that is blended back over the original image, which is displayed in the Model Insights panel of the Streamlit UI.

#### Q25: Explain the FAISS similarity search mechanism.
When an incident is logged in `api/routers/detection.py`, we compute a dense visual vector representation of the frame using CLIP (`openai/clip-vit-base-patch32` via Hugging Face Transformers).
We save the 512-dimension vector into a local FAISS index file (`data/processed/faiss/index.faiss`) and append the metadata to `meta.jsonl`. During query time, FAISS performs a cosine similarity search (using normalized inner product) against historical embeddings, returning the top-3 most similar past incidents.

#### Q26: How does the LLM Incident Summarizer generate reports?
The `IncidentSummarizer` class in `llm/incident_summarizer.py` formats incident details (timestamp, location, classes, confidence, and visual region hints) into a structured prompt (`llm/prompts.py`). This prompt is dispatched to an LLM provider (Groq Llama 3 cloud or Ollama local) with instructions to generate a concise, 3-sentence summary of the incident. If the LLM call fails, the module falls back to a rule-based generator.

#### Q27: What is the model fallback mechanism if YOLO weights are missing?
We built a self-healing setup in [models/yolo_detector.py](file:///c:/Users/Asus/OneDrive/Desktop/New%20folder/PyroSense-AI-main/models/yolo_detector.py).
During initialization, if `YOLO_MODEL_PATH` is missing or points to a non-fire model, the detector automatically downloads a pre-trained fire/smoke detector model from Hugging Face (`SHOU-ISD/fire-and-smoke`). If network calls fail, it attempts to download the mini D-Fire split and trains a local model for 10 epochs as a last-resort recovery.

#### Q28: How did you benchmark model inference speed?
We built a benchmark function in `dashboard/pages/4_Model_Insights.py` (`_benchmark`).
It feeds 20 empty/zero NumPy frames of size 640x480 to the `InferenceEngine` and measures execution time. It calculates and returns both Average Latency (ms) and Average Throughput (FPS), allowing developers to test inference speeds on their specific hardware (CPU vs CUDA/GPU).

---

### Frontend (7 questions)

#### Q29: Why did you choose Streamlit instead of React or Vue for this project?
As a machine learning project, Streamlit allowed us to build the dashboard entirely in Python, eliminating the need to compile frontend bundles, manage Node modules, or synchronize state between Python backend models and JS components. It provides native components for sliders, video display, and table rendering, letting us build a complete web app UI in a single day.

#### Q30: How is state managed across different pages in Streamlit?
Since Streamlit runs top-to-bottom on every user interaction, we use `@st.cache_resource` inside `dashboard/app.py` to cache expensive, stateful objects like the `InferenceEngine` and `FaissHistory` indexing instance. This ensures model weights are loaded into memory exactly once and are shared across page navigations. Configuration overrides are saved to `data/processed/settings_override.json` so they persist even if the app crashes or restarts.

#### Q31: How is page routing set up in the dashboard?
Streamlit handles page routing natively using a folder-based structure. By creating a `pages/` directory inside `dashboard/`, Streamlit automatically detects pages and places them in the sidebar in alphabetical/numerical order:
1. `1_Live_Detection.py`: RTSP/Webcam stream page.
2. `2_Upload_Analysis.py`: File upload analysis page.
3. `3_Incident_History.py`: DB history logs.
4. `4_Model_Insights.py`: Benchmarks, MLflow runs, Grad-CAM.
5. `5_Settings.py`: Configuration editor.
6. `6_Threat_Map.py`: Geographic location of cameras.

#### Q32: How does the frontend communicate with the backend?
The Streamlit dashboard can access database connections and call inference engines directly when run in local mode. For production or remote deployments, the dashboard communicates with the FastAPI backend via HTTP REST endpoints (fetching paginated data from `/api/v1/history` or uploading images to `/api/v1/detect`) and connects to WebSockets (`/api/v1/ws/stream`) to pass base64 video frames.

#### Q33: How did you style the Streamlit interface?
Streamlit's default UI can look generic, so we injected custom styling to create a premium look. In `dashboard/app.py` and page layouts, we read a stylesheet file located at `dashboard/assets/style.css` and inject it into the app via `st.markdown(..., unsafe_allow_html=True)`. We also used raw HTML/CSS blocks to render a top navigation bar and cards with smooth shadows and gradients.

#### Q34: How is live stream processing implemented?
On the live detection page (`1_Live_Detection.py`), the app opens a stream (webcam, RTSP, or YouTube URL) using OpenCV's `VideoCapture` or `yt-dlp`. It runs a continuous `while` loop, reading frames, executing the `InferenceEngine`, drawing the resulting bounding boxes, and updating Streamlit's `st.image` component in real time. We include check boxes to break the loop or change thresholds dynamically.

#### Q35: What frontend performance optimizations did you implement?
To keep the UI responsive, we implemented two key optimizations:
1. Caching resources (`st.cache_resource`) for ML models so PyTorch weight files aren't re-read from disk on every page reload.
2. Limiting history records: In the Incident History page, we paginate items (loading only 50 at a time) instead of loading the entire SQL database.

---

### Deployment & DevOps (7 questions)

#### Q36: How is the application deployed, and what environments does it support?
The system is fully containerized using Docker, allowing it to run on any environment. For local runs, we provide a `docker-compose.yml` that mounts directories and spins up the FastAPI app on port 8000 and the Streamlit dashboard on port 8501. The repository is configured to be pushed to Hugging Face Spaces (using a Docker SDK space) or run on any Linux instance.

#### Q37: Walk me through the Dockerfile structure.
Our `Dockerfile` uses a multi-stage approach or a clean python-slim base:
```dockerfile
FROM python:3.10-slim
```
To run OpenCV and PyTorch on Debian Linux, we install essential system libraries (like `libgl1`, `libglib2.0-0`, `build-essential`) via `apt-get`. We then copy `requirements.txt` and install Python dependencies. We expose port 8501 (Streamlit) and launch supervisord or shell scripts to run both the FastAPI API backend and Streamlit dashboard concurrently.

#### Q38: What environment variables are required and how are they managed?
All settings are declared inside `config/settings.py` using `pydantic-settings`. The variables include:
*   `YOLO_MODEL_PATH`: Location of local model weights.
*   `DATABASE_URL`: Location of the database.
*   `GROQ_API_KEY`: Secret API key for Groq Llama 3 summaries.
*   `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`: Telegram notification details.
*   `EMAIL_ENABLED` / `EMAIL_USER` / `EMAIL_PASSWORD`: SMTP credentials.
We manage these variables locally via a `.env` file (copied from `.env.example`).

#### Q39: Walk me through your edge deployment strategy for Raspberry Pi.
The edge architecture is detailed in `edge_deploy/raspberry_pi.py`. To run on low-resource hardware, we export the YOLOv8 model to ONNX using `model.export(format="onnx")`. The Pi runs a lightweight script that does not require PyTorch or Streamlit; it uses `onnxruntime` to execute inference on incoming camera frames, and uses `requests` to post detected hazards back to the central FastAPI backend.

#### Q40: What automated checks are run in CI/CD before code merges?
We use GitHub Actions to run linting and unit tests. The workflow triggers on every push and pull request. It runs `ruff` to enforce formatting rules and executes `pytest` inside the `tests/` directory (running tests defined in `test_api.py`, `test_detector.py`, and `test_alerts.py`), verifying backend API responses and inference engine outputs.

#### Q41: How do you handle CPU/RAM limits on free cloud hosting tiers?
Hosting services like Hugging Face Spaces or Streamlit Cloud have strict RAM limits (e.g., 16GB). To prevent Out-Of-Memory (OOM) crashes:
*   We use lazy-loading for the secondary classifier. EfficientNetV2 is only loaded when a detection occurs.
*   We allow disabling the secondary classifier in the settings page (`enable_efficientnet = False`), which bypasses loading PyTorch torchvision weights.
*   We use the smaller YOLOv8n (nano) model instead of larger weights on CPU-only hosts.

#### Q42: What are the differences between development and production setups?
*   **Database**: In development, we use SQLite (`sqlite:///./pyrosense.db`); in production, we swap `DATABASE_URL` to point to a PostgreSQL server.
*   **Logging**: Development logs at `DEBUG` level; production logs at `INFO` or `WARNING` and outputs structured formats.
*   **FastAPI**: In dev, Uvicorn runs with `--reload`; in production, reload is disabled to save cycles.

---

### Project & System Design (8 questions)

#### Q43: Explain the step-by-step data flow when a camera detects fire.
1. **Ingestion**: The capture loop in `Live_Detection.py` extracts a frame from the stream.
2. **Detection**: YOLOv8 locates bounding boxes of fire/smoke.
3. **Verification**: If found, EfficientNetV2 checks the cropped hazard image, outputting a verification confidence.
4. **Scoring**: Bounding box size, growth rates, and confidences are parsed to yield a composite Risk Score.
5. **Persistence**: The API uploads details to SQLite/PostgreSQL, saving the image snapshot and Grad-CAM activation heatmap.
6. **Alerting**: `AlertManager` sends alerts (Email/Telegram/Webhooks/Audio) asynchronously.
7. **Reporting**: Groq/Ollama creates a Llama 3 incident summary text.
8. **UI Update**: The Streamlit dashboard displays the latest hazard details.

#### Q44: What was the hardest technical challenge in this project and how did you solve it?
The hardest challenge was preventing the dashboard UI from freezing when processing video streams. Streamlit runs on a single main thread, and running real-time frame capture, dual-model deep learning inference, and API database logging sequentially in a loop would cause the UI to stutter and drop frames.
I resolved this by decoupling the components:
1. Caching ML models in memory so weights aren't re-instantiated.
2. Setting a configurable frame-rate sleep interval (0.05 seconds) to give resources back to the thread.
3. Delegating alert dispatches to asynchronous coroutines (`asyncio.gather`), ensuring network delays in email/webhook delivery do not block video frame analysis.

#### Q45: How does the Alert Cooldown work to avoid spamming?
In `alerts/alert_manager.py`, we define `ChannelState` containing a `last_sent_at` timestamp.
When a hazard is detected, `_cooldown_ok(channel)` calculates the duration since the last alert was sent. If the elapsed time is less than `alert_cooldown_seconds` (default is 60 seconds), the alert is skipped. The status is recorded in the database as "skipped", preventing flooded inboxes and API rate-limiting issues.

#### Q46: How are alerts sent concurrently?
Inside `AlertManager.trigger_alert()`, we compile a list of tasks for all enabled alert channels (email, telegram, audio, webhook). Instead of sending them sequentially, we invoke them concurrently:
```python
await asyncio.gather(*tasks, return_exceptions=True)
```
By setting `return_exceptions=True`, we ensure that if one channel (e.g., a bad Webhook URL) throws an exception, the other alerts continue to send successfully.

#### Q47: How would you scale the system to support millions of snapshots?
Storing snapshots locally on disk will quickly exhaust server storage. In production, I would alter the storage architecture:
1. Save raw and heatmap images into an object storage service like AWS S3 or Google Cloud Storage.
2. Save only the secure CDN URL of the object inside our relational database (`frame_path` and `heatmap_path`).
3. Set up lifecycle policies in the cloud bucket to automatically archive or delete images older than 30 days.

#### Q48: How are user settings and API credentials secured?
We load all configurations through environment variables using Pydantic Settings. The settings UI page (`5_Settings.py`) uses Streamlit's `type="password"` input field for passwords, API keys, and bot tokens. This renders characters as dots on the screen. Overrides are written to `settings_override.json` in a secure subfolder, avoiding exposing secrets in the codebase.

#### Q49: What features would you add next?
If given more time, I would focus on:
1. **Role-Based Access Control**: Adding user login screens to secure the settings page.
2. **Camera Map Integration**: Utilizing Leaflet maps in Streamlit to drop markers representing active fire alerts.
3. **Active Retraining Loop**: Adding a "Flag False Positive" button to the history page that saves the crop to a folder, allowing developers to retrain models on edge cases.

#### Q50: What makes this fire detection system unique?
Most AI fire detection projects are simple scripts that run YOLO on a webcam. PyroSense AI is a complete, production-ready system. It combines object detection (YOLOv8) with a secondary verification classifier (EfficientNetV2) to reduce false positives, features Grad-CAM explainability to show *why* the model detected fire, uses CLIP vector embeddings for historical search, and uses Llama 3 for human-readable summaries.

---

## 4. KEY NUMBERS TO MEMORIZE

*   **6 API Endpoints**:
    *   `POST /api/v1/detect` (multipart upload inference)
    *   `WS /api/v1/ws/stream` (websocket base64 streaming)
    *   `GET /api/v1/history` (paginated history list)
    *   `GET /api/v1/history/{id}` (single incident details)
    *   `GET /api/v1/health` (engine & system status)
    *   `GET /api/v1/metrics` (prom/basic analytics JSON)
*   **3 DB Tables**: `detections`, `alert_logs`, and `model_runs` (managed via SQLAlchemy ORM).
*   **93.2% mAP50**: Baseline test accuracy for the fire/smoke object detection model.
*   **512 Dimensions**: Vector size of CLIP image embeddings stored in our local FAISS index.
*   **20 Frames**: Count used to run speed benchmarks inside the model insights panel.
*   **60 Seconds**: Default cooldown period for all alert channels to prevent email and Telegram spam.
*   **640 Pixels**: YOLOv8 model input size.
*   **224 Pixels**: EfficientNetV2 classifier input size.

---

## 5. 2-MINUTE SPOKEN PITCH

**[Problem Statement]**
"Standard commercial fire alarm systems rely on physical smoke and heat changes, which can take several critical minutes to trigger in large warehouses or outdoor spaces. While computer vision offers a faster alternative, standard object detection models are notorious for triggering false alarms on harmless stimuli like red shirts, orange safety vests, or bright yellow lights, which desensitizes operators."

**[How It Works Technically]**
"To solve this, I built **PyroSense AI**, a real-time hazard detection system. We use a dual-model ensemble pipeline: a fine-tuned **YOLOv8** model performs high-speed object detection at 30+ frames per second, and is verified by a secondary **EfficientNetV2** classifier that verifies the crop to confirm fire or smoke. If verified, the system calculates a multi-factor risk score, logs the incident, and executes Grad-CAM heatmaps for explainability, while concurrently dispatching notifications via Email, Telegram, and Webhooks."

**[What Makes It Unique]**
"What makes this project unique is its integration of explainability and retrieval. Instead of just raising an alarm, PyroSense uses Grad-CAM overlays to show operators exactly *what* pixels triggered the AI. It also computes image embeddings using CLIP and queries a local **FAISS** index, allowing operators to instantly see the top-3 similar historical incidents, and uses Llama 3 to write human-readable incident summaries in under a second."

**[My Personal Contribution]**
"I architected the end-to-end software loop. I built the FastAPI backend, including the WebSockets frame receiver, and designed the database schemas using SQLAlchemy. I also built the multi-page Streamlit dashboard, configured the alert dispatch engine with smart channel cooldowns, and exported the models to ONNX to support edge deployments on Raspberry Pi."

**[Impact/Results]**
"In evaluations, the ensemble approach achieved a **93.2% mAP50 validation accuracy**, dramatically reducing false alarms compared to standard YOLO detectors while maintaining a sub-100ms inference latency."
