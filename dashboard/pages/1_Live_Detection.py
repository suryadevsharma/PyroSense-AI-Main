"""Streamlit page: Live detection feed (mission-control UI)."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import efficientnet_enabled, get_settings
from dashboard.app import get_engine
from dashboard.components.metrics_panel import render_metrics_panel
from database import crud
from database.session import SessionLocal
from utils.logger import logger


def _load_css() -> None:
    css_path = Path("dashboard/assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _risk_label(risk: float) -> Tuple[str, str]:
    if risk >= 90:
        return "CRITICAL", "#DC2626"
    if risk >= 70:
        return "HIGH", "#E53E3E"
    if risk >= 40:
        return "MEDIUM", "#F59E0B"
    if risk >= 15:
        return "LOW", "#10B981"
    return "SAFE", "#10B981"


def _hud_overlay(frame_bgr: np.ndarray, fps: float, risk_score: float) -> np.ndarray:
    """Overlay HUD: FPS (top-left), timestamp (top-right), risk bar (bottom)."""

    try:
        import cv2
    except Exception:
        return frame_bgr

    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cv2.putText(out, f"FPS: {fps:.1f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 106), 2)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(out, ts, (max(12, w - tw - 12), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 146, 165), 2)

    # Risk bar at bottom
    bar_y1 = h - 14
    bar_y2 = h - 6
    cv2.rectangle(out, (0, bar_y1), (w, bar_y2), (30, 35, 55), -1)
    fill = int(np.clip(risk_score, 0, 100) / 100.0 * w)
    color = (0, 212, 106)
    if risk_score >= 90:
        color = (0, 0, 255)
    elif risk_score >= 70:
        color = (0, 69, 255)
    elif risk_score >= 40:
        color = (0, 184, 255)
    cv2.rectangle(out, (0, bar_y1), (fill, bar_y2), color, -1)
    return out


def _log_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    dets = payload.get("detections") or []
    top = dets[0] if dets else {}
    cls = str(top.get("class_name", payload.get("primary_class", "none")))
    conf = float(top.get("score", payload.get("ensemble_conf", 0.0)))
    risk = payload.get("risk") or {}
    return {
        "timestamp": str(payload.get("timestamp", "")),
        "class_name": cls,
        "confidence": conf,
        "risk_score": float(risk.get("score", 0.0)),
        "risk_severity": str(risk.get("severity", "LOW")),
    }


def main() -> None:
    try:
        st.set_page_config(page_title="PyroSense AI", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
    _load_css()

    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;padding:0 0 24px;border-bottom:1px solid #E5E7EB;">'
        '<div>'
        '<h1 style="font-family:JetBrains Mono,monospace;font-size:22px;color:#111827;margin:0;font-weight:600;">LIVE DETECTION FEED</h1>'
        '<p style="font-family:monospace;font-size:11px;color:#9CA3AF;margin:4px 0 0;text-transform:uppercase;letter-spacing:0.1em;">Real-time fire &amp; smoke monitoring</p>'
        '</div>'
        '<div id="live-indicator" style="display:flex;align-items:center;gap:8px;padding:8px 16px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:20px;">'
        '<span class="status-dot active"></span>'
        '<span style="font-family:monospace;font-size:12px;color:#10B981;font-weight:600;text-transform:uppercase;">MONITORING</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    if "live_log" not in st.session_state:
        st.session_state.live_log = []
    if "detections_session" not in st.session_state:
        st.session_state.detections_session = 0
    if "peak_conf" not in st.session_state:
        st.session_state.peak_conf = 0.0
    if "last_alert" not in st.session_state:
        st.session_state.last_alert = {"channel": "-", "status": "-", "sent_at": "-"}
    if "_last_db_save" not in st.session_state:
        st.session_state._last_db_save = 0.0

    try:
        eng = get_engine()
        if efficientnet_enabled():
            st.caption(
                "EfficientNet verifier loads on the **first frame that contains a detection** (one-time download possible)."
            )
        if not getattr(eng.yolo, "model_ready", True):
            st.warning("Fire/smoke model not ready yet. It may be downloading or training. Try again in a minute.")
    except Exception as e:
        st.warning(f"Model initialization failed: {e}")
        return

    left, right = st.columns([0.7, 0.3], gap="large")

    with left:
        # IMPORTANT: Do NOT use st.tabs() for source selection, because Streamlit executes all tab blocks,
        # which can overwrite state and show the wrong source. Use a single explicit selector instead.
        source_kind = st.radio(
            "SOURCE",
            ["WEBCAM", "RTSP STREAM", "UPLOAD VIDEO", "DEMO MODE"],
            horizontal=True,
            label_visibility="collapsed",
            key="live_source_kind",
        )

        rtsp_url = ""
        cam_index = 0
        upload_file = None
        if source_kind == "WEBCAM":
            cam_index = int(st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1))
        elif source_kind == "RTSP STREAM":
            rtsp_url = st.text_input("RTSP URL", value="rtsp://")
        elif source_kind == "UPLOAD VIDEO":
            upload_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], key="live_upload_video")
        else:
            st.caption("Demo mode loops through sample images (and synthetic fire/smoke frames if available).")

        frame_slot = st.empty()

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("START"):
                st.session_state.live_running = True
                st.session_state._video_frame_idx = 0
        with b2:
            if st.button("STOP"):
                st.session_state.live_running = False
                st.session_state._video_frame_idx = 0
        with b3:
            if st.button("SCREENSHOT"):
                st.session_state._take_screenshot = True
        with b4:
            if st.button("RECORD"):
                st.session_state._record = not bool(st.session_state.get("_record", False))

        if not st.session_state.live_running:
            frame_slot.info("Press START to begin monitoring.")

    with right:
        risk_card = st.empty()
        metrics_row = st.empty()
        log_box = st.empty()
        alerts_box = st.empty()

    alert_banner = st.empty()

    fps_window: List[float] = []
    video_writer = None

    def update_right(payload: Dict[str, Any]) -> None:
        risk = payload.get("risk") or {}
        risk_score = float(risk.get("score", 0.0))
        label, color = _risk_label(risk_score)

        risk_cls = 'critical' if risk_score >= 90 else 'high' if risk_score >= 70 else 'medium' if risk_score >= 40 else 'low'
        card_cls = 'pyro-card alert' if label in ['HIGH', 'CRITICAL'] else 'pyro-card'
        risk_card.markdown(
            f'<div class="{card_cls}">'
            f'<div style="font-family:monospace;font-size:10px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.12em;">RISK LEVEL</div>'
            f'<div style="font-family:monospace;font-size:28px;color:{color};font-weight:800;margin-top:6px;">{label}</div>'
            f'<div style="font-family:monospace;font-size:10px;color:#9CA3AF;margin-top:2px;">SCORE: {int(risk_score)}/100</div>'
            f'<div class="risk-bar" style="margin-top:10px;">'
            f'<div class="risk-fill {risk_cls}" style="width:{min(100.0, max(0.0, risk_score))}%;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        with metrics_row:
            render_metrics_panel(
                fps=float(payload.get("_fps", 0.0)),
                detections=int(st.session_state.detections_session),
                confidence=float(st.session_state.peak_conf),
                inference_ms=float(payload.get("inference_time_ms", 0.0)),
            )

        rows = st.session_state.live_log[:10]
        log_html = ""
        for r in rows:
            cls = str(r.get("class_name", "none")).lower()
            bg = "rgba(229,62,62,0.06)" if "fire" in cls else "rgba(75,85,99,0.06)"
            border = "rgba(229,62,62,0.2)" if "fire" in cls else "#E5E7EB"
            badge_cls = "fire" if "fire" in cls else ("smoke" if "smoke" in cls else "safe")
            ts_display = r.get('timestamp', '').replace('T', ' ')[5:19]
            conf_pct = float(r.get('confidence', 0)) * 100
            risk_val = float(r.get('risk_score', 0))
            log_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:10px 12px;margin-bottom:8px;background:{bg};'
                f'border:1px solid {border};border-radius:10px;">'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="font-family:monospace;font-size:10px;color:#9CA3AF;">{ts_display}</span>'
                f'<span class="detection-badge {badge_cls}">{str(r.get("class_name", "")).upper()}</span>'
                f'</div>'
                f'<div style="font-family:monospace;font-size:11px;color:#4B5563;">'
                f'{conf_pct:.0f}% &bull; RISK {risk_val:.0f}</div></div>'
            )

        empty_msg = '<div style="font-family:monospace;color:#9CA3AF;font-size:11px;">No detections yet.</div>'
        log_box.markdown(
            f'<div class="pyro-card">'
            f'<div style="font-family:monospace;font-size:10px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">DETECTION LOG (LAST 10)</div>'
            f'<div style="max-height:320px;overflow:auto;">{log_html or empty_msg}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Query DB for latest alert status
        alert_info_html = ""
        try:
            from database.models import AlertLog, Detection
            with SessionLocal() as db:
                latest_det = db.query(Detection).order_by(Detection.id.desc()).first()
                if latest_det:
                    logs = db.query(AlertLog).filter(AlertLog.detection_id == latest_det.id).all()
                    
                    email_log = next((l for l in logs if l.channel == "email"), None)
                    tg_log = next((l for l in logs if l.channel == "telegram"), None)
                    wh_log = next((l for l in logs if l.channel == "webhook"), None)
                    
                    def get_status_str(log_item):
                        if not log_item:
                            return "<span style='color:#9CA3AF;'>Disabled / Skipped</span>"
                        if log_item.status == "sent":
                            return "<span style='color:#10B981;'>✅ Delivered</span>"
                        if log_item.status == "failed":
                            return f"<span style='color:#EF4444;'>❌ Failed</span> <span style='font-size:9px;color:#EF4444;'>({log_item.error_msg or 'Unknown Error'})</span>"
                        if log_item.status == "skipped":
                            return f"<span style='color:#F59E0B;'>⚠️ Skipped</span> <span style='font-size:9px;color:#9CA3AF;'>({log_item.error_msg or 'cooldown'})</span>"
                        return f"<span style='color:#6B7280;'>{log_item.status}</span>"

                    email_status = get_status_str(email_log)
                    tg_status = get_status_str(tg_log)
                    wh_status = get_status_str(wh_log)
                    
                    ai_summary = "✅ Generated" if (latest_det.llm_summary and latest_det.llm_summary != "Agent analysis in progress...") else "⏳ Pending"
                    db_status = f"✅ Saved (ID #{latest_det.id})"
                    ts_str = latest_det.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                    
                    alert_info_html = (
                        f"<b>Last Event:</b> {ts_str}<br/>"
                        f"<b>Type:</b> {latest_det.class_name.upper()} ({latest_det.confidence:.2%})<br/>"
                        f"<hr style='margin: 8px 0; border: none; border-top: 1px solid rgba(0,0,0,0.06);'/>"
                        f"<b>Email:</b> {email_status}<br/>"
                        f"<b>Telegram:</b> {tg_status}<br/>"
                        f"<b>Webhook:</b> {wh_status}<br/>"
                        f"<b>AI Summary:</b> {ai_summary}<br/>"
                        f"<b>Database Log:</b> {db_status}"
                    )
                else:
                    alert_info_html = "<div style='color:#9CA3AF;'>No alerts triggered yet.</div>"
        except Exception as e:
            alert_info_html = f"<div style='color:#EF4444;'>Error loading status: {e}</div>"

        alerts_box.markdown(
            f'<div class="pyro-card">'
            f'<div style="font-family:monospace;font-size:10px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">ACTIVE ALERTS STATUS</div>'
            f'<div style="font-family:monospace;font-size:11px;color:#4B5563;line-height:1.7;">'
            f'{alert_info_html}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    def frame_iter_demo():
        from inference.stream_processor import StreamProcessor

        sp = StreamProcessor(samples_dir=str(get_settings().data_dir / "samples"))
        for sf in sp.frames("demo", demo_mode=True):
            if not st.session_state.live_running:
                break
            yield sf.frame_bgr
            time.sleep(0.12)

    def frame_iter():
        if source_kind == "UPLOAD VIDEO":
            if upload_file is None:
                saved_path = st.session_state.get("_uploaded_video_path")
                if not saved_path or not Path(saved_path).exists():
                    return []
                dst = Path(saved_path)
            else:
                import tempfile

                up_dir = Path("data/processed/uploads")
                up_dir.mkdir(parents=True, exist_ok=True)
                dst = up_dir / upload_file.name
                if st.session_state.get("_uploaded_video_name") != upload_file.name or not dst.exists():
                    dst.write_bytes(upload_file.getvalue())
                    st.session_state._uploaded_video_name = upload_file.name
                    st.session_state._uploaded_video_path = str(dst)

            import cv2

            cap = cv2.VideoCapture(str(dst))
            if not cap.isOpened():
                st.error("Unable to open your uploaded video file.")
                return []

            original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            st.session_state._upload_video_fps = original_fps
            skip = max(1, int(round(original_fps / 8.0)))
            frame_interval = skip / original_fps

            saved_pos = int(st.session_state.get("_video_frame_idx", 0))
            if saved_pos > 0:
                for _ in range(saved_pos):
                    if not cap.grab():
                        break

            try:
                frame_idx = saved_pos
                end_of_video = False
                while cap.isOpened() and st.session_state.live_running:
                    t0 = time.perf_counter()

                    fr = None
                    end_of_video = False
                    for _ in range(skip):
                        ok, frame = cap.read()
                        if not ok:
                            end_of_video = True
                            break
                        fr = frame
                        frame_idx += 1

                    if fr is None:
                        break

                    st.session_state._video_frame_idx = frame_idx
                    yield fr

                    if end_of_video:
                        break

                    elapsed = time.perf_counter() - t0
                    wait = frame_interval - elapsed
                    if wait > 0:
                        time.sleep(wait)
            finally:
                cap.release()
                if end_of_video or not st.session_state.live_running:
                    st.session_state._video_frame_idx = 0
            return

        if source_kind == "RTSP STREAM":
            import cv2

            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                st.warning("Unable to open RTSP stream. Try DEMO MODE.")
                return []
            while cap.isOpened() and st.session_state.live_running:
                ok, fr = cap.read()
                if not ok:
                    break
                yield fr
            cap.release()
            return

        if source_kind == "WEBCAM":
            import cv2

            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.warning("Webcam not available. Switching to DEMO MODE.")
                yield from frame_iter_demo()
                return
            while cap.isOpened() and st.session_state.live_running:
                ok, fr = cap.read()
                if not ok:
                    break
                yield fr
            cap.release()
            return

        yield from frame_iter_demo()

    if st.session_state.live_running:
        try:
            import cv2
        except Exception:
            cv2 = None

        infer_every = 1
        frame_count = 0
        last_payload = None
        last_risk_score = 0.0

        t_last = time.perf_counter()
        for fr in frame_iter():
            if not st.session_state.live_running:
                break

            try:
                if frame_count == 0 and source_kind == "UPLOAD VIDEO":
                    upload_fps = float(st.session_state.get("_upload_video_fps", 0))
                    if upload_fps > 0:
                        infer_every = max(1, int(round(upload_fps / 5.0)))

                frame_count += 1
                now = time.perf_counter()
                dt = max(1e-6, now - t_last)
                t_last = now
                fps = 1.0 / dt
                fps_window.append(fps)
                fps_window = fps_window[-20:]
                fps_smooth = float(sum(fps_window) / len(fps_window))

                run_inference = (frame_count % infer_every == 0) or last_payload is None

                if run_inference:
                    payload = eng.detect_image(fr)
                    payload["_fps"] = fps_smooth
                    last_payload = payload

                    dets = payload.get("detections") or []
                    if dets:
                        st.session_state.detections_session += 1
                        confs = [float(d.get("score", 0.0)) for d in dets if isinstance(d, dict)]
                        st.session_state.peak_conf = max(float(st.session_state.peak_conf), max(confs) if confs else 0.0)
                        st.session_state.live_log = [_log_row(payload)] + st.session_state.live_log
                        st.session_state.live_log = st.session_state.live_log[:50]

                        db_cooldown = 3.0
                        now_mono = time.perf_counter()
                        if now_mono - st.session_state._last_db_save >= db_cooldown:
                            st.session_state._last_db_save = now_mono
                            try:
                                top_det = dets[0] if dets else {}
                                boxes = [
                                    tuple(d.get("bbox_xyxy", [0, 0, 0, 0]))
                                    for d in dets if isinstance(d, dict)
                                ]
                                risk_info = payload.get("risk") or {}
                                with SessionLocal() as db:
                                    crud.create_detection(
                                        db,
                                        timestamp=datetime.now(timezone.utc),
                                        class_name=str(payload.get("primary_class", "unknown")),
                                        confidence=float(payload.get("ensemble_conf", 0.0)),
                                        boxes_xyxy=boxes,
                                        frame_path=None,
                                        heatmap_path=None,
                                        llm_summary=None,
                                        source=source_kind,
                                        risk_score=float(risk_info.get("score", 0.0)),
                                    )
                            except Exception as db_err:
                                logger.warning(f"DB save failed: {db_err}")

                    annotated = payload.get("_annotated_frame", fr)
                    risk = payload.get("risk") or {}
                    last_risk_score = float(risk.get("score", 0.0))
                    frame_show = _hud_overlay(annotated, fps=fps_smooth, risk_score=last_risk_score)
                else:
                    payload = last_payload
                    payload["_fps"] = fps_smooth
                    dets = payload.get("detections") or []
                    frame_show = _hud_overlay(fr, fps=fps_smooth, risk_score=last_risk_score)

                if st.session_state.get("_take_screenshot") and cv2 is not None:
                    st.session_state._take_screenshot = False
                    out_dir = Path(get_settings().snapshots_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
                    cv2.imwrite(str(out_dir / name), frame_show)
                    st.toast(f"📸 Screenshot saved: {name}")

                recording = bool(st.session_state.get("_record", False))
                if recording and cv2 is not None:
                    if video_writer is None:
                        out_dir = Path("data/processed/recordings")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        h, w = frame_show.shape[:2]
                        video_writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (w, h))
                        st.toast("🔴 Recording started...")
                    video_writer.write(frame_show)
                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        st.toast("✅ Recording saved to data/processed/recordings/")

                with left:
                    frame_slot.image(frame_show, channels="BGR")

                if run_inference:
                    fire_conf = 0.0
                    for d in dets:
                        if not isinstance(d, dict):
                            continue
                        if "fire" in str(d.get("class_name", "")).lower():
                            fire_conf = max(fire_conf, float(d.get("score", 0.0)))

                    if fire_conf > 0.7:
                        alert_ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                        alert_banner.markdown(
                            f'<div style="background:#FEF2F2;border:2px solid #E53E3E;border-radius:12px;padding:16px 24px;margin:16px 0;animation:pulse-border 1s infinite;display:flex;align-items:center;justify-content:space-between;">'
                            f'<div style="display:flex;align-items:center;gap:12px;">'
                            f'<span style="font-size:24px;">&#x1F525;</span>'
                            f'<div>'
                            f'<div style="font-family:monospace;font-size:16px;color:#E53E3E;font-weight:700;text-transform:uppercase;">FIRE DETECTED</div>'
                            f'<div style="font-family:monospace;font-size:12px;color:#4B5563;">Confidence: {fire_conf*100:.0f}% &mdash; Zone: local &mdash; {alert_ts}</div>'
                            f'</div></div>'
                            f'<div style="font-family:monospace;font-size:11px;color:#10B981;">&#x2713; ALERTS DISPATCHED</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        alert_banner.empty()

                    update_right(payload)

            except Exception as e:
                logger.warning(f"Frame processing error (skipping): {e}")
                continue

        if video_writer is not None:
            video_writer.release()


if __name__ == "__main__":
    main()

