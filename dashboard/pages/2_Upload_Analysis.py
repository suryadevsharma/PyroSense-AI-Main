"""Streamlit page: Upload analysis (mission-control UI)."""

from __future__ import annotations

import sys
import json
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.app import get_engine, get_summarizer
from database import crud
from database.session import SessionLocal
from inference.gradcam_explainer import GradCamExplainer
from utils.image_utils import pil_to_bgr
from utils.logger import logger


def _load_css() -> None:
    css_path = Path("dashboard/assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _simple_pdf_bytes(title: str, body: str) -> bytes:
    """Create a minimal PDF (no external deps)."""

    text = (title + "\n\n" + body).replace("\r", "")
    # Very small, safe PDF generator (single page, Helvetica).
    lines = [l[:120] for l in text.split("\n")]
    y = 760
    content = "BT\n/F1 12 Tf\n72 780 Td\n"
    for line in lines[:55]:
        esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content += f"0 -14 Td ({esc}) Tj\n"
        y -= 14
    content += "ET\n"
    stream = content.encode("latin-1", errors="ignore")
    parts: List[bytes] = []
    parts.append(b"%PDF-1.4\n")
    xref = []

    def obj(n: int, data: bytes) -> None:
        xref.append(sum(len(p) for p in parts))
        parts.append(f"{n} 0 obj\n".encode() + data + b"\nendobj\n")

    obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
    obj(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    obj(5, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    xref_start = sum(len(p) for p in parts)
    parts.append(b"xref\n0 6\n0000000000 65535 f \n")
    for off in xref:
        parts.append(f"{off:010d} 00000 n \n".encode())
    parts.append(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n")
    parts.append(str(xref_start).encode() + b"\n%%EOF\n")
    return b"".join(parts)


def main() -> None:
    try:
        st.set_page_config(page_title="PyroSense AI", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
    _load_css()

    st.markdown(
        '<div style="padding:0 0 24px;border-bottom:1px solid #E5E7EB;margin-bottom:18px;">'
        '<div style="font-family:JetBrains Mono,monospace;font-size:22px;color:#111827;font-weight:700;">UPLOAD ANALYSIS</div>'
        '<div style="font-family:monospace;font-size:11px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.1em;margin-top:6px;">Analyze images and videos for fire and smoke</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="pyro-card" style="border:1px dashed rgba(229,62,62,0.25);background:rgba(229,62,62,0.03);">'
        '<div style="font-family:monospace;font-size:11px;color:#4B5563;">Drag &amp; drop an image or video below. Supported: JPG/PNG/MP4/AVI/MOV</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    up = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], label_visibility="collapsed")
    if up is None:
        return

    eng = get_engine()
    summarizer = get_summarizer()
    expl = GradCamExplainer(yolo_model=getattr(eng.yolo, "model", None) or getattr(eng.yolo, "_model", None))

    suffix = Path(up.name).suffix.lower()
    raw = up.getvalue()

    if suffix in {".jpg", ".jpeg", ".png"}:
        from PIL import Image

        frame = pil_to_bgr(Image.open(BytesIO(raw)).convert("RGB"))
        det = eng.yolo.detect_image(frame)
        payload = eng.detect_image(frame)
        llm = summarizer.summarize(det, location="Upload")
        three = expl.generate_heatmap(frame)

        if det.scores:
            try:
                risk_info = payload.get("risk") or {}
                with SessionLocal() as db:
                    crud.create_detection(
                        db,
                        timestamp=datetime.now(timezone.utc),
                        class_name=str(payload.get("primary_class", "unknown")),
                        confidence=float(payload.get("ensemble_conf", 0.0)),
                        boxes_xyxy=[(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in det.boxes],
                        frame_path=None,
                        heatmap_path=None,
                        llm_summary=llm,
                        source=f"upload:{up.name}",
                        risk_score=float(risk_info.get("score", 0.0)),
                    )
            except Exception as e:
                logger.warning(f"DB save failed for upload: {e}")

        tabs = st.tabs(["DETECTION", "HEATMAP", "REPORT"])
        with tabs[0]:
            c1, c2 = st.columns(2)
            c1.image(frame, channels="BGR")
            c2.image(det.annotated_frame, channels="BGR")
        with tabs[1]:
            st.image(three, channels="BGR")
        with tabs[2]:
            st.markdown(
                f'<div class="pyro-card alert">'
                f'<div style="font-family:monospace;font-size:10px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.12em;">AI INCIDENT REPORT</div>'
                f'<div style="font-family:monospace;font-size:12px;color:#4B5563;margin-top:10px;line-height:1.7;">{llm}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Detection results table (with Explain action)
        rows: List[Dict[str, Any]] = []
        for b, s, cn in zip(det.boxes, det.scores, det.class_names):
            risk = float(payload.get("risk", {}).get("score", 0.0))
            sev = str(payload.get("risk", {}).get("severity", "LOW"))
            rows.append({"Class": cn, "Confidence": float(s), "Risk Level": sev, "Bounding Box": list(map(lambda x: round(float(x), 1), b))})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=240)

        st.markdown("### Actions")
        for i, r in enumerate(rows):
            cols = st.columns([2, 1, 1, 2])
            cols[0].markdown(f"<span class='detection-badge {'fire' if 'fire' in r['Class'].lower() else 'smoke'}'>{r['Class']}</span>", unsafe_allow_html=True)
            cols[1].write(f"{r['Confidence']*100:.1f}%")
            cols[2].write(r["Risk Level"])
            if cols[3].button("Explain", key=f"explain_{i}"):
                prompt = (
                    "Explain this detection in 2 sentences for a safety operator. "
                    f"Class={r['Class']}, confidence={r['Confidence']*100:.1f}%, bbox={r['Bounding Box']}."
                )
                try:
                    # reuse incident summarizer provider via direct call
                    from config.settings import get_settings as _gs
                    settings = _gs()
                    if settings.llm_provider == "groq":
                        from groq import Groq
                        if not settings.groq_api_key:
                            raise RuntimeError("GROQ_API_KEY not set")
                        client = Groq(api_key=settings.groq_api_key)
                        resp = client.chat.completions.create(
                            model=settings.groq_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=120,
                        )
                        st.info((resp.choices[0].message.content or "").strip())
                    else:
                        import ollama
                        resp = ollama.chat(model=settings.ollama_model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
                        st.info((resp.get("message", {}).get("content") or "").strip())
                except Exception:
                    st.info("This region likely contains visual cues consistent with fire/smoke. Verify the area and escalate if confirmed.")

        st.markdown("### Downloads / Evidence")
        d1, d2, d3 = st.columns(3)
        try:
            import cv2
            ok, buf = cv2.imencode(".jpg", det.annotated_frame)
            annotated_bytes = buf.tobytes() if ok else b""
        except Exception:
            annotated_bytes = b""
        d1.download_button("DOWNLOAD ANNOTATED", data=annotated_bytes, file_name="pyrosense_annotated.jpg", mime="image/jpeg", disabled=not bool(annotated_bytes))

        pdf_bytes = _simple_pdf_bytes("PyroSense AI Incident Report", llm)
        d2.download_button("DOWNLOAD REPORT PDF", data=pdf_bytes, file_name="pyrosense_report.pdf", mime="application/pdf")

        st.markdown("### Similar Incidents Lookup")
        if st.checkbox("Show Similar Incidents"):
            from dashboard.app import load_similarity_search
            searcher = load_similarity_search()
            similar = searcher.search_similar(frame, top_k=3)
            if similar:
                for s in similar:
                    cols = st.columns([1, 4])
                    if s.get("frame_path"):
                        fp = Path(s["frame_path"])
                        if fp.exists():
                            cols[0].image(str(fp))
                    cols[1].markdown(
                        f"**Incident #{s['detection_id']}**  \n"
                        f"Class: {s['class_name']} | Match Score: {s['score']:.2%}  \n"
                        f"Time: {s['timestamp']}"
                    )
            else:
                st.info("No similar incidents found.")
        return

    # Video path: quick processing preview
    st.markdown("<div class='pyro-card'><div style='font-family:monospace; color:#4B5563;'>Video uploaded. Use Live Detection → UPLOAD VIDEO for real-time processing.</div></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

