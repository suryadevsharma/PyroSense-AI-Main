"""Streamlit dashboard entrypoint for PyroSense AI.

Run:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import streamlit as st

# Ensure project root is importable when run via Streamlit.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import get_settings
from database.models import Detection
from database.session import SessionLocal
from database.migrations.init_db import init_db

try:
    init_db()
except Exception as e:
    from utils.logger import logger
    logger.warning(f"Database auto-init failed on dashboard startup: {e}")


st.set_page_config(
    page_title="PyroSense AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path("dashboard/assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)



@st.cache_resource
def load_detector():
    """Cached inference engine for Streamlit."""
    from inference.detector import InferenceEngine
    return InferenceEngine()


@st.cache_resource
def get_db_engine():
    """Cached DB engine for Streamlit."""
    from database.session import init_engine
    return init_engine()


@st.cache_resource
def load_similarity_search():
    """Cached similarity search index for Streamlit."""
    from llm.faiss_history import FaissHistory
    return FaissHistory.get_shared_instance()


@st.cache_resource
def get_engine():
    """Cached inference engine wrapper for backwards compatibility."""
    return load_detector()


@st.cache_resource
def get_summarizer():
    """Cached LLM summarizer."""

    from llm.incident_summarizer import IncidentSummarizer

    return IncidentSummarizer()


@st.cache_resource
def get_faiss():
    """Cached FAISS history index wrapper for backwards compatibility."""
    return load_similarity_search()


@st.cache_data(ttl=5)
def _get_today_stats() -> Dict[str, int]:
    """Return today's summary utilizing optimized SQL aggregation queries and cache."""

    try:
        from sqlalchemy import func
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        with SessionLocal() as db:
            total = db.query(func.count(Detection.id)).filter(Detection.timestamp >= start).scalar() or 0
            fire = db.query(func.count(Detection.id)).filter(
                Detection.timestamp >= start,
                Detection.class_name.ilike("%fire%")
            ).scalar() or 0
            smoke = db.query(func.count(Detection.id)).filter(
                Detection.timestamp >= start,
                Detection.class_name.ilike("%smoke%")
            ).scalar() or 0
        return {"total": int(total), "fire": int(fire), "smoke": int(smoke), "alerts_sent": 0}
    except Exception:
        return {"total": 0, "fire": 0, "smoke": 0, "alerts_sent": 0}


# Top Navigation Bar
st.markdown(
    """
<div class="top-navbar">
  <div class="brand">
    <div class="brand-icon">&#x1F6E1;</div>
    <div>
      <div class="brand-text">IntelliGuard</div>
      <div class="brand-sub">AI Fire &amp; Smoke Detection System</div>
    </div>
  </div>
  <div class="nav-status">
    <div class="status-pill">
      <span class="status-dot active"></span>
      System Online
    </div>
    <div class="nav-info">YOLOv8s + EfficientNetV2</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar — brand + navigation
with st.sidebar:
    st.markdown(
        '<div style="padding:20px 0 24px;border-bottom:1px solid #E5E7EB;">'
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<div style="width:32px;height:32px;background:linear-gradient(135deg,#E53E3E,#F97316);'
        'border-radius:8px;display:flex;align-items:center;justify-content:center;'
        'font-size:16px;box-shadow:0 2px 6px rgba(229,62,62,0.25);">&#x1F6E1;</div>'
        '<div>'
        '<div style="font-family:Inter,sans-serif;font-size:16px;font-weight:700;color:#111827;">IntelliGuard</div>'
        '<div style="font-family:Inter,sans-serif;font-size:10px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.1em;">Fire Detection</div>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="padding:14px 0;border-bottom:1px solid #E5E7EB;">'
        '<div style="display:flex;align-items:center;gap:8px;font-family:Inter,sans-serif;font-size:12px;color:#10B981;font-weight:500;">'
        '<span class="status-dot active"></span>System Operational</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#9CA3AF;margin-top:4px;">MODEL: YOLOv8s-FireSmoke v1.2</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Quick stats in sidebar
    try:
        stats = _get_today_stats()
    except Exception:
        stats = {"total": 0, "fire": 0, "smoke": 0, "alerts_sent": 0}

    st.markdown(
        f'<div style="padding:14px 0;border-bottom:1px solid #E5E7EB;">'
        f'<div style="font-family:Inter,sans-serif;font-size:10px;color:#9CA3AF;'
        f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;font-weight:600;">Today\'s Summary</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'
        f'<div style="text-align:center;padding:10px 8px;background:#FEF2F2;border-radius:10px;border:1px solid #FECACA;">'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:20px;color:#DC2626;font-weight:700;">{stats["fire"]}</div>'
        f'<div style="font-family:Inter,sans-serif;font-size:10px;color:#9CA3AF;font-weight:500;">FIRE</div></div>'
        f'<div style="text-align:center;padding:10px 8px;background:#F3F4F6;border-radius:10px;border:1px solid #E5E7EB;">'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:20px;color:#6B7280;font-weight:700;">{stats["smoke"]}</div>'
        f'<div style="font-family:Inter,sans-serif;font-size:10px;color:#9CA3AF;font-weight:500;">SMOKE</div></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Alert channel status
    settings = get_settings()
    channels = {
        "Email": bool(settings.email_enabled),
        "Telegram": bool(settings.telegram_enabled),
        "Webhook": bool(settings.webhook_enabled),
    }
    channel_html = ""
    for name, enabled in channels.items():
        color = "#10B981" if enabled else "#D1D5DB"
        bg = "#D1FAE5" if enabled else "#F3F4F6"
        label = "ON" if enabled else "OFF"
        channel_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;font-family:Inter,sans-serif;font-size:12px;">'
            f'<span style="color:#4B5563;">{name}</span>'
            f'<span style="background:{bg};color:{color};font-size:10px;font-weight:600;'
            f'padding:2px 8px;border-radius:10px;">{label}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="padding:14px 0;">'
        f'<div style="font-family:Inter,sans-serif;font-size:10px;color:#9CA3AF;'
        f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;font-weight:600;">'
        f'Alert Channels</div>'
        f'{channel_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
