"""Streamlit page: Settings panel (mission-control UI)."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alerts.email_alert import EmailAlert
from alerts.telegram_alert import TelegramAlert
from alerts.webhook_dispatcher import WebhookDispatcher
from config.settings import get_settings
from database.models import AlertLog, Detection
from database.session import SessionLocal
from inference.detector import InferenceEngine


OVERRIDE_PATH = Path("data/processed/settings_override.json")


def _load_css() -> None:
    css_path = Path("dashboard/assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _load_override() -> Dict[str, Any]:
    if OVERRIDE_PATH.exists():
        try:
            return json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_override(data: Dict[str, Any]) -> None:
    OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _mask_email(email: Optional[str]) -> str:
    if not email:
        return "Not Configured"
    if "@" not in email:
        return "********"
    parts = email.split("@", 1)
    user_part = parts[0]
    domain_part = parts[1]
    if len(user_part) <= 2:
        return "**@" + domain_part
    return user_part[:2] + "********@" + domain_part


def _mask_credential(val: Optional[str], secret_name: str = "Secret") -> str:
    if not val:
        return "Not Configured"
    return f"Configured via Hugging Face {secret_name}"


def _mask_chat_id(chat_id: Optional[str]) -> str:
    if not chat_id:
        return "Not Configured"
    chat_id_str = str(chat_id)
    if len(chat_id_str) <= 3:
        return "***"
    return chat_id_str[:2] + "********"


def _mask_database_url(url: str) -> str:
    if "@" in url and "://" in url:
        try:
            proto, rest = url.split("://", 1)
            credentials, location = rest.split("@", 1)
            if ":" in credentials:
                user, password = credentials.split(":", 1)
                return f"{proto}://{user}:********@{location}"
        except Exception:
            pass
    return url


def main() -> None:
    try:
        st.set_page_config(page_title="PyroSense AI", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
    _load_css()

    st.markdown(
        """
    <div style="padding: 0 0 18px; border-bottom: 1px solid rgba(0,0,0,0.06); margin-bottom: 18px;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:22px; color:#111827; font-weight:700;">
        SETTINGS
      </div>
      <div style="font-family:monospace; font-size:11px; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.1em; margin-top:6px;">
        Detection engine, alerts, risk scoring, and system controls
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    s = get_settings()
    override = _load_override()

    with st.expander("SECTION 1 — DETECTION ENGINE", expanded=True):
        conf = st.slider("Confidence threshold", 0.1, 0.9, float(override.get("CONF_THRESHOLD", s.conf_threshold)), 0.01)
        iou = st.slider("IOU threshold", 0.1, 0.9, float(override.get("IOU_THRESHOLD", s.iou_threshold)), 0.01)
        model_path = st.text_input("Model path", value=str(override.get("YOLO_MODEL_PATH", s.yolo_model_path)))
        uploaded = st.file_uploader("Browse (upload .pt weights)", type=["pt"])
        if uploaded is not None:
            out = Path("models/weights") / uploaded.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(uploaded.getvalue())
            model_path = str(out)
            st.success(f"Saved weights to {out}")
        device = st.radio("Device", ["auto", "cpu", "cuda", "mps"], index=["auto", "cpu", "cuda", "mps"].index(str(override.get("DEVICE", s.device))))

        if st.button("Test Model"):
            try:
                import numpy as np
                from PIL import Image
                from utils.image_utils import pil_to_bgr

                samples = list((Path("data/samples")).glob("*.jpg")) + list((Path("data/samples")).glob("*.png"))
                if not samples:
                    st.warning("No samples found in data/samples/. Run `python data/download_datasets.py --dataset dfire-mini` first.")
                else:
                    img = pil_to_bgr(Image.open(samples[0]).convert("RGB"))
                    eng = InferenceEngine()
                    out = eng.detect_image(img)
                    st.json(out)
            except Exception as e:
                st.error(f"Model test failed: {e}")

        if st.button("Save detection overrides"):
            override["CONF_THRESHOLD"] = conf
            override["IOU_THRESHOLD"] = iou
            override["YOLO_MODEL_PATH"] = model_path
            override["DEVICE"] = device
            _save_override(override)
            st.success("Saved detection overrides. Restart services to apply.")

    with st.expander("SECTION 2 — ALERT CHANNELS", expanded=False):
        # EMAIL
        st.markdown("<div class='pyro-card'>", unsafe_allow_html=True)
        st.markdown("### 📧 Email Channel")
        
        email_configured = bool(s.email_user and s.email_password and s.email_recipient)
        email_status = "✅ Loaded from Environment" if (s.email_enabled and email_configured) else ("❌ Disabled" if not s.email_enabled else "⚠️ Configuration Missing")
        
        st.markdown(
            f"**SMTP Host**: `{s.email_smtp_host}`  \n"
            f"**SMTP Port**: `{s.email_smtp_port}`  \n"
            f"**SMTP User**: `{_mask_email(s.email_user)}`  \n"
            f"**Password**: `{_mask_credential(s.email_password, 'EMAIL_PASSWORD')}`  \n"
            f"**Recipient**: `{_mask_email(s.email_recipient)}`  \n"
            f"**Status**: {email_status}"
        )
        
        if st.button("Send Test Email"):
            if not email_configured:
                st.error("Email configuration is incomplete on the server.")
            else:
                r = EmailAlert(
                    smtp_host=s.email_smtp_host,
                    smtp_port=int(s.email_smtp_port),
                    user=s.email_user,
                    password=s.email_password,
                    recipient=s.email_recipient
                ).send(
                    subject="[PyroSense] Test Email",
                    timestamp="now",
                    location="Settings",
                    class_name="fire",
                    confidence_pct=99.0,
                    risk_score=80.0,
                    risk_severity="HIGH",
                    llm_summary="Test message from PyroSense AI.",
                )
                if r.ok:
                    st.success("Sent.")
                else:
                    st.error(f"Failed: {r.error}")
        st.markdown("</div>", unsafe_allow_html=True)

        # TELEGRAM
        st.markdown("<div class='pyro-card'>", unsafe_allow_html=True)
        st.markdown("### 🤖 Telegram Channel")
        
        tg_configured = bool(s.telegram_bot_token and s.telegram_chat_id)
        tg_status = "✅ Loaded from Environment" if (s.telegram_enabled and tg_configured) else ("❌ Disabled" if not s.telegram_enabled else "⚠️ Configuration Missing")
        
        st.markdown(
            f"**Bot Token**: `{_mask_credential(s.telegram_bot_token, 'TELEGRAM_BOT_TOKEN')}`  \n"
            f"**Chat ID**: `{_mask_chat_id(s.telegram_chat_id)}`  \n"
            f"**Status**: {tg_status}"
        )
        
        if st.button("Send Test Message"):
            if not tg_configured:
                st.error("Telegram configuration is incomplete on the server.")
            else:
                import asyncio

                async def _send():
                    tga = TelegramAlert(bot_token=s.telegram_bot_token, chat_id=s.telegram_chat_id)
                    return await tga.send_message("PyroSense AI test message ✅")

                res = asyncio.run(_send())
                if res.ok:
                    st.success("Sent.")
                else:
                    st.error(f"Failed: {res.error}")
        st.markdown("</div>", unsafe_allow_html=True)

        # WEBHOOK
        st.markdown("<div class='pyro-card'>", unsafe_allow_html=True)
        st.markdown("### 🔗 Webhook Channel")
        
        wh_configured = bool(s.webhook_url)
        wh_status = "✅ Loaded from Environment" if (s.webhook_enabled and wh_configured) else ("❌ Disabled" if not s.webhook_enabled else "⚠️ Configuration Missing")
        
        wh_url_str = str(s.webhook_url) if s.webhook_url else "Not Configured"
        st.markdown(
            f"**URL**: `{wh_url_str}`  \n"
            f"**Status**: {wh_status}"
        )
        
        if st.button("Test Webhook"):
            if not wh_configured:
                st.error("Webhook configuration is incomplete on the server.")
            else:
                import asyncio
                import httpx

                async def _go():
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        r = await client.post(str(s.webhook_url), json={"test": True, "service": "pyrosense"})
                        return r.status_code, r.text[:200]

                code, text = asyncio.run(_go())
                if 200 <= code < 300:
                    st.success(f"Webhook responded: {code}")
                else:
                    st.error(f"{code}: {text}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("SECTION 3 — RISK SCORING", expanded=False):
        cw = st.slider("Confidence weight", 0.0, 1.0, float(override.get("RISK_W_CONF", 0.4)), 0.05)
        aw = st.slider("Detection area weight", 0.0, 1.0, float(override.get("RISK_W_AREA", 0.3)), 0.05)
        gw = st.slider("Growth rate weight", 0.0, 1.0, float(override.get("RISK_W_GROWTH", 0.2)), 0.05)
        sw = st.slider("Smoke proximity weight", 0.0, 1.0, float(override.get("RISK_W_SMOKE", 0.1)), 0.05)
        if st.button("Reset to Defaults"):
            override.update({"RISK_W_CONF": 0.4, "RISK_W_AREA": 0.3, "RISK_W_GROWTH": 0.2, "RISK_W_SMOKE": 0.1})
            _save_override(override)
            st.success("Reset risk weights.")
        if st.button("Save risk overrides"):
            override["RISK_W_CONF"] = cw
            override["RISK_W_AREA"] = aw
            override["RISK_W_GROWTH"] = gw
            override["RISK_W_SMOKE"] = sw
            _save_override(override)
            st.success("Saved risk overrides.")

    with st.expander("SECTION 4 — SYSTEM", expanded=False):
        st.write(f"Database: `{_mask_database_url(s.database_url)}`")
        confirm = st.checkbox("I understand this will delete all history")
        if st.button("Clear History", disabled=not confirm):
            with SessionLocal() as db:
                db.query(AlertLog).delete()
                db.query(Detection).delete()
                db.commit()
            st.success("History cleared.")
        if st.button("Export All Data"):
            with SessionLocal() as db:
                dets = db.query(Detection).order_by(Detection.timestamp.desc()).all()
            data = [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat(),
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "risk_score": d.risk_score,
                    "source": d.source,
                    "llm_summary": d.llm_summary,
                }
                for d in dets
            ]
            st.download_button("Download JSON export", data=json.dumps(data, indent=2).encode("utf-8"), file_name="pyrosense_export.json")
        st.caption("Version: PyroSense AI 1.0.0")


if __name__ == "__main__":
    main()

