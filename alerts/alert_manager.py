"""Async AlertManager orchestrating multi-channel notifications.

Channels:
  - Email
  - Telegram
  - Audio
  - Webhook

The manager enforces per-channel cooldown to prevent alert spam and persists
delivery status into the database.

Example:
    >>> from alerts.alert_manager import AlertManager
    >>> _ = AlertManager()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from config.settings import get_settings
from database import crud
from database.session import SessionLocal
from utils.logger import logger

try:
    from alerts.audio_alert import AudioAlert
except ImportError:
    AudioAlert = None

try:
    from alerts.email_alert import EmailAlert
except ImportError:
    EmailAlert = None

try:
    from alerts.telegram_alert import TelegramAlert
except ImportError:
    TelegramAlert = None

try:
    from alerts.webhook_dispatcher import WebhookDispatcher
except ImportError:
    WebhookDispatcher = None



@dataclass
class ChannelState:
    last_sent_at: Optional[datetime] = None


class AlertManager:
    """Dispatch alerts concurrently with cooldown and DB logging."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cooldown = int(self.settings.alert_cooldown_seconds)
        self._state: Dict[str, ChannelState] = {k: ChannelState() for k in ["email", "telegram", "audio", "webhook"]}

        self._email: Optional[EmailAlert] = None
        self._telegram: Optional[TelegramAlert] = None
        self._audio = AudioAlert() if AudioAlert else None
        self._webhook: Optional[WebhookDispatcher] = None

        if EmailAlert and self.settings.email_enabled and self.settings.email_user and self.settings.email_password and self.settings.email_recipient:
            self._email = EmailAlert(
                smtp_host=self.settings.email_smtp_host,
                smtp_port=self.settings.email_smtp_port,
                user=self.settings.email_user,
                password=self.settings.email_password,
                recipient=self.settings.email_recipient,
            )
        if TelegramAlert and self.settings.telegram_enabled and self.settings.telegram_bot_token and self.settings.telegram_chat_id:
            self._telegram = TelegramAlert(bot_token=self.settings.telegram_bot_token, chat_id=self.settings.telegram_chat_id)
        if WebhookDispatcher and self.settings.webhook_enabled and self.settings.webhook_url:
            self._webhook = WebhookDispatcher(str(self.settings.webhook_url))

    def _cooldown_ok(self, channel: str) -> bool:
        st = self._state[channel]
        if st.last_sent_at is None:
            return True
        return (datetime.utcnow() - st.last_sent_at) >= timedelta(seconds=self.cooldown)

    def _mark_sent(self, channel: str) -> None:
        self._state[channel].last_sent_at = datetime.utcnow()

    async def trigger_alert(self, detection_payload: Dict[str, Any], *, detection_id: int, location: str) -> None:
        """Trigger all enabled alert channels concurrently.

        `detection_payload` should include keys produced by `InferenceEngine.detect_image()`.

        Example:
            >>> import asyncio
            >>> from alerts.alert_manager import AlertManager
            >>> am = AlertManager()
            >>> asyncio.get_event_loop().run_until_complete(am.trigger_alert({"primary_class":"fire","ensemble_conf":0.9,"risk":{"score":80,"severity":"HIGH"},"timestamp":"x"}, detection_id=1, location="Test"))
        """

        tasks = []
        if self._email and self._cooldown_ok("email"):
            tasks.append(self._send_email(detection_payload, detection_id=detection_id, location=location))
        else:
            self._log_skip(detection_id, "email")

        if self._telegram and self._cooldown_ok("telegram"):
            tasks.append(self._send_telegram(detection_payload, detection_id=detection_id, location=location))
        else:
            self._log_skip(detection_id, "telegram")

        if self._audio and self._cooldown_ok("audio"):
            tasks.append(self._send_audio(detection_payload, detection_id=detection_id, location=location))
        else:
            self._log_skip(detection_id, "audio")

        if self._webhook and self._cooldown_ok("webhook"):
            tasks.append(self._send_webhook(detection_payload, detection_id=detection_id, location=location))
        else:
            self._log_skip(detection_id, "webhook")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _log_skip(self, detection_id: int, channel: str) -> None:
        with SessionLocal() as db:
            crud.create_alert_log(db, detection_id=detection_id, channel=channel, status="skipped", sent_at=None, error_msg="disabled or cooldown")

    async def _send_email(self, payload: Dict[str, Any], *, detection_id: int, location: str) -> None:
        assert self._email is not None
        try:
            primary = str(payload.get("primary_class", "incident"))
            ts = str(payload.get("timestamp", datetime.utcnow().isoformat()))
            conf = float(payload.get("ensemble_conf", 0.0)) * 100.0
            risk = payload.get("risk") or {}
            llm_summary = str(payload.get("llm_summary", ""))
            subject = f"[PyroSense AI] {primary.upper()} alert"
            # Wrap in run_in_executor
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(
                None,
                lambda: self._email.send(
                    subject=subject,
                    timestamp=ts,
                    location=location,
                    class_name=primary,
                    confidence_pct=conf,
                    risk_score=float(risk.get("score", 0.0)),
                    risk_severity=str(risk.get("severity", "LOW")),
                    llm_summary=llm_summary,
                )
            )
            with SessionLocal() as db:
                crud.create_alert_log(
                    db,
                    detection_id=detection_id,
                    channel="email",
                    status="sent" if r.ok else "failed",
                    sent_at=datetime.utcnow() if r.ok else None,
                    error_msg=r.error,
                )
            if r.ok:
                self._mark_sent("email")
        except Exception as e:
            logger.warning(f"Email channel failed: {e}")
            with SessionLocal() as db:
                crud.create_alert_log(db, detection_id=detection_id, channel="email", status="failed", sent_at=None, error_msg=str(e))

    async def _send_telegram(self, payload: Dict[str, Any], *, detection_id: int, location: str) -> None:
        assert self._telegram is not None
        try:
            primary = str(payload.get("primary_class", "incident"))
            ts = str(payload.get("timestamp", datetime.utcnow().isoformat()))
            conf = float(payload.get("ensemble_conf", 0.0)) * 100.0
            risk = payload.get("risk") or {}
            llm_summary = str(payload.get("llm_summary", ""))
            text = (
                f"🔥 PyroSense AI Alert\n"
                f"Type: {primary}\nLocation: {location}\nTime (UTC): {ts}\n"
                f"Confidence: {conf:.0f}%\nRisk: {float(risk.get('score',0)):.0f} ({risk.get('severity','LOW')})\n\n"
                f"{llm_summary}"
            )
            r = await self._telegram.send_message(text)
            with SessionLocal() as db:
                crud.create_alert_log(
                    db,
                    detection_id=detection_id,
                    channel="telegram",
                    status="sent" if r.ok else "failed",
                    sent_at=datetime.utcnow() if r.ok else None,
                    error_msg=r.error,
                )
            if r.ok:
                self._mark_sent("telegram")
        except Exception as e:
            logger.warning(f"Telegram channel failed: {e}")
            with SessionLocal() as db:
                crud.create_alert_log(db, detection_id=detection_id, channel="telegram", status="failed", sent_at=None, error_msg=str(e))

    async def _send_audio(self, payload: Dict[str, Any], *, detection_id: int, location: str) -> None:
        try:
            primary = str(payload.get("primary_class", "incident"))
            text = f"Warning. {primary} detected at {location}. Please evacuate and notify responders."
            r = self._audio.trigger(text)
            with SessionLocal() as db:
                crud.create_alert_log(
                    db,
                    detection_id=detection_id,
                    channel="audio",
                    status="sent" if r.ok else "failed",
                    sent_at=datetime.utcnow() if r.ok else None,
                    error_msg=r.error,
                )
            if r.ok:
                self._mark_sent("audio")
        except Exception as e:
            logger.warning(f"Audio channel failed: {e}")
            with SessionLocal() as db:
                crud.create_alert_log(db, detection_id=detection_id, channel="audio", status="failed", sent_at=None, error_msg=str(e))

    async def _send_webhook(self, payload: Dict[str, Any], *, detection_id: int, location: str) -> None:
        assert self._webhook is not None
        try:
            data = dict(payload)
            data["location"] = location
            r = await self._webhook.send(data)
            with SessionLocal() as db:
                crud.create_alert_log(
                    db,
                    detection_id=detection_id,
                    channel="webhook",
                    status="sent" if r.ok else "failed",
                    sent_at=datetime.utcnow() if r.ok else None,
                    error_msg=r.error,
                )
            if r.ok:
                self._mark_sent("webhook")
        except Exception as e:
            logger.warning(f"Webhook channel failed: {e}")
            with SessionLocal() as db:
                crud.create_alert_log(db, detection_id=detection_id, channel="webhook", status="failed", sent_at=None, error_msg=str(e))

    async def send_single_alert(self, channel: str, message: str, detection_id: int, location: str, payload: dict) -> dict:
        """Send an alert to a single channel directly, respecting channel configuration and cooldown."""
        if channel not in self._state:
            return {"status": "failed", "error": f"Invalid channel: {channel}"}

        if not self._cooldown_ok(channel):
            self._log_skip(detection_id, channel)
            return {"status": "skipped", "reason": f"Cooldown active for channel: {channel}"}

        temp_payload = dict(payload)
        temp_payload["llm_summary"] = message

        try:
            if channel == "email":
                if not self._email:
                    return {"status": "failed", "error": "Email channel is not configured or disabled"}
                await self._send_email(temp_payload, detection_id=detection_id, location=location)
                return {"status": "success", "channel": "email"}
            elif channel == "telegram":
                if not self._telegram:
                    return {"status": "failed", "error": "Telegram channel is not configured or disabled"}
                await self._send_telegram(temp_payload, detection_id=detection_id, location=location)
                return {"status": "success", "channel": "telegram"}
            elif channel == "webhook":
                if not self._webhook:
                    return {"status": "failed", "error": "Webhook channel is not configured or disabled"}
                await self._send_webhook(temp_payload, detection_id=detection_id, location=location)
                return {"status": "success", "channel": "webhook"}
            elif channel == "audio":
                await self._send_audio(temp_payload, detection_id=detection_id, location=location)
                return {"status": "success", "channel": "audio"}
        except Exception as e:
            logger.warning(f"Failed to send alert on channel {channel}: {e}")
            return {"status": "failed", "error": str(e)}

        return {"status": "failed", "error": f"Channel {channel} is not handled or configured"}


