"""Async AlertManager orchestrating multi-channel notifications.

Channels:
  - Email
  - Telegram
  - Audio
  - Webhook

The manager enforces per-channel cooldown to prevent alert spam and persists
delivery status into the database.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional

from sqlalchemy import select

from config.settings import get_settings
from database import crud
from database.models import AlertLog
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


class IncidentStatus(str, Enum):
    SAFE = "SAFE"
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"


@dataclass
class IncidentState:
    status: IncidentStatus = IncidentStatus.SAFE
    consecutive_detections: int = 0
    consecutive_safe_frames: int = 0
    last_detection_time: Optional[datetime] = None
    alerts_sent_count: int = 0
    max_severity: str = "LOW"
    max_risk_score: float = 0.0
    last_alert_sent_at: Optional[datetime] = None


@dataclass
class ChannelState:
    last_sent_at: Optional[datetime] = None


class AlertManager:
    """Dispatch alerts concurrently with cooldown and DB logging."""

    _instance: Optional[AlertManager] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AlertManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_shared_instance(cls) -> AlertManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset_state(self) -> None:
        """Helper to clear states for unit tests."""
        self._state = {k: ChannelState() for k in ["email", "telegram", "audio", "webhook"]}
        self._incidents = {}

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
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

        # Incident state tracking per source/location
        self._incidents: Dict[str, IncidentState] = {}

    def _get_persistent_last_sent_at(self, channel: str) -> Optional[datetime]:
        try:
            with SessionLocal() as db:
                stmt = (
                    select(AlertLog.sent_at)
                    .where(AlertLog.channel == channel)
                    .where(AlertLog.status == "sent")
                    .order_by(AlertLog.sent_at.desc())
                    .limit(1)
                )
                res = db.execute(stmt).scalars().first()
                if res:
                    if res.tzinfo is not None:
                        return res.astimezone(timezone.utc).replace(tzinfo=None)
                    return res
        except Exception as e:
            logger.warning(f"Database error querying persistent alert logs: {e}")
        return None

    def _cooldown_ok(self, channel: str) -> bool:
        st = self._state[channel]
        now = datetime.utcnow()
        if st.last_sent_at is not None:
            if (now - st.last_sent_at) < timedelta(seconds=self.cooldown):
                return False

        # Fallback to database query to determine persistent cooldown
        last_sent = self._get_persistent_last_sent_at(channel)
        if last_sent is not None:
            if (now - last_sent) < timedelta(seconds=self.cooldown):
                st.last_sent_at = last_sent
                return False
        return True

    def _mark_sent(self, channel: str) -> None:
        self._state[channel].last_sent_at = datetime.utcnow()

    async def trigger_alert(
        self,
        detection_payload: Dict[str, Any],
        *,
        detection_id: int,
        location: str,
        source: str = "unknown"
    ) -> None:
        """Process detection payload through temporal verification and incident state machine.

        `detection_payload` should include keys produced by `InferenceEngine.detect_image()`.
        """
        primary_class = str(detection_payload.get("primary_class", "none")).lower()
        is_threat = primary_class in ["fire", "smoke", "flame", "fire_smoke", "incident"]
        
        # Load settings
        min_consecutive = self.settings.min_consecutive_detections
        clear_seconds = self.settings.incident_clear_seconds
        max_alerts = self.settings.max_alerts_per_incident
        cooldown_seconds = self.settings.alert_cooldown_seconds
        
        # Incident state tracking per source/location combination
        key = f"{source}:{location}"
        if key not in self._incidents:
            self._incidents[key] = IncidentState()
        inc = self._incidents[key]

        now = datetime.utcnow()
        risk = detection_payload.get("risk") or {}
        risk_score = float(risk.get("score", 0.0))
        severity = str(risk.get("severity", "LOW")).upper()

        severity_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        curr_sev_level = severity_map.get(severity, 1)
        max_sev_level = severity_map.get(inc.max_severity, 1)

        if is_threat:
            inc.consecutive_detections += 1
            inc.consecutive_safe_frames = 0
            inc.last_detection_time = now
            
            # Temporal verification check
            if inc.consecutive_detections >= min_consecutive:
                if inc.status != IncidentStatus.ACTIVE:
                    # SAFE -> ACTIVE or RESOLVED -> ACTIVE transition: trigger alert
                    inc.status = IncidentStatus.ACTIVE
                    inc.max_severity = severity
                    inc.max_risk_score = risk_score
                    await self._dispatch_channels(detection_payload, detection_id=detection_id, location=location, is_resolution=False)
                    inc.alerts_sent_count = 1
                    inc.last_alert_sent_at = now
                else:
                    # Already ACTIVE: check if we should send a follow-up alert
                    should_alert = False
                    reason = ""
                    
                    # Risk level increase significantly (e.g., from Low/Medium -> Critical)
                    if curr_sev_level > max_sev_level:
                        should_alert = True
                        reason = f"escalation from {inc.max_severity} to {severity}"
                    # Cooldown check
                    elif (inc.last_alert_sent_at is None or (now - inc.last_alert_sent_at) >= timedelta(seconds=cooldown_seconds)) and inc.alerts_sent_count < max_alerts:
                        # Validate with persistent cooldown
                        persistent_ok = True
                        for ch in ["email", "telegram", "webhook"]:
                            if self._channel_enabled(ch):
                                last_sent = self._get_persistent_last_sent_at(ch)
                                if last_sent is not None and (now - last_sent) < timedelta(seconds=cooldown_seconds):
                                    persistent_ok = False
                                    break
                        if persistent_ok:
                            should_alert = True
                            reason = "cooldown expired"
                    
                    if should_alert:
                        if curr_sev_level > max_sev_level:
                            inc.max_severity = severity
                        inc.max_risk_score = max(inc.max_risk_score, risk_score)
                        
                        await self._dispatch_channels(detection_payload, detection_id=detection_id, location=location, is_resolution=False)
                        inc.alerts_sent_count += 1
                        inc.last_alert_sent_at = now
                    else:
                        # Log why the alert was skipped
                        self._log_skip(detection_id, "email", "duplicate alert suppressed (active incident)")
                        self._log_skip(detection_id, "telegram", "duplicate alert suppressed (active incident)")
                        self._log_skip(detection_id, "webhook", "duplicate alert suppressed (active incident)")
            else:
                self._log_skip(detection_id, "email", f"temporal verification in progress (consecutive detections: {inc.consecutive_detections}/{min_consecutive})")
                self._log_skip(detection_id, "telegram", f"temporal verification in progress (consecutive detections: {inc.consecutive_detections}/{min_consecutive})")
                self._log_skip(detection_id, "webhook", f"temporal verification in progress (consecutive detections: {inc.consecutive_detections}/{min_consecutive})")
        else:
            inc.consecutive_detections = 0
            
            if inc.status == IncidentStatus.ACTIVE:
                # Check clear interval
                if inc.last_detection_time is None or (now - inc.last_detection_time) >= timedelta(seconds=clear_seconds):
                    # Transition to RESOLVED
                    inc.status = IncidentStatus.RESOLVED
                    await self._dispatch_channels(detection_payload, detection_id=detection_id, location=location, is_resolution=True)
                    # Reset incident parameters
                    inc.alerts_sent_count = 0
                    inc.max_severity = "LOW"
                    inc.max_risk_score = 0.0
                    inc.last_alert_sent_at = None

    def _channel_enabled(self, channel: str) -> bool:
        if channel == "email":
            return self._email is not None
        elif channel == "telegram":
            return self._telegram is not None
        elif channel == "webhook":
            return self._webhook is not None
        elif channel == "audio":
            return self._audio is not None
        return False

    async def _dispatch_channels(
        self,
        payload: Dict[str, Any],
        *,
        detection_id: int,
        location: str,
        is_resolution: bool
    ) -> None:
        tasks = []
        dispatch_payload = dict(payload)
        
        if is_resolution:
            dispatch_payload["primary_class"] = "RESOLVED"
            dispatch_payload["llm_summary"] = f"Incident at {location} has been RESOLVED. No threat detected for the last {self.settings.incident_clear_seconds} seconds."
            
        if self._email:
            tasks.append(self._send_email(dispatch_payload, detection_id=detection_id, location=location, is_resolution=is_resolution))
        else:
            self._log_skip(detection_id, "email", "disabled")
            
        if self._telegram:
            tasks.append(self._send_telegram(dispatch_payload, detection_id=detection_id, location=location, is_resolution=is_resolution))
        else:
            self._log_skip(detection_id, "telegram", "disabled")
            
        if self._audio and not is_resolution:  # Don't trigger siren for resolution
            tasks.append(self._send_audio(dispatch_payload, detection_id=detection_id, location=location))
        else:
            self._log_skip(detection_id, "audio", "disabled" if self._audio is None else "skipped for resolution")
            
        if self._webhook:
            tasks.append(self._send_webhook(dispatch_payload, detection_id=detection_id, location=location))
        else:
            self._log_skip(detection_id, "webhook", "disabled")
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _log_skip(self, detection_id: int, channel: str, reason: str = "disabled or cooldown") -> None:
        try:
            with SessionLocal() as db:
                crud.create_alert_log(db, detection_id=detection_id, channel=channel, status="skipped", sent_at=None, error_msg=reason)
        except Exception as e:
            logger.warning(f"Database error writing skip alert log: {e}")

    async def _send_email(self, payload: Dict[str, Any], *, detection_id: int, location: str, is_resolution: bool = False) -> None:
        assert self._email is not None
        try:
            primary = str(payload.get("primary_class", "incident"))
            ts = str(payload.get("timestamp", datetime.utcnow().isoformat()))
            conf = float(payload.get("ensemble_conf", 0.0)) * 100.0
            risk = payload.get("risk") or {}
            llm_summary = str(payload.get("llm_summary", ""))
            
            if is_resolution:
                subject = f"[PyroSense AI] Incident RESOLVED at {location}"
            else:
                subject = f"[PyroSense AI] {primary.upper()} alert at {location}"
                
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

    async def _send_telegram(self, payload: Dict[str, Any], *, detection_id: int, location: str, is_resolution: bool = False) -> None:
        assert self._telegram is not None
        try:
            primary = str(payload.get("primary_class", "incident"))
            ts = str(payload.get("timestamp", datetime.utcnow().isoformat()))
            conf = float(payload.get("ensemble_conf", 0.0)) * 100.0
            risk = payload.get("risk") or {}
            llm_summary = str(payload.get("llm_summary", ""))
            
            if is_resolution:
                text = (
                    f"✅ PyroSense AI Incident RESOLVED\n"
                    f"Location: {location}\n"
                    f"Time (UTC): {ts}\n\n"
                    f"The threat has cleared. No fire/smoke detected for the last {self.settings.incident_clear_seconds} seconds."
                )
            else:
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
            self._log_skip(detection_id, channel, "cooldown active")
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
