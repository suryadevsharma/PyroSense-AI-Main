"""Email alert channel using smtplib with an HTML template.

Example:
    >>> from alerts.email_alert import EmailAlert
    >>> _ = EmailAlert(smtp_host="smtp.gmail.com", smtp_port=587, user="u", password="p", recipient="r")
"""

from __future__ import annotations

import smtplib
import time
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from utils.logger import logger


@dataclass
class EmailResult:
    ok: bool
    error: Optional[str]


# Respect ALERT_COOLDOWN_SECONDS — use a module-level timestamp to prevent spam.
LAST_EMAIL_SENT_AT = 0.0


def send_email_alert(subject: str, body: str, snapshot_bytes: Optional[bytes] = None) -> EmailResult:
    """Send a plain text email alert with optional snapshot attachment."""
    global LAST_EMAIL_SENT_AT
    settings = get_settings()
    if not settings.email_enabled:
        return EmailResult(ok=False, error="Email disabled")

    # Cooldown check
    cooldown = float(settings.alert_cooldown_seconds)
    now = time.time()
    if now - LAST_EMAIL_SENT_AT < cooldown:
        logger.info("Email alert throttled due to cooldown")
        return EmailResult(ok=False, error="Cooldown active")

    EMAIL_USER = settings.email_user
    EMAIL_PASSWORD = settings.email_password
    EMAIL_RECIPIENT = settings.email_recipient
    EMAIL_SMTP_HOST = settings.email_smtp_host
    EMAIL_SMTP_PORT = settings.email_smtp_port

    if not (EMAIL_USER and EMAIL_PASSWORD and EMAIL_RECIPIENT):
        return EmailResult(ok=False, error="Missing email configuration credentials")

    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECIPIENT
    msg.attach(MIMEText(body, 'plain'))
    if snapshot_bytes:
        img = MIMEImage(snapshot_bytes)
        img.add_header('Content-Disposition', 'attachment', filename='snapshot.jpg')
        msg.attach(img)

    try:
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, [EMAIL_RECIPIENT], msg.as_string())
        LAST_EMAIL_SENT_AT = now
        logger.info("Email alert sent successfully.")
        return EmailResult(ok=True, error=None)
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return EmailResult(ok=False, error=str(e))


class EmailAlert:
    """Send a styled HTML email."""

    def __init__(self, *, smtp_host: str, smtp_port: int, user: str, password: str, recipient: str) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = int(smtp_port)
        self.user = user
        self.password = password
        self.recipient = recipient
        self.template_path = Path("alerts/templates/email_template.html")

    def send(
        self,
        *,
        subject: str,
        timestamp: str,
        location: str,
        class_name: str,
        confidence_pct: float,
        risk_score: float,
        risk_severity: str,
        llm_summary: str,
    ) -> EmailResult:
        """Send an HTML email using the local template."""

        try:
            html = self.template_path.read_text(encoding="utf-8")
        except Exception as e:
            return EmailResult(ok=False, error=f"Email template missing: {e}")

        try:
            body = html.format(
                timestamp=timestamp,
                location=location,
                class_name_upper=class_name.upper(),
                confidence_pct=float(confidence_pct),
                risk_score=float(risk_score),
                risk_severity=risk_severity,
                llm_summary=llm_summary,
            )
        except Exception as e:
            return EmailResult(ok=False, error=f"Template format error: {e}")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.user
        msg["To"] = self.recipient
        msg.attach(MIMEText(body, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(self.user, self.password)
                server.sendmail(self.user, [self.recipient], msg.as_string())
            logger.info("Email alert sent.")
            return EmailResult(ok=True, error=None)
        except Exception as e:
            logger.warning(f"Email alert failed: {e}")
            return EmailResult(ok=False, error=str(e))

