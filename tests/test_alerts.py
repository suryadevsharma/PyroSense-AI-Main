"""Alert system tests with mocked channels."""

from __future__ import annotations

import asyncio

from alerts.alert_manager import AlertManager


def test_alert_manager_runs_without_enabled_channels(monkeypatch) -> None:
    """Ensure alert manager doesn't crash when channels disabled."""

    am = AlertManager()
    payload = {
        "primary_class": "fire",
        "ensemble_conf": 0.9,
        "risk": {"score": 80, "severity": "HIGH"},
        "timestamp": "2026-01-01T00:00:00Z",
        "llm_summary": "Test summary.",
        "detections": [],
        "inference_time_ms": 10.0,
    }

    asyncio.get_event_loop().run_until_complete(am.trigger_alert(payload, detection_id=1, location="Test"))


def test_settings_sanitizes_credentials() -> None:
    """Ensure settings fields strip carriage returns, quotes, and whitespace."""
    from config.settings import Settings
    s = Settings(
        TELEGRAM_BOT_TOKEN=" 'token123\r' ",
        TELEGRAM_CHAT_ID=' "chat_id_abc\n" ',
        EMAIL_USER=" 'user@example.com' ",
        WEBHOOK_URL=" http://example.com/webhook\r "
    )
    assert s.telegram_bot_token == "token123"
    assert s.telegram_chat_id == "chat_id_abc"
    assert s.email_user == "user@example.com"
    assert str(s.webhook_url).rstrip("/") == "http://example.com/webhook"

