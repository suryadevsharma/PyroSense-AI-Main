import pytest
import asyncio
from datetime import datetime, timedelta
from alerts.alert_manager import AlertManager, IncidentStatus, IncidentState
from config.settings import get_settings
from database.session import SessionLocal
from database.models import AlertLog, Detection
from database import crud
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select


@pytest.mark.asyncio
async def test_alert_state_machine_scenarios():
    # Force settings
    settings = get_settings()
    settings.alert_cooldown_seconds = 60
    settings.min_consecutive_detections = 3
    settings.incident_clear_seconds = 30
    settings.max_alerts_per_incident = 3

    # Initialize shared AlertManager instance
    am = AlertManager.get_shared_instance()
    am._email = MagicMock()
    am._telegram = MagicMock()
    am._webhook = MagicMock()

    # Define test payloads
    fire_payload_med = {
        "primary_class": "fire",
        "ensemble_conf": 0.9,
        "risk": {"score": 50, "severity": "MEDIUM"},
        "timestamp": "2026-06-26T23:36:00Z",
    }
    
    fire_payload_crit = {
        "primary_class": "fire",
        "ensemble_conf": 0.95,
        "risk": {"score": 90, "severity": "CRITICAL"},
        "timestamp": "2026-06-26T23:36:05Z",
    }

    safe_payload = {
        "primary_class": "none",
        "ensemble_conf": 0.05,
        "risk": {"score": 0, "severity": "LOW"},
        "timestamp": "2026-06-26T23:36:10Z",
    }

    # Helper function to mock dispatch and run state checks
    async def run_scenario_tests():
        # Clean DB
        with SessionLocal() as db:
            db.query(AlertLog).delete()
            db.query(Detection).delete()
            db.commit()

        # Reset singleton state
        am.reset_state()
        
        # Mock actual dispatch methods
        am._send_email = AsyncMock()
        am._send_telegram = AsyncMock()
        am._send_webhook = AsyncMock()

        # ==========================================
        # Scenario 4: False positive lasting one frame
        # ==========================================
        # Frame 1: Detection
        await am.trigger_alert(fire_payload_med, detection_id=1, location="TestLoc", source="TestCam")
        assert am._send_email.call_count == 0
        assert am._send_telegram.call_count == 0
        assert am._send_webhook.call_count == 0
        assert am._incidents["TestCam:TestLoc"].consecutive_detections == 1
        assert am._incidents["TestCam:TestLoc"].status == IncidentStatus.SAFE

        # Frame 2: Safe frame (should reset counter)
        await am.trigger_alert(safe_payload, detection_id=2, location="TestLoc", source="TestCam")
        assert am._send_email.call_count == 0
        assert am._incidents["TestCam:TestLoc"].consecutive_detections == 0
        assert am._incidents["TestCam:TestLoc"].status == IncidentStatus.SAFE

        # ==========================================
        # Scenario 1: Continuous fire video for 3 minutes
        # ==========================================
        # Frame 1 of fire
        await am.trigger_alert(fire_payload_med, detection_id=3, location="TestLoc", source="TestCam")
        # Frame 2 of fire
        await am.trigger_alert(fire_payload_med, detection_id=4, location="TestLoc", source="TestCam")
        # Frame 3 of fire (reaches min_consecutive threshold of 3)
        await am.trigger_alert(fire_payload_med, detection_id=5, location="TestLoc", source="TestCam")
        
        assert am._send_email.call_count == 1
        assert am._send_telegram.call_count == 1
        assert am._send_webhook.call_count == 1
        assert am._incidents["TestCam:TestLoc"].status == IncidentStatus.ACTIVE
        assert am._incidents["TestCam:TestLoc"].alerts_sent_count == 1

        # Subsequent frames (Frames 4 through 10) during continuous fire (no duplicates)
        for i in range(6, 13):
            await am.trigger_alert(fire_payload_med, detection_id=i, location="TestLoc", source="TestCam")
            
        assert am._send_email.call_count == 1
        assert am._send_telegram.call_count == 1
        assert am._send_webhook.call_count == 1
        assert am._incidents["TestCam:TestLoc"].alerts_sent_count == 1

        # ==========================================
        # Scenario 3: Risk escalates from Moderate (Medium) to Critical
        # ==========================================
        # Escalation frame
        await am.trigger_alert(fire_payload_crit, detection_id=13, location="TestLoc", source="TestCam")
        assert am._send_email.call_count == 2
        assert am._send_telegram.call_count == 2
        assert am._send_webhook.call_count == 2
        assert am._incidents["TestCam:TestLoc"].max_severity == "CRITICAL"
        assert am._incidents["TestCam:TestLoc"].alerts_sent_count == 2

        # ==========================================
        # Scenario 2: Fire disappears and returns after clear interval
        # ==========================================
        # Simulate clear interval passing (35 seconds in the past)
        am._incidents["TestCam:TestLoc"].last_detection_time = datetime.utcnow() - timedelta(seconds=35)
        
        # Send safe frame -> should transition to RESOLVED and send a resolution notification
        await am.trigger_alert(safe_payload, detection_id=14, location="TestLoc", source="TestCam")
        assert am._incidents["TestCam:TestLoc"].status == IncidentStatus.RESOLVED
        assert am._send_email.call_count == 3  # +1 resolution
        assert am._send_telegram.call_count == 3
        
        # Fire returns: send 3 consecutive fire frames
        await am.trigger_alert(fire_payload_med, detection_id=15, location="TestLoc", source="TestCam")
        await am.trigger_alert(fire_payload_med, detection_id=16, location="TestLoc", source="TestCam")
        await am.trigger_alert(fire_payload_med, detection_id=17, location="TestLoc", source="TestCam")
        
        assert am._incidents["TestCam:TestLoc"].status == IncidentStatus.ACTIVE
        assert am._send_email.call_count == 4  # +1 new alert
        assert am._send_telegram.call_count == 4
        assert am._send_webhook.call_count == 4

    await run_scenario_tests()
