"""API endpoint tests for PyroSense AI."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "db_ok" in data


def test_detect_endpoint_with_test_image() -> None:
    client = TestClient(app)
    img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    data = {"location": "UnitTest", "source": "test"}
    r = client.post("/api/v1/detect", files=files, data=data)
    assert r.status_code == 200
    payload = r.json()
    assert "detections" in payload
    assert "risk" in payload


import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llm.incident_summarizer import IncidentSummarizer, SESSION_MEMORIES


@pytest.mark.asyncio
async def test_agent_react_loop_and_tools():
    """Test the agent's run_agent ReAct loop with mocked Groq SDK client."""
    summarizer = IncidentSummarizer()

    # Mock settings
    summarizer.settings = MagicMock()
    summarizer.settings.groq_api_key = "test_key"
    summarizer.settings.groq_model = "llama3-8b-8192"

    # Mock Groq client
    mock_client = MagicMock()

    # 1st response: Model decides to call send_alert and search_similar_incidents tools
    mock_tool_call_1 = MagicMock()
    mock_tool_call_1.id = "call_1"
    mock_tool_call_1.function.name = "send_alert"
    mock_tool_call_1.function.arguments = '{"channel": "telegram", "message": "Fire detected in Room A!"}'

    mock_tool_call_2 = MagicMock()
    mock_tool_call_2.id = "call_2"
    mock_tool_call_2.function.name = "search_similar_incidents"
    mock_tool_call_2.function.arguments = '{"embedding": "current"}'

    mock_msg_1 = MagicMock()
    mock_msg_1.content = "I should search for similar incidents and send a Telegram alert."
    mock_msg_1.tool_calls = [mock_tool_call_1, mock_tool_call_2]

    # 2nd response: Model produces final JSON content
    mock_msg_2 = MagicMock()
    mock_msg_2.content = '{"action_taken": "Sent alert via telegram and searched similar incidents", "summary": "Fire detected in Room A at 2026-06-26. Standard protocol initiated. Evacuate immediately.", "escalated": false, "false_positive": false}'
    mock_msg_2.tool_calls = []

    mock_choice_1 = MagicMock()
    mock_choice_1.message = mock_msg_1

    mock_choice_2 = MagicMock()
    mock_choice_2.message = mock_msg_2

    mock_resp_1 = MagicMock()
    mock_resp_1.choices = [mock_choice_1]

    mock_resp_2 = MagicMock()
    mock_resp_2.choices = [mock_choice_2]

    mock_client.chat.completions.create.side_effect = [mock_resp_1, mock_resp_2]

    # Mock AlertManager
    mock_am = MagicMock()
    mock_am.send_single_alert = AsyncMock(return_value={"status": "success"})

    # Clear memories first
    SESSION_MEMORIES.clear()

    # Mock FaissHistory
    mock_faiss = MagicMock()
    mock_faiss.search_similar.return_value = [{"detection_id": 1, "score": 0.9}]

    # Run run_agent
    with patch("groq.Groq", return_value=mock_client), \
         patch("alerts.alert_manager.AlertManager", return_value=mock_am), \
         patch("llm.faiss_history.FaissHistory", return_value=mock_faiss):

        context = {
            "risk_score": 80.0,
            "severity": "HIGH",
            "bbox": [(0, 0, 10, 10)],
            "confidence": 0.95,
            "timestamp": "2026-06-26T23:36:00Z",
            "location": "Room A"
        }
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        db = MagicMock()
        payload = {"primary_class": "fire"}

        res = await summarizer.run_agent(context, frame, db, detection_id=42, payload=payload)

        # Assertions
        assert res["escalated"] is False
        assert res["false_positive"] is False
        assert "Room A" in res["summary"]
        assert len(SESSION_MEMORIES) == 1
        assert SESSION_MEMORIES[0]["location"] == "Room A"
        assert mock_am.send_single_alert.call_count == 1
        mock_am.send_single_alert.assert_called_with("telegram", "Fire detected in Room A!", 42, "Room A", payload)


def test_fallback_flow_on_agent_failure(monkeypatch):
    """Test that detection router correctly falls back to standard pipeline if agent raises an error."""
    client = TestClient(app)

    # Force agent_flow to fail or raise an exception
    async def mock_run_agent_failed(*args, **kwargs):
        raise RuntimeError("Groq Connection Error")

    monkeypatch.setattr(IncidentSummarizer, "run_agent", mock_run_agent_failed)
    # Mock settings to enable groq provider so the agent block is entered
    from config.settings import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "dummy_key")

    img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    data = {"location": "FallbackTest", "source": "test"}

    # Post to detect endpoint
    r = client.post("/api/v1/detect", files=files, data=data)
    assert r.status_code == 200
    payload = r.json()
    assert "detections" in payload
    assert "risk" in payload
    # Check that LLM summary is generated via fallback
    assert payload["llm_summary"] != ""
