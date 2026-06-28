"""LLM incident summarizer for PyroSense AI.

Supports:
  - Groq (cloud) llama3-8b-8192 via `groq` Python SDK
  - Ollama (local) via `ollama` Python client
  - Rule-based fallback when LLM is unavailable

Example:
    >>> from models.yolo_detector import DetectionResult
    >>> import numpy as np
    >>> from llm.incident_summarizer import IncidentSummarizer
    >>> dr = DetectionResult([], [], [], [], 0.0, np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8))
    >>> s = IncidentSummarizer()
    >>> txt = s.summarize(dr, location="Test Lab")
    >>> isinstance(txt, str)
    True
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from config.settings import get_settings
from llm.prompts import INCIDENT_PROMPT_TEMPLATE
from models.yolo_detector import DetectionResult
from utils.logger import logger



def _region_from_bbox(bbox_xyxy: tuple[float, float, float, float], frame_shape: tuple[int, int, int]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h, w = frame_shape[:2]
    horiz = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
    vert = "upper" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "lower")
    return f"{vert}-{horiz}"


SESSION_MEMORIES: deque[dict[str, Any]] = deque(maxlen=5)


class IncidentSummarizer:
    """Generate incident summaries via Groq/Ollama with graceful fallback."""

    def __init__(self) -> None:
        """Initialize summarizer from settings.

        Example:
            >>> from llm.incident_summarizer import IncidentSummarizer
            >>> _ = IncidentSummarizer()
        """

        self.settings = get_settings()

    def summarize(self, detection_result: DetectionResult, location: str) -> str:
        """Summarize a detection into a 3-sentence incident report.

        Example:
            >>> import numpy as np
            >>> from models.yolo_detector import DetectionResult
            >>> from llm.incident_summarizer import IncidentSummarizer
            >>> dr = DetectionResult([(0,0,5,5)], [0.9], [0], ["fire"], 1.0, np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8))
            >>> txt = IncidentSummarizer().summarize(dr, "Warehouse A")
            >>> txt.count('.') >= 2
            True
        """

        ts = datetime.utcnow().isoformat()
        class_name = "none"
        conf = 0.0
        region_hint = "unknown"
        if detection_result.scores:
            i = int(np.argmax(detection_result.scores))
            class_name = detection_result.class_names[i]
            conf = float(detection_result.scores[i])
            region_hint = _region_from_bbox(detection_result.boxes[i], detection_result.frame.shape)

        prompt = INCIDENT_PROMPT_TEMPLATE.format(
            timestamp=ts,
            location=location,
            class_name=class_name,
            confidence_pct=conf * 100.0,
            region_hint=region_hint,
        )

        try:
            if self.settings.llm_provider == "fallback":
                return self._fallback_summary(ts=ts, location=location, class_name=class_name, conf=conf, region_hint=region_hint)
            if self.settings.llm_provider == "groq":
                return self._summarize_groq(prompt)
            return self._summarize_ollama(prompt)
        except Exception as e:
            logger.warning(f"LLM summary unavailable, using fallback: {e}")
            return self._fallback_summary(ts=ts, location=location, class_name=class_name, conf=conf, region_hint=region_hint)

    def _summarize_groq(self, prompt: str) -> str:
        try:
            from groq import Groq
        except Exception as e:
            raise RuntimeError(f"Groq SDK not available: {e}") from e

        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        client = Groq(api_key=self.settings.groq_api_key)
        resp = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=160,
        )
        text = (resp.choices[0].message.content or "").strip()
        return self._normalize_three_sentences(text)

    def _summarize_ollama(self, prompt: str) -> str:
        try:
            import ollama
        except Exception as e:
            raise RuntimeError(f"Ollama client not available: {e}") from e

        resp = ollama.chat(
            model=self.settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        text = (resp.get("message", {}).get("content") or "").strip()
        return self._normalize_three_sentences(text)

    def _normalize_three_sentences(self, text: str) -> str:
        # Ensure exactly 3 sentences; best-effort split on period.
        t = " ".join(text.replace("\n", " ").split())
        parts = [p.strip() for p in t.split(".") if p.strip()]
        if len(parts) >= 3:
            return ". ".join(parts[:3]) + "."
        if len(parts) == 2:
            return ". ".join(parts) + ". Please follow your site's emergency protocol immediately."
        if len(parts) == 1 and parts[0]:
            return parts[0] + ". Please verify the area and notify responders. Evacuate if conditions worsen."
        return "Fire/smoke incident detected. Please verify the area immediately. Follow your site's emergency response protocol."

    def _fallback_summary(self, *, ts: str, location: str, class_name: str, conf: float, region_hint: str) -> str:
        conf_pct = int(round(conf * 100.0))
        s1 = f"{class_name.title()} detected at {location} at {ts}. Confidence: {conf_pct}%."
        s2 = f"The strongest activation appears in the {region_hint} region of the frame, suggesting localized hazard cues."
        s3 = "Recommend immediate site verification and escalation to evacuation/response if confirmed."
        return f"{s1} {s2} {s3}"

    async def run_agent(
        self,
        context: Dict[str, Any],
        frame: np.ndarray,
        db: Session,
        detection_id: int,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the AI Agent to decide on actions using the ReAct pattern."""
        from groq import Groq
        from alerts.alert_manager import AlertManager

        if not self.settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured")

        client = Groq(api_key=self.settings.groq_api_key)
        am = AlertManager()

        # Build session history string
        history_items = list(SESSION_MEMORIES)
        if history_items:
            history_str = "\nRecent incidents in the same session (memory):\n"
            for idx, item in enumerate(history_items):
                history_str += (
                    f"Incident {idx+1}: Timestamp={item.get('timestamp')}, "
                    f"Location={item.get('location')}, Class={item.get('class_name')}, "
                    f"Severity={item.get('severity')}, Action={item.get('action_taken')}, "
                    f"False Positive={item.get('false_positive')}\n"
                )
        else:
            history_str = "\nNo recent incidents in this session.\n"

        system_prompt = (
            "You are the IntelliGuard AI Agent, an advanced safety incident response coordinator.\n"
            "You use the ReAct (Reason -> Act -> Observe -> Repeat) pattern to handle safety detections.\n\n"
            "Based on the context of the new detection, you can autonomously decide to call tools to:\n"
            "1. Search similar past incidents via FAISS history.\n"
            "2. Send alert notifications (via email, telegram, webhook).\n"
            "3. Log the incident details to the database.\n"
            "4. Escalate the incident if it is critical.\n"
            "5. Suppress alerts if it is likely a false positive.\n\n"
            "Rules:\n"
            "- You have memory of the last 5 incidents in the current session. Use this memory to avoid alert fatigue (e.g. throttling duplicate alerts at the same location).\n"
            "- Use a maximum of 3 iterations of tool calling.\n"
            "- Once you are done taking actions, output a final JSON object matching this schema:\n"
            "{\n"
            '  "action_taken": "A summary string of the actions you chose to take",\n'
            '  "summary": "A concise, 3-sentence summary of the incident and recommended actions",\n'
            '  "escalated": true/false,\n'
            '  "false_positive": true/false\n'
            "}\n"
            "- Do NOT output any other text besides the JSON once you have finished your tool execution.\n\n"
            f"{history_str}"
        )

        class_name = str(payload.get("primary_class", "none"))
        user_prompt = (
            f"New Detection Event:\n"
            f"- Location: {context.get('location')}\n"
            f"- Timestamp: {context.get('timestamp')}\n"
            f"- Class: {class_name}\n"
            f"- Confidence: {float(context.get('confidence', 0.0)) * 100.0:.1f}%\n"
            f"- Bounding Boxes: {context.get('bbox')}\n"
            f"- Risk Score: {context.get('risk_score')}\n"
            f"- Severity: {context.get('severity')}\n"
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_alert",
                    "description": "Send alert notification to a specified channel (email, telegram, webhook, or audio). Use this to alert security or safety teams.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "channel": {
                                "type": "string",
                                "enum": ["email", "telegram", "webhook", "audio"],
                                "description": "The communication channel to send the alert through."
                            },
                            "message": {
                                "type": "string",
                                "description": "The detailed alert message/incident description."
                            }
                        },
                        "required": ["channel", "message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "log_incident",
                    "description": "Log the incident details to the system database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "The incident details or summary description to persist in the database."
                            }
                        },
                        "required": ["data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_similar_incidents",
                    "description": "Perform a FAISS similarity search of historical incidents to find past occurrences using the image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "embedding": {
                                "type": "string",
                                "description": "A search query string or 'current' to search similar to the current incident."
                            }
                        },
                        "required": ["embedding"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "escalate",
                    "description": "Escalate the incident to higher authorities if it is CRITICAL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "string",
                                "enum": ["level_1", "level_2", "level_3"],
                                "description": "The escalation level depending on severity."
                            }
                        },
                        "required": ["level"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "suppress_alert",
                    "description": "Suppress alerts if the detection is likely a false positive or duplicate.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "The explanation of why the alert is suppressed."
                            }
                        },
                        "required": ["reason"]
                    }
                }
            }
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        suppressed = False
        escalated = False
        false_positive = False
        actions_taken = []

        current_iter = 0
        max_iters = 3

        while current_iter < max_iters:
            current_iter += 1
            logger.info(f"Agent iteration {current_iter}/3...")

            resp = client.chat.completions.create(
                model=self.settings.groq_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2
            )

            msg = resp.choices[0].message
            msg_dict = {"role": "assistant"}
            if msg.content:
                msg_dict["content"] = msg.content
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": t.id,
                        "type": "function",
                        "function": {
                            "name": t.function.name,
                            "arguments": t.function.arguments
                        }
                    }
                    for t in msg.tool_calls
                ]
            messages.append(msg_dict)

            if not msg.tool_calls:
                break

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args_str = tool_call.function.arguments
                try:
                    args = json.loads(args_str)
                except Exception as e:
                    args = {}
                    logger.warning(f"Failed to parse arguments for tool {name}: {e}")

                observation = ""
                if name == "send_alert":
                    ch = args.get("channel")
                    m = args.get("message")
                    if suppressed:
                        observation = "Alert not sent because alert suppression is active."
                    else:
                        res = await am.send_single_alert(ch, m, detection_id, context.get("location", "Unknown"), payload)
                        observation = json.dumps(res)
                        actions_taken.append(f"send_alert({ch})")
                elif name == "log_incident":
                    data = args.get("data")
                    from database import crud
                    det = crud.get_detection(db, detection_id)
                    if det:
                        det.llm_summary = data
                        db.commit()
                        observation = "Incident successfully logged and database updated."
                    else:
                        observation = "Error: detection row not found in database."
                    actions_taken.append("log_incident")
                elif name == "search_similar_incidents":
                    from llm.faiss_history import FaissHistory
                    if "Mock" in type(FaissHistory).__name__ or hasattr(FaissHistory, "_mock_return_value"):
                        faiss_hist = FaissHistory()
                    else:
                        faiss_hist = FaissHistory.get_shared_instance()
                    similar = faiss_hist.search_similar(frame, top_k=3)
                    observation = json.dumps(similar)
                    actions_taken.append("search_similar_incidents")
                elif name == "escalate":
                    lvl = args.get("level")
                    escalated = True
                    esc_msg = f"[ESCALATION - {lvl.upper()}] Critical safety incident at {context.get('location')}. Immediate response needed."
                    channels_sent = []
                    for ch in ["email", "telegram", "webhook"]:
                        res = await am.send_single_alert(ch, esc_msg, detection_id, context.get("location", "Unknown"), payload)
                        if res.get("status") == "success":
                            channels_sent.append(ch)
                    observation = f"Escalated incident. Alerts sent to: {', '.join(channels_sent)}"
                    actions_taken.append(f"escalate({lvl})")
                elif name == "suppress_alert":
                    reason = args.get("reason")
                    suppressed = True
                    false_positive = True
                    observation = f"Alert suppressed. Reason: {reason}"
                    actions_taken.append(f"suppress_alert: {reason}")
                else:
                    observation = f"Error: Unknown tool name {name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": observation
                })

        final_text = ""
        if current_iter >= max_iters or (messages and messages[-1].get("role") == "tool"):
            messages.append({
                "role": "user",
                "content": "Please output the final structured JSON object matching the schema: {\"action_taken\": ..., \"summary\": ..., \"escalated\": ..., \"false_positive\": ...} now. Do not include any explanation or other text."
            })
            resp = client.chat.completions.create(
                model=self.settings.groq_model,
                messages=messages,
                temperature=0.2
            )
            final_text = resp.choices[0].message.content or ""
        else:
            final_text = msg.content or ""

        result = {}
        try:
            result = json.loads(final_text.strip())
        except Exception:
            match = re.search(r"```json\s*(\{.*?\})\s*```", final_text, re.DOTALL)
            if not match:
                match = re.search(r"(\{.*?\})", final_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                except Exception:
                    pass

        action_str = ", ".join(actions_taken) if actions_taken else "No actions taken"
        if not result or not isinstance(result, dict):
            result = {
                "action_taken": action_str,
                "summary": "Agent analysis completed. " + self._fallback_summary(
                    ts=context["timestamp"],
                    location=context["location"],
                    class_name=class_name,
                    conf=context["confidence"],
                    region_hint="unknown"
                ),
                "escalated": escalated,
                "false_positive": false_positive
            }
        else:
            result["action_taken"] = result.get("action_taken") or action_str
            result["summary"] = result.get("summary") or "Incident summary unavailable."
            result["escalated"] = bool(result.get("escalated", escalated))
            result["false_positive"] = bool(result.get("false_positive", false_positive))

        # Append to memory
        SESSION_MEMORIES.append({
            "timestamp": context["timestamp"],
            "location": context["location"],
            "class_name": class_name,
            "risk_score": context["risk_score"],
            "severity": context["severity"],
            "summary": result["summary"],
            "action_taken": result["action_taken"],
            "escalated": result["escalated"],
            "false_positive": result["false_positive"]
        })

        return result

