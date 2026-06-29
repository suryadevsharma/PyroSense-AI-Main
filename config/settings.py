"""Typed application settings for PyroSense AI.

This module centralizes configuration for the API, dashboard, models, alerts,
LLM provider, and MLflow. Values are loaded from environment variables (e.g.
via a `.env` file) using `pydantic-settings`.

Example:
    >>> from config.settings import get_settings
    >>> s = get_settings()
    >>> s.conf_threshold
    0.5
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional


def _default_database_url() -> str:
    """SQLite in CWD when writable; otherwise /tmp (Streamlit Cloud, read-only mounts)."""

    try:
        test = Path.cwd() / ".intelliguard_write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return "sqlite:///./pyrosense.db"
    except OSError:
        return "sqlite:////tmp/pyrosense.db"

from pydantic import AliasChoices, Field, HttpUrl, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Model
    # Default to the pretrained fire/smoke weights (auto-downloaded if missing).
    yolo_model_path: str = Field(default="models/weights/fire_smoke_yolov8.pt", alias="YOLO_MODEL_PATH")
    conf_threshold: float = Field(default=0.25, alias="CONF_THRESHOLD", ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, alias="IOU_THRESHOLD", ge=0.0, le=1.0)
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(default="auto", alias="DEVICE")
    enable_efficientnet: bool = Field(
        default=False,
        validation_alias=AliasChoices("ENABLE_EFFICIENTNET", "enable_efficientnet"),
        description="Secondary EfficientNet verifier (slow first load; set false for YOLO-only, faster startup).",
    )

    # Database
    database_url: str = Field(default_factory=_default_database_url, alias="DATABASE_URL")

    # LLM
    llm_provider: Literal["groq", "ollama", "fallback"] = Field(default="groq", alias="LLM_PROVIDER")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-8b-8192", alias="GROQ_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")

    # Alerts
    alert_cooldown_seconds: PositiveInt = Field(default=60, alias="ALERT_COOLDOWN_SECONDS")
    min_consecutive_detections: int = Field(default=3, alias="MIN_CONSECUTIVE_DETECTIONS")
    incident_clear_seconds: int = Field(default=30, alias="INCIDENT_CLEAR_SECONDS")
    max_alerts_per_incident: int = Field(default=3, alias="MAX_ALERTS_PER_INCIDENT")
    email_enabled: bool = Field(default=False, alias="EMAIL_ENABLED")
    email_smtp_host: str = Field(default="smtp.gmail.com", alias="EMAIL_SMTP_HOST")
    email_smtp_port: int = Field(default=587, alias="EMAIL_SMTP_PORT", ge=1, le=65535)
    email_user: Optional[str] = Field(default=None, alias="EMAIL_USER")
    email_password: Optional[str] = Field(default=None, alias="EMAIL_PASSWORD")
    email_recipient: Optional[str] = Field(default=None, alias="EMAIL_RECIPIENT")

    telegram_enabled: bool = Field(default=False, alias="TELEGRAM_ENABLED")
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")

    webhook_enabled: bool = Field(default=False, alias="WEBHOOK_ENABLED")
    webhook_url: Optional[HttpUrl] = Field(default=None, alias="WEBHOOK_URL")

    # MLflow
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")

    # Risk Weights (overridable in settings UI)
    risk_w_conf: float = Field(default=0.4, alias="RISK_W_CONF")
    risk_w_area: float = Field(default=0.3, alias="RISK_W_AREA")
    risk_w_growth: float = Field(default=0.2, alias="RISK_W_GROWTH")
    risk_w_smoke: float = Field(default=0.1, alias="RISK_W_SMOKE")

    @field_validator("telegram_bot_token", "telegram_chat_id", "email_user", "email_password", "email_recipient", "groq_api_key", mode="before")
    @classmethod
    def clean_credentials(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            # Strip standard quotes, trailing spaces, returns, and non-printable characters
            cleaned = "".join(ch for ch in v.strip() if ch.isprintable())
            cleaned = cleaned.strip("'\"")
            return cleaned
        return v

    @field_validator("webhook_url", mode="before")
    @classmethod
    def clean_webhook_url(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = "".join(ch for ch in v.strip() if ch.isprintable())
            cleaned = cleaned.strip("'\"")
            return cleaned
        return v

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    snapshots_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed" / "snapshots")
    heatmaps_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed" / "heatmaps")
    faiss_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed" / "faiss")

    def ensure_dirs(self) -> None:
        """Create directories required for runtime artifacts.

        Example:
            >>> from config.settings import get_settings
            >>> get_settings().ensure_dirs()
        """

        for p in (self.snapshots_dir, self.heatmaps_dir, self.faiss_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance.

    Example:
        >>> from config.settings import get_settings
        >>> s1 = get_settings()
        >>> s2 = get_settings()
        >>> s1 is s2
        True
    """

    s = Settings()
    s.ensure_dirs()

    # Load settings overrides from settings_override.json if it exists
    override_path = s.project_root / "data" / "processed" / "settings_override.json"
    if override_path.exists():
        try:
            import json
            data = json.loads(override_path.read_text(encoding="utf-8"))
            for k, v in data.items():
                field_name = k.lower()
                if hasattr(s, field_name):
                    # Convert to appropriate types
                    field_type = s.__annotations__.get(field_name)
                    if field_type is bool:
                        setattr(s, field_name, bool(v))
                    elif field_type is float:
                        setattr(s, field_name, float(v))
                    elif field_type is int:
                        setattr(s, field_name, int(v))
                    else:
                        setattr(s, field_name, v)
        except Exception as e:
            from utils.logger import logger
            logger.warning(f"Failed to load settings override JSON: {e}")

    if s.llm_provider == "groq" and not s.groq_api_key:
        from utils.logger import logger
        logger.warning("GROQ_API_KEY is missing. Falling back to rule-based summary (llm_provider='fallback').")
        s.llm_provider = "fallback"
    return s


def efficientnet_enabled() -> bool:
    """Whether the secondary EfficientNet verifier is on (safe if Settings predates this field)."""

    return bool(getattr(get_settings(), "enable_efficientnet", True))

