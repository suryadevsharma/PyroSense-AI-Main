"""Telegram alert channel + bot command handlers.

The alert channel sends push messages. The bot command handlers implement:
  - /status
  - /snapshot
  - /history
  - /threshold 0.7

Example:
    >>> from alerts.telegram_alert import TelegramAlert
    >>> _ = TelegramAlert(bot_token="token", chat_id="123")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from config.settings import get_settings
from database import crud
from database.session import SessionLocal
from utils.image_utils import encode_image_base64_jpeg
from utils.logger import logger


@dataclass
class TelegramResult:
    ok: bool
    error: Optional[str]


class TelegramAlert:
    """Send Telegram message/photo notifications."""

    def __init__(self, *, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def send_message(self, text: str) -> TelegramResult:
        try:
            from telegram import Bot
            from telegram.error import TelegramError
            from telegram.request import HTTPXRequest
        except Exception as e:
            return TelegramResult(ok=False, error=f"python-telegram-bot unavailable: {e}")

        try:
            req = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
            async with Bot(token=self.bot_token, request=req) as bot:
                await bot.send_message(chat_id=self.chat_id, text=text)
            return TelegramResult(ok=True, error=None)
        except TelegramError as e:
            logger.warning(f"Telegram send_message failed (TelegramError): {e}")
            return TelegramResult(ok=False, error=str(e))
        except Exception as e:
            logger.warning(f"Telegram send_message failed: {e}")
            return TelegramResult(ok=False, error=str(e))

    async def send_photo(self, image_bgr, caption: str) -> TelegramResult:
        try:
            from telegram import Bot
            from telegram.error import TelegramError
            from telegram.request import HTTPXRequest
        except Exception as e:
            return TelegramResult(ok=False, error=f"python-telegram-bot unavailable: {e}")

        try:
            import base64
            from io import BytesIO

            b64 = encode_image_base64_jpeg(image_bgr)
            buf = BytesIO(base64.b64decode(b64))
            req = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
            async with Bot(token=self.bot_token, request=req) as bot:
                await bot.send_photo(chat_id=self.chat_id, photo=buf, caption=caption)
            return TelegramResult(ok=True, error=None)
        except TelegramError as e:
            logger.warning(f"Telegram send_photo failed (TelegramError): {e}")
            return TelegramResult(ok=False, error=str(e))
        except Exception as e:
            logger.warning(f"Telegram send_photo failed: {e}")
            return TelegramResult(ok=False, error=str(e))


def build_telegram_application(get_snapshot_bgr: Callable[[], Any]):
    """Create a Telegram `Application` with required command handlers.

    The returned application can be started by calling:
      `await app.initialize(); await app.start(); await app.updater.start_polling()`

    Example:
        >>> from alerts.telegram_alert import build_telegram_application
        >>> _ = build_telegram_application(lambda: None)
    """

    settings = get_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes
    except Exception as e:
        raise RuntimeError(f"python-telegram-bot required: {e}") from e

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        with SessionLocal() as db:
            det = crud.latest_detection(db)
        if det is None:
            await update.message.reply_text("PyroSense AI status: OK. No detections yet.")
            return
        await update.message.reply_text(
            f"PyroSense AI status: OK.\nLast detection: {det.class_name} ({det.confidence:.2f}) at {det.timestamp.isoformat()} UTC\nRisk: {det.risk_score:.0f}"
        )

    async def cmd_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        img = get_snapshot_bgr()
        if img is None:
            await update.message.reply_text("Snapshot unavailable.")
            return
        try:
            import base64
            from io import BytesIO
            from telegram import InputFile

            b64 = encode_image_base64_jpeg(img)
            buf = BytesIO(base64.b64decode(b64))
            await update.message.reply_photo(photo=buf, caption="Current snapshot")
        except Exception as e:
            await update.message.reply_text(f"Failed to send snapshot: {e}")

    async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        with SessionLocal() as db:
            rows = crud.last_n_detections(db, n=5)
        if not rows:
            await update.message.reply_text("No detections recorded yet.")
            return
        lines = ["Last 5 detections:"]
        for r in rows:
            lines.append(f"- {r.timestamp.isoformat()} UTC | {r.class_name} | conf={r.confidence:.2f} | risk={r.risk_score:.0f}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not context.args:
            await update.message.reply_text("Usage: /threshold 0.7")
            return
        try:
            value = float(context.args[0])
            if not (0.0 <= value <= 1.0):
                raise ValueError("must be between 0 and 1")
        except Exception as e:
            await update.message.reply_text(f"Invalid threshold: {e}")
            return

        # Persist in a small JSON settings override file used by dashboard/API.
        override_path = Path("data/processed/settings_override.json")
        override_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if override_path.exists():
            try:
                data = json.loads(override_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data["CONF_THRESHOLD"] = value
        override_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        await update.message.reply_text(f"Updated confidence threshold override to {value:.2f}. Restart services to apply.")

    try:
        from telegram.error import TelegramError
        from telegram.request import HTTPXRequest
        req = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
        app = Application.builder().token(settings.telegram_bot_token).request(req).build()
        app.add_handler(CommandHandler("status", cmd_status))
        app.add_handler(CommandHandler("snapshot", cmd_snapshot))
        app.add_handler(CommandHandler("history", cmd_history))
        app.add_handler(CommandHandler("threshold", cmd_threshold))
        return app
    except TelegramError as e:
        logger.warning(f"Failed to build Telegram application (TelegramError): {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to build Telegram application: {e}")
        return None


def start_telegram_bot():
    """Start Telegram Bot polling in a background thread."""
    import threading
    import asyncio

    settings = get_settings()
    if not (settings.telegram_enabled and settings.telegram_bot_token and settings.telegram_chat_id):
        logger.info("Telegram bot not enabled or missing token/chat_id; skipping polling start.")
        return

    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from utils.image_utils import get_last_processed_frame
            app = build_telegram_application(get_last_processed_frame)
            if app is None:
                logger.warning("Telegram bot application could not be built.")
                return
            logger.info("Starting Telegram bot polling in background thread...")
            app.run_polling(close_loop=True, stop_signals=None)
        except Exception as e:
            logger.error(f"Error in Telegram bot polling thread: {e}")
        finally:
            loop.close()

    t = threading.Thread(target=_thread_target, daemon=True, name="TelegramBotPolling")
    t.start()


