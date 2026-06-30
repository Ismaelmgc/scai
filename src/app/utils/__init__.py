"""Utility helpers: logging, seeds, IO."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import cast

import numpy as np
import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog for the whole application."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, level.upper(), structlog.INFO)
            if hasattr(structlog, level.upper())
            else 20
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across numpy/random/torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: Path | str) -> Path:
    """Create directory if it does not exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def notify_telegram(text: str) -> None:
    """Best-effort Telegram message (run summaries, fill alerts, failures).

    No-op unless TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set; never raises — a
    notification problem must not affect (or fail) the caller. Uses urllib
    (stdlib), NOT requests, which isn't importable on the GitHub runner.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        return
    try:
        import urllib.request
        payload = json.dumps({
            "chat_id": chat, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
    except Exception as e:
        get_logger(__name__).warning("telegram_notify_failed", error=str(e))
