"""Structured logging for BioLit. No bare print() in application code."""

from __future__ import annotations

import logging
import sys
from typing import Any

from biolit.core.config import get_settings


def configure_logging(level: str | None = None) -> None:
    """Configure root logger once for the process."""
    settings = get_settings()
    log_level = (level or settings.log_level).upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(log_level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; ensures logging is configured."""
    configure_logging()
    return logging.getLogger(name)


def log_extra(**kwargs: Any) -> dict[str, Any]:
    """Helper for structured extra fields on log records."""
    return kwargs
