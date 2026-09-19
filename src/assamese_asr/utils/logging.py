"""Standard-library logging setup shared by the CLI scripts.

Library code only ever calls :func:`get_logger`; configuring handlers is the
job of the entry points (``scripts/``).
"""

from __future__ import annotations

import logging
import os
from typing import Final, TextIO

ENV_LOG_LEVEL: Final = "ASSAMESE_ASR_LOG_LEVEL"
DEFAULT_LOG_LEVEL: Final = "INFO"

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT: Final = "%Y-%m-%dT%H:%M:%S%z"


def _resolve_level(level: str | int | None) -> int:
    if level is None:
        level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"unknown log level: {level!r}")
    return resolved


def configure_logging(level: str | int | None = None, *, stream: TextIO | None = None) -> None:
    """Configure process-wide logging for a CLI entry point.

    ``level`` defaults to ``$ASSAMESE_ASR_LOG_LEVEL`` and then to ``INFO``.
    Existing handlers are replaced so repeated calls in tests stay predictable.
    """
    resolved = _resolve_level(level)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logging.basicConfig(level=resolved, handlers=[handler], force=True)


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name``."""
    return logging.getLogger(name)
