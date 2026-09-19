"""Shared utilities: logging configuration and CLI exit codes."""

from .exit_codes import (
    EXIT_FAILURE,
    EXIT_MISSING_DEPENDENCY,
    EXIT_MODEL_UNAVAILABLE,
    EXIT_OK,
    EXIT_USAGE,
)
from .logging import configure_logging, get_logger

__all__ = [
    "EXIT_FAILURE",
    "EXIT_MISSING_DEPENDENCY",
    "EXIT_MODEL_UNAVAILABLE",
    "EXIT_OK",
    "EXIT_USAGE",
    "configure_logging",
    "get_logger",
]
