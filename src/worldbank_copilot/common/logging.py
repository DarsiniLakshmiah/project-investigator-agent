"""Minimal logging setup shared by modules, scripts and notebooks."""

from __future__ import annotations

import logging
import sys

_ROOT_LOGGER_NAME = "worldbank_copilot"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _CurrentStderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at emit time (not at handler creation)."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def configure_logging(level: str = "INFO") -> None:
    """Configure the package logger once; later calls only update the level."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not logger.handlers:
        handler = _CurrentStderrHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under the package root logger."""
    if name == _ROOT_LOGGER_NAME or name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
