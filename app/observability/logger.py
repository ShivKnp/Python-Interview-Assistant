"""
Structured JSON logger — replaces all print() calls with levelled, machine-
parseable log lines written to stdout and to a rotating JSONL file.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


class JSONLineHandler(logging.Handler):
    """Writes each log record as a single JSON line to a file."""

    def __init__(self, log_path: str):
        super().__init__()
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra"):
            entry.update(record.extra)
        try:
            self._file.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        self._file.close()
        super().close()


class _EnterpriseLogger(logging.Logger):
    """Extended logger with a .data() helper for structured fields."""

    def data(self, level: int, msg: str, **kwargs: Any) -> None:
        """Log message with arbitrary keyword data attached."""
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, "(unknown)", 0, msg, (), None
            )
            record.extra = kwargs  # type: ignore[attr-defined]
            self.handle(record)

    def info_data(self, msg: str, **kwargs: Any) -> None:
        self.data(logging.INFO, msg, **kwargs)

    def error_data(self, msg: str, **kwargs: Any) -> None:
        self.data(logging.ERROR, msg, **kwargs)


def _build_logger() -> _EnterpriseLogger:
    logging.setLoggerClass(_EnterpriseLogger)
    log = logging.getLogger("enterprise_rag")
    log.setLevel(logging.DEBUG)

    if not log.handlers:
        # Console handler (human-readable)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        log.addHandler(console)

        # JSONL file handler
        log_path = os.getenv("LOG_PATH", "./logs/pipeline.jsonl")
        try:
            log.addHandler(JSONLineHandler(log_path))
        except Exception:
            pass  # Don't crash if log dir isn't writable

    return log  # type: ignore[return-value]


logger: _EnterpriseLogger = _build_logger()  # type: ignore[assignment]
