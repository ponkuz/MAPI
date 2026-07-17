from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Small standard-library JSON formatter for command-line research runs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "symbol", "horizon", "output", "sample_count"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_structured_logging(level: str = "INFO") -> None:
    """Configure one JSON stream handler without duplicating handlers."""

    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(getattr(handler, "_mapi_json", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._mapi_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
