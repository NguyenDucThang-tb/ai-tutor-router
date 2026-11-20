"""Minimal JSON logs with request correlation; prompts and credentials are omitted."""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    FIELDS = (
        "request_id",
        "router_strategy",
        "model_tier",
        "selected_model",
        "executed_model",
        "provider",
        "attempt_number",
        "latency_ms",
        "status",
        "error_category",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
    )

    def format(self, record: logging.LogRecord) -> str:
        item = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field in self.FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                item[field] = value
        return json.dumps(item, separators=(",", ":"), ensure_ascii=False)


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not any(getattr(handler, "_tutor_json_handler", False) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._tutor_json_handler = True
        root.addHandler(handler)
    root.setLevel(level)
