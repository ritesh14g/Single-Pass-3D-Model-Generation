"""Structured logging.

Engineering rule 2 (spec §2.4): every stage emits timing to a structured log,
because the 15-minute budget is tracked live rather than measured afterwards.

Two sinks, one call site:
  * console — human readable, for the operator watching a demo run;
  * ``run.jsonl`` — one JSON object per record, for the QA report and for the
    timing tables in §8.5. Structured fields passed as ``extra={"fields": {...}}``
    land in the JSONL as first-class keys, not interpolated into the message.

Degradations and graceful-downgrade events get dedicated helpers. Spec §2.4
rule 4 asks for them to be logged *loudly*: they carry ``event="degradation"``
so the report can list every concession the pipeline made under time pressure.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

LOGGER_ROOT = "ps17"

_COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;196m",
    "CRITICAL": "\033[48;5;196;38;5;231m",
}
_RESET = "\033[0m"

# Records carrying these events are worth making visually distinct on console.
_EVENT_MARKERS = {
    "degradation": "!",
    "downgrade": "!",
    "gate": "#",
    "stage_start": ">",
    "stage_end": "<",
}


class JsonlFormatter(logging.Formatter):
    """One JSON object per line, with structured fields hoisted to top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if key not in payload:
                    payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_json_default)


class ConsoleFormatter(logging.Formatter):
    """Compact, optionally coloured console output with a monotonic clock."""

    def __init__(self, color: bool = True, start_time: float | None = None):
        super().__init__()
        self.color = color and sys.stderr.isatty()
        self.start_time = start_time if start_time is not None else time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed = record.created - self.start_time
        stamp = f"{int(elapsed // 60):02d}:{elapsed % 60:05.2f}"
        name = record.name[len(LOGGER_ROOT) + 1 :] if record.name.startswith(LOGGER_ROOT + ".") else record.name
        fields = getattr(record, "fields", None) or {}
        marker = _EVENT_MARKERS.get(fields.get("event", ""), " ")
        suffix = ""
        # Show the handful of fields an operator actually reads mid-run; the
        # full record is always in the JSONL.
        shown = {k: v for k, v in fields.items() if k in ("stage", "duration_s", "remaining_s", "action", "reason")}
        if shown:
            suffix = "  " + " ".join(f"{k}={_short(v)}" for k, v in shown.items())
        line = f"{stamp} {marker} {record.levelname[0]} {name:<22} {record.getMessage()}{suffix}"
        if self.color:
            line = f"{_COLORS.get(record.levelname, '')}{line}{_RESET}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _json_default(value: Any) -> str:
    return str(value)


def setup_logging(
    run_dir: Path | str | None = None,
    level: str = "INFO",
    jsonl: bool = True,
    console_color: bool = True,
    start_time: float | None = None,
) -> logging.Logger:
    """Configure the ``ps17`` logger tree. Idempotent — safe to call twice."""
    logger = logging.getLogger(LOGGER_ROOT)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(ConsoleFormatter(color=console_color, start_time=start_time))
    logger.addHandler(console)

    if jsonl and run_dir is not None:
        path = Path(run_dir) / "run.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Append: a resumed run continues the same timing record.
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(JsonlFormatter())
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger. ``name`` is usually the module's ``__name__``."""
    if name.startswith("src."):
        name = name[len("src.") :]
    return logging.getLogger(f"{LOGGER_ROOT}.{name}")


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Log with structured fields attached."""
    logger.log(level, msg, extra={"fields": fields})


def log_degradation(logger: logging.Logger, stage: str, action: str, reason: str, **fields: Any) -> None:
    """Record a quality-for-time concession. Always WARNING — never silent."""
    log_event(
        logger,
        logging.WARNING,
        f"degraded {stage}: {action}",
        event="degradation",
        stage=stage,
        action=action,
        reason=reason,
        **fields,
    )


def log_downgrade(logger: logging.Logger, component: str, fallback: str, reason: str, **fields: Any) -> None:
    """Record a graceful-degradation fallback (missing sensor, absent model)."""
    log_event(
        logger,
        logging.WARNING,
        f"{component} unavailable -> falling back to {fallback}",
        event="downgrade",
        component=component,
        fallback=fallback,
        reason=reason,
        **fields,
    )
