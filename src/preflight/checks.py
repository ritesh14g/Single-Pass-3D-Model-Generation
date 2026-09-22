"""Check records for the input check: what was tested, the verdict, and how to fix it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PASS, WARN, BLOCK, INFO = "pass", "warn", "block", "info"
GROUPS = ("Format", "Video", "Telemetry", "Sync", "Authenticity", "Feasibility")


@dataclass
class Check:
    id: str
    group: str
    label: str
    status: str
    value: Any
    detail: str
    fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verdict(checks: list[Check]) -> str:
    if any(c.status == BLOCK for c in checks):
        return "BLOCKED"
    if any(c.status == WARN for c in checks):
        return "READY_WITH_WARNINGS"
    return "READY"


def band(value: float | None, good: float, ok: float, higher_is_better: bool = True) -> str:
    """PASS at or beyond ``good``, WARN at or beyond ``ok``, else BLOCK."""
    if value is None:
        return INFO
    if higher_is_better:
        return PASS if value >= good else WARN if value >= ok else BLOCK
    return PASS if value <= good else WARN if value <= ok else BLOCK


def soft(status: str) -> str:
    """The same band, but never blocking (for checks that are advice, not requirements)."""
    return WARN if status == BLOCK else status
