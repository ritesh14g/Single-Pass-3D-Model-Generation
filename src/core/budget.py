"""Time-budget enforcement (spec §9).

Processing Speed is 20% of the score and the 15-minute ceiling is a hard
requirement, so the budget is a runtime object every stage consults — not a
number in a slide.

The contract each stage honours:

    with budget.stage("condition") as sb:
        for i, frame in enumerate(frames):
            if sb.should_degrade(progress=i / n):
                level = sb.degrade("reduce_resolution", "projected overrun")
                ...

``should_degrade`` projects the stage's finish time from observed progress, so
a stage reacts *before* it blows the deadline rather than after. Degradation
follows the configured ladder in order; when the ladder is exhausted and the
deadline still passes, the stage raises :class:`BudgetExceeded` and the caller
decides whether that is fatal (it usually is not — most stages can emit a
partial, honestly-flagged result).
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.core.logging import get_logger, log_degradation, log_event

log = get_logger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when a stage overruns and the degradation ladder is exhausted."""


@dataclass
class Degradation:
    """One recorded quality-for-time concession."""

    stage: str
    action: str
    reason: str
    at_s: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "at_s": round(self.at_s, 3),
            **self.details,
        }


class StageBudget:
    """Deadline tracker for a single stage."""

    def __init__(
        self,
        name: str,
        allotted_s: float,
        parent: "Budget",
        ladder: list[str],
        warn_at_fraction: float,
        enabled: bool = True,
    ):
        self.name = name
        self.allotted_s = float(allotted_s)
        self.parent = parent
        self.ladder = list(ladder)
        self.warn_at_fraction = float(warn_at_fraction)
        self.enabled = enabled
        self.started_at = time.monotonic()
        self._ladder_index = 0
        self._warned = False
        self.degradations: list[Degradation] = []

    # -- Clock --------------------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_s(self) -> float:
        """Time left for this stage, also capped by the whole-run deadline."""
        stage_left = self.allotted_s - self.elapsed_s
        return min(stage_left, self.parent.remaining_s)

    @property
    def fraction_used(self) -> float:
        if self.allotted_s <= 0:
            return 0.0
        return self.elapsed_s / self.allotted_s

    def exceeded(self) -> bool:
        return self.enabled and self.remaining_s <= 0

    # -- Projection ---------------------------------------------------------
    def projected_total_s(self, progress: float) -> float:
        """Estimate the stage's total runtime from fractional progress."""
        progress = min(max(progress, 1e-6), 1.0)
        return self.elapsed_s / progress

    def should_degrade(self, progress: float | None = None) -> bool:
        """True when this stage is on track to overrun.

        With ``progress`` (0-1) the decision is a projection; without it, it is
        the simpler "we are past the warn fraction" test.
        """
        if not self.enabled or self.allotted_s <= 0:
            return False
        if progress is None:
            over = self.fraction_used >= self.warn_at_fraction
        else:
            over = self.projected_total_s(progress) > self.allotted_s
        if over and not self._warned:
            self._warned = True
            log_event(
                log,
                logging.WARNING,
                f"stage {self.name} projected to overrun",
                event="budget_warn",
                stage=self.name,
                elapsed_s=round(self.elapsed_s, 2),
                allotted_s=self.allotted_s,
                projected_s=round(self.projected_total_s(progress), 2) if progress else None,
            )
        return over

    # -- Ladder -------------------------------------------------------------
    def next_action(self) -> str | None:
        """Peek at the next concession without consuming it."""
        if self._ladder_index >= len(self.ladder):
            return None
        return self.ladder[self._ladder_index]

    def degrade(self, action: str | None = None, reason: str = "time budget", **details: Any) -> str | None:
        """Consume one rung of the degradation ladder and record it.

        Returns the action taken, or ``None`` when the ladder is exhausted —
        at which point the caller should either finish with what it has or
        raise :class:`BudgetExceeded`.
        """
        if action is None:
            action = self.next_action()
            if action is None:
                return None
            self._ladder_index += 1
        else:
            # An explicitly named action still advances the ladder past it, so
            # a stage cannot apply the same concession twice in a loop.
            if action in self.ladder[self._ladder_index :]:
                self._ladder_index = self.ladder.index(action, self._ladder_index) + 1
        record = Degradation(self.name, action, reason, self.elapsed_s, details)
        self.degradations.append(record)
        self.parent.degradations.append(record)
        log_degradation(log, self.name, action, reason, **details)
        return action

    def check(self) -> None:
        """Raise if the deadline has passed and nothing is left to give up."""
        if self.exceeded() and self.next_action() is None:
            raise BudgetExceeded(
                f"stage {self.name} exceeded its {self.allotted_s:.0f}s budget "
                f"(elapsed {self.elapsed_s:.0f}s) with no degradations left"
            )


class Budget:
    """Whole-run budget owning per-stage sub-budgets."""

    def __init__(
        self,
        total_s: float = 900.0,
        stages: dict[str, float] | None = None,
        ladder: list[str] | None = None,
        warn_at_fraction: float = 0.8,
        enabled: bool = True,
        degradation_enabled: bool = True,
    ):
        self.total_s = float(total_s)
        self.stage_allotments = dict(stages or {})
        self.ladder = list(ladder or [])
        self.warn_at_fraction = warn_at_fraction
        self.enabled = enabled
        self.degradation_enabled = degradation_enabled
        self.started_at = time.monotonic()
        self.degradations: list[Degradation] = []
        self.stage_times: dict[str, float] = {}
        self._stages: dict[str, StageBudget] = {}

    @classmethod
    def from_config(cls, cfg: Any) -> "Budget":
        """Build from the ``budget`` section of a :class:`~src.core.config.Config`."""
        section = cfg.get_path("budget", {}) if hasattr(cfg, "get_path") else cfg
        get = section.get if hasattr(section, "get") else dict(section).get
        degradation = get("degradation", {}) or {}
        dget = degradation.get if hasattr(degradation, "get") else dict(degradation).get
        return cls(
            total_s=get("total_s", 900.0),
            stages=dict(get("stages", {}) or {}),
            ladder=list(dget("ladder", []) or []),
            warn_at_fraction=dget("warn_at_fraction", 0.8),
            enabled=bool(get("enabled", True)),
            degradation_enabled=bool(dget("enabled", True)),
        )

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_s(self) -> float:
        if not self.enabled:
            return float("inf")
        return self.total_s - self.elapsed_s

    def allotment_for(self, name: str) -> float:
        """Stage allotment, rescaled when earlier stages have run long.

        If conditioning ate into the MVS allowance, MVS learns that here rather
        than discovering it at the deadline: the remaining stages share what is
        actually left, in proportion to their configured weights.
        """
        nominal = float(self.stage_allotments.get(name, self.total_s))
        if not self.enabled:
            return nominal
        pending = [s for s in self.stage_allotments if s not in self.stage_times and s != name]
        pending_total = sum(float(self.stage_allotments[s]) for s in pending) + nominal
        if pending_total <= 0:
            return nominal
        share = nominal / pending_total
        available = max(self.remaining_s, 0.0)
        return min(nominal, available * share) if available < pending_total else nominal

    @contextmanager
    def stage(self, name: str) -> Iterator[StageBudget]:
        """Scope a stage, timing it and recording the result."""
        sb = StageBudget(
            name=name,
            allotted_s=self.allotment_for(name),
            parent=self,
            ladder=self.ladder if self.degradation_enabled else [],
            warn_at_fraction=self.warn_at_fraction,
            enabled=self.enabled,
        )
        self._stages[name] = sb
        log_event(log, logging.INFO, f"stage {name} start", event="stage_start", stage=name,
                  allotted_s=round(sb.allotted_s, 1), remaining_s=round(self.remaining_s, 1))
        try:
            yield sb
        finally:
            self.stage_times[name] = sb.elapsed_s
            log_event(
                log,
                logging.INFO,
                f"stage {name} done",
                event="stage_end",
                stage=name,
                duration_s=round(sb.elapsed_s, 2),
                allotted_s=round(sb.allotted_s, 1),
                over_budget=sb.elapsed_s > sb.allotted_s,
                degradations=len(sb.degradations),
                remaining_s=round(self.remaining_s, 1),
            )

    def summary(self) -> dict[str, Any]:
        """Timing table for the QA report."""
        return {
            "total_s": self.total_s,
            "elapsed_s": round(self.elapsed_s, 2),
            "within_budget": self.elapsed_s <= self.total_s,
            "stages": {k: round(v, 2) for k, v in self.stage_times.items()},
            "allotments": dict(self.stage_allotments),
            "degradations": [d.to_dict() for d in self.degradations],
        }
