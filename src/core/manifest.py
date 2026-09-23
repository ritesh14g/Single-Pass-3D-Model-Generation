"""Run manifest — the resumability backbone (spec §2.4 rule 1).

Every stage writes its outputs to disk and registers them here, so a crash in
export does not cost a re-run of ingest. The manifest is a single JSON file at
``<run_dir>/manifest.json`` holding, per stage: status, timing, the artifacts it
produced, the metrics it measured, warnings it raised, and a fingerprint of the
config subtree it depended on.

That fingerprint is what makes resume trustworthy rather than merely fast. A
completed stage is reused only when

  1. it is marked DONE, and
  2. every artifact it registered still exists on disk, and
  3. the config it was computed under has not changed.

Otherwise the stage is stale, and everything downstream of it is invalidated
too — silently reusing a frame set computed under a different blur threshold is
exactly the kind of error that costs a day of debugging on day 14.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from src.core.logging import get_logger, log_event

log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1

# Ordered pipeline stages. Order matters: invalidating a stage invalidates
# everything after it.
STAGE_ORDER = [
    "preflight",
    "ingest",
    "condition",
    "track_b",
    "refine_ba",
    "track_a",
    # Stage 3 works in metres, so it runs after georeferencing and before export.
    "geo",
    "fusion",
    "export",
    "qa",
]

# Config subtrees each stage depends on. A change anywhere in these paths makes
# a previously completed stage stale.
STAGE_CONFIG_DEPS: dict[str, list[str]] = {
    "preflight": ["preflight", "ingest.telemetry"],
    # The input check's measured telemetry offset feeds ingest, so ingest is stale when it changes.
    "ingest": ["ingest", "preflight"],
    "condition": ["condition"],
    "track_b": ["recon.track_b", "device"],
    "refine_ba": ["recon.refine_ba"],
    # The hybrid runs inside track_a, so the Track B model and the run mode change its output too.
    "track_a": ["recon.track_a", "recon.track_b", "run.mode", "device"],
    # The monocular fill uses the Track B predictor; confidence uses export.confidence_full_views.
    "fusion": ["fusion", "recon.track_b", "export.confidence_full_views"],
    "geo": ["geo"],
    "export": ["export", "geo"],
    "qa": ["qa"],
}


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageRecord:
    """Persisted state for one pipeline stage."""

    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: float | None = None
    ended_at: float | None = None
    duration_s: float | None = None
    config_fingerprint: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    degradations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "config_fingerprint": self.config_fingerprint,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "degradations": self.degradations,
        }
        if self.error:
            data["error"] = self.error
        if self.skip_reason:
            data["skip_reason"] = self.skip_reason
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageRecord":
        return cls(
            name=data["name"],
            status=StageStatus(data.get("status", "pending")),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            duration_s=data.get("duration_s"),
            config_fingerprint=data.get("config_fingerprint"),
            artifacts=dict(data.get("artifacts", {})),
            metrics=dict(data.get("metrics", {})),
            warnings=list(data.get("warnings", [])),
            degradations=list(data.get("degradations", [])),
            error=data.get("error"),
            skip_reason=data.get("skip_reason"),
        )


class StageHandle:
    """Handed to stage bodies so they can register outputs as they go."""

    def __init__(self, manifest: "RunManifest", record: StageRecord):
        self._manifest = manifest
        self._record = record

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def dir(self) -> Path:
        """Per-stage working directory, created on demand."""
        path = self._manifest.run_dir / self._record.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add_artifact(self, key: str, path: os.PathLike[str] | str) -> Path:
        """Register an output file or directory produced by this stage."""
        resolved = Path(path)
        self._record.artifacts[key] = self._manifest.relativize(resolved)
        self._manifest.save()
        return resolved

    def add_metric(self, key: str, value: Any) -> None:
        self._record.metrics[key] = value

    def add_metrics(self, values: dict[str, Any]) -> None:
        self._record.metrics.update(values)

    def warn(self, message: str, **fields: Any) -> None:
        """Record a warning that must reach the QA report, not just the log."""
        self._record.warnings.append(message)
        log_event(log, logging.WARNING, message, stage=self._record.name, **fields)

    def record_degradations(self, degradations: list[Any]) -> None:
        for item in degradations:
            self._record.degradations.append(item.to_dict() if hasattr(item, "to_dict") else dict(item))


class RunManifest:
    """The per-run state file, plus resume logic."""

    def __init__(self, run_dir: Path | str, config: dict[str, Any] | None = None, run_id: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.config: dict[str, Any] = config or {}
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.stages: dict[str, StageRecord] = {name: StageRecord(name) for name in STAGE_ORDER}
        self.inputs: dict[str, Any] = {}
        self.environment: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}

    # -- Persistence --------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.run_dir / MANIFEST_NAME

    def relativize(self, path: Path) -> str:
        """Store artifact paths relative to the run dir so runs stay portable."""
        try:
            return str(Path(path).resolve().relative_to(self.run_dir.resolve()).as_posix())
        except ValueError:
            return str(Path(path).resolve())

    def resolve(self, stored: str) -> Path:
        candidate = Path(stored)
        return candidate if candidate.is_absolute() else self.run_dir / candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "inputs": self.inputs,
            "environment": self.environment,
            "config": self.config,
            "stages": {name: rec.to_dict() for name, rec in self.stages.items()},
            "summary": self.summary,
        }

    def save(self) -> None:
        """Atomic write — a crash mid-save must not corrupt the manifest."""
        self.updated_at = time.time()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, default=str)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.run_dir, prefix=".manifest-", suffix=".tmp", delete=False
        )
        try:
            with handle:
                handle.write(payload)
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, run_dir: Path | str) -> "RunManifest":
        run_dir = Path(run_dir)
        with (run_dir / MANIFEST_NAME).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        manifest = cls(run_dir, config=data.get("config", {}), run_id=data.get("run_id"))
        manifest.created_at = data.get("created_at", manifest.created_at)
        manifest.updated_at = data.get("updated_at", manifest.updated_at)
        manifest.inputs = data.get("inputs", {})
        manifest.environment = data.get("environment", {})
        manifest.summary = data.get("summary", {})
        for name, record in data.get("stages", {}).items():
            manifest.stages[name] = StageRecord.from_dict(record)
        for name in STAGE_ORDER:
            manifest.stages.setdefault(name, StageRecord(name))
        return manifest

    @classmethod
    def open(cls, run_dir: Path | str, config: dict[str, Any], resume: bool = True) -> "RunManifest":
        """Load an existing manifest when resuming, else start a fresh one."""
        run_dir = Path(run_dir)
        if resume and (run_dir / MANIFEST_NAME).is_file():
            manifest = cls.load(run_dir)
            manifest.config = config
            return manifest
        return cls(run_dir, config=config)

    # -- Resume logic -------------------------------------------------------
    def fingerprint(self, stage: str) -> str:
        """Stable hash of the config subtrees this stage depends on."""
        deps = STAGE_CONFIG_DEPS.get(stage, [stage])
        material: dict[str, Any] = {}
        for path in deps:
            node: Any = self.config
            for part in path.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    node = None
                    break
            material[path] = node
        blob = json.dumps(material, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def artifacts_present(self, stage: str) -> bool:
        record = self.stages[stage]
        for key, stored in record.artifacts.items():
            if not self.resolve(stored).exists():
                log_event(
                    log,
                    logging.INFO,
                    f"artifact {key} for stage {stage} is missing; stage will re-run",
                    stage=stage,
                    artifact=key,
                    path=stored,
                )
                return False
        return True

    def is_reusable(self, stage: str) -> bool:
        """True when a completed stage's outputs can be trusted as-is."""
        record = self.stages[stage]
        if record.status is not StageStatus.DONE:
            return False
        if record.config_fingerprint != self.fingerprint(stage):
            log_event(
                log,
                logging.INFO,
                f"config for stage {stage} changed since last run; stage will re-run",
                stage=stage,
            )
            return False
        return self.artifacts_present(stage)

    def invalidate_from(self, stage: str) -> None:
        """Reset ``stage`` and everything downstream of it to PENDING."""
        if stage not in STAGE_ORDER:
            return
        for name in STAGE_ORDER[STAGE_ORDER.index(stage) :]:
            record = self.stages[name]
            if record.status is not StageStatus.PENDING:
                self.stages[name] = StageRecord(name)

    def should_run(self, stage: str, force: bool = False) -> bool:
        """Decide whether ``stage`` needs executing; invalidates downstream if so."""
        if force or not self.is_reusable(stage):
            self.invalidate_from(stage)
            self.save()
            return True
        return False

    def artifact(self, stage: str, key: str) -> Path:
        """Fetch a registered artifact path, raising a useful error if absent."""
        record = self.stages.get(stage)
        if record is None or key not in record.artifacts:
            raise KeyError(f"stage {stage!r} registered no artifact {key!r}")
        return self.resolve(record.artifacts[key])

    def has_artifact(self, stage: str, key: str) -> bool:
        record = self.stages.get(stage)
        return bool(record and key in record.artifacts and self.resolve(record.artifacts[key]).exists())

    # -- Execution ----------------------------------------------------------
    @contextmanager
    def stage(self, name: str) -> Iterator[StageHandle]:
        """Run a stage, persisting status and timing around it."""
        if name not in self.stages:
            self.stages[name] = StageRecord(name)
        record = self.stages[name]
        record.status = StageStatus.RUNNING
        record.started_at = time.time()
        record.error = None
        record.config_fingerprint = self.fingerprint(name)
        self.save()
        handle = StageHandle(self, record)
        try:
            yield handle
        except BaseException as exc:
            record.status = StageStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            record.ended_at = time.time()
            record.duration_s = record.ended_at - (record.started_at or record.ended_at)
            self.save()
            raise
        record.status = StageStatus.DONE
        record.ended_at = time.time()
        record.duration_s = record.ended_at - (record.started_at or record.ended_at)
        self.save()

    def mark_skipped(self, name: str, reason: str) -> None:
        """Record a stage that was deliberately not run (mode, or a fallback)."""
        record = self.stages.setdefault(name, StageRecord(name))
        record.status = StageStatus.SKIPPED
        # Not a warning: a stage that was never meant to run in this mode has
        # nothing wrong with it, and counting it as a warning would bury the
        # real ones.
        record.skip_reason = reason
        self.save()

    def status_table(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": name,
                "status": self.stages[name].status.value,
                "duration_s": round(self.stages[name].duration_s, 2) if self.stages[name].duration_s else None,
                "artifacts": len(self.stages[name].artifacts),
                "warnings": len(self.stages[name].warnings),
            }
            for name in STAGE_ORDER
        ]
