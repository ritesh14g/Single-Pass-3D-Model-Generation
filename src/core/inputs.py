"""Optional inputs: what the pipeline can use, and what it does without it.

Every stage of this pipeline is built to run on less than ideal input. A clip
with no barometer still reconstructs; one with no GPS reconstructs scale-free;
one without ultralytics masks movers geometrically. Each of those decisions is
already logged as a ``downgrade`` event, but a warning buried in ``run.jsonl``
is invisible at the moment someone is reading a scorecard and wondering why a
number looks wrong.

This module turns those scattered decisions into one declared ledger: for each
optional input, was it present, what ran instead, and what that costs. It is
*derived* from what the run already recorded rather than threaded through the
stage functions as another parameter, so adding an entry here costs nothing at
the call sites and cannot drift out of sync with a stage's real behaviour.

``present`` is deliberately three-valued. ``None`` means "not determined" —
the stage that would have decided never ran — which is a different statement
from "absent", and the UI must not render an unknown as a missing input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Status values for a single optional input.
PRESENT = "present"
ABSENT = "absent"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class OptionalInput:
    """One input the pipeline can exploit but does not require."""

    key: str
    label: str
    spec_ref: str
    stage: str                  # the stage whose metrics decide this
    fallback: str               # what runs when it is absent
    impact: str                 # what is lost or degraded without it
    resolve: Callable[[dict[str, Any]], "tuple[str, str]"]

    def describe(self, metrics: "dict[str, Any] | None") -> dict[str, Any]:
        if metrics is None:
            status, detail = UNKNOWN, f"stage {self.stage!r} has not run"
        else:
            status, detail = self.resolve(metrics)
        return {
            "key": self.key,
            "label": self.label,
            "spec_ref": self.spec_ref,
            "stage": self.stage,
            "status": status,
            "detail": detail,
            # Only meaningful when the input is missing; kept on every row so a
            # reader can see what the fallback *would* be.
            "fallback": self.fallback,
            "impact": self.impact,
        }


# --------------------------------------------------------------------------
# Resolvers
#
# Each reads the metrics dict of its own stage. They return (status, detail)
# and must never raise: a missing key means the run predates the metric, which
# is "unknown", not "absent".
# --------------------------------------------------------------------------
def _telemetry_flag(flag: str, present_detail: str, absent_detail: str):
    def resolve(metrics: dict[str, Any]) -> tuple[str, str]:
        telemetry = metrics.get("telemetry")
        if not isinstance(telemetry, dict) or flag not in telemetry:
            return UNKNOWN, "no telemetry metrics recorded"
        source = telemetry.get("source", "telemetry")
        if telemetry[flag]:
            return PRESENT, f"{present_detail} (from {source})"
        return ABSENT, absent_detail
    return resolve


def _resolve_hardware_decode(metrics: dict[str, Any]) -> tuple[str, str]:
    video = metrics.get("video")
    if not isinstance(video, dict) or "hardware_decode" not in video:
        return UNKNOWN, "no video metrics recorded"
    if video["hardware_decode"]:
        return PRESENT, f"accelerator engaged for {video.get('fourcc', 'this codec')}"
    return ABSENT, "OpenCV did not engage an accelerator for this build"


def _resolve_keyframes(metrics: dict[str, Any]) -> tuple[str, str]:
    selection = metrics.get("selection")
    if not isinstance(selection, dict) or "keyframes_preferred" not in selection:
        return UNKNOWN, "no selection metrics recorded"
    if selection["keyframes_preferred"]:
        return PRESENT, "I-frames preferred during selection"
    return ABSENT, "PyAV is not installed; frame types are unavailable through OpenCV"


def _resolve_semantic_masking(metrics: dict[str, Any]) -> tuple[str, str]:
    if "dynamic_model_available" not in metrics:
        return UNKNOWN, "no conditioning metrics recorded"
    if metrics["dynamic_model_available"]:
        return PRESENT, "semantic detector loaded"
    return ABSENT, str(metrics.get("dynamic_unavailable_reason") or "detector unavailable")


def _resolve_altitude_fusion(metrics: dict[str, Any]) -> tuple[str, str]:
    gps = metrics.get("gps_filter")
    if not isinstance(gps, dict) or "altitude_source" not in gps:
        return UNKNOWN, "no GPS conditioning metrics recorded"
    source = str(gps["altitude_source"])
    if source == "baro+gps_complementary":
        return PRESENT, "barometric and GPS altitude fused"
    return ABSENT, f"altitude came from {source!r} alone"


def _resolve_pose_prior(metrics: dict[str, Any]) -> tuple[str, str]:
    """Frame centre / slant range — a georeferencing prior only KLV carries."""
    telemetry = metrics.get("telemetry")
    if not isinstance(telemetry, dict):
        return UNKNOWN, "no telemetry metrics recorded"
    source = str(telemetry.get("source", ""))
    if source.startswith("klv:"):
        return PRESENT, f"sensor pointing and frame centre from {source}"
    return ABSENT, "only MISB ST 0601 telemetry carries frame centre and slant range"


OPTIONAL_INPUTS: tuple[OptionalInput, ...] = (
    OptionalInput(
        key="gps", label="GPS position", spec_ref="§4.2", stage="ingest",
        fallback="scale-free reconstruction",
        impact="the model has no metric scale and cannot be georeferenced",
        resolve=_telemetry_flag("has_gps", "lat/lon fixes available",
                                "no usable lat/lon in any telemetry source"),
    ),
    OptionalInput(
        key="alt_baro", label="Barometric altitude", spec_ref="§5.6", stage="ingest",
        fallback="GPS altitude only",
        impact="vertical position is noisier; no complementary baro/GPS fusion",
        resolve=_telemetry_flag("has_baro", "barometric altitude available",
                                "no barometric or relative altitude column"),
    ),
    OptionalInput(
        key="altitude_fusion", label="Baro/GPS altitude fusion", spec_ref="§5.6",
        stage="condition",
        fallback="whichever altitude source exists, unfused",
        impact="vertical datum is less stable over the flight",
        resolve=_resolve_altitude_fusion,
    ),
    OptionalInput(
        key="attitude", label="Platform attitude", spec_ref="§4.2", stage="ingest",
        fallback="orientation solved from imagery alone",
        impact="no roll/pitch/yaw prior for bundle adjustment",
        resolve=_telemetry_flag("has_attitude", "roll/pitch/yaw available",
                                "telemetry carries no attitude columns"),
    ),
    OptionalInput(
        key="focal_mm", label="Focal length", spec_ref="§4.2", stage="ingest",
        fallback="self-calibrated intrinsics",
        impact="focal length is solved from the imagery; weaker on low-parallax scenes",
        resolve=_telemetry_flag("has_focal", "focal length reported",
                                "telemetry carries no focal length"),
    ),
    OptionalInput(
        key="rtk", label="RTK/PPK fixes", spec_ref="§5.6", stage="ingest",
        fallback="standard GPS weighting",
        impact="GPS positions are not tightened in the adjustment",
        resolve=_telemetry_flag("has_rtk", "RTK/PPK fixes detected",
                                "no RTK/PPK flag in telemetry"),
    ),
    OptionalInput(
        key="pose_prior", label="Sensor pointing / frame centre", spec_ref="§8.1",
        stage="ingest",
        fallback="georeferencing from platform position alone",
        impact="no direct ground-intersection prior for the footprint",
        resolve=_resolve_pose_prior,
    ),
    OptionalInput(
        key="hardware_decode", label="Hardware video decode", spec_ref="§4.1",
        stage="ingest",
        fallback="software decode",
        impact="decode dominates the §4.3 time budget (see S1-4)",
        resolve=_resolve_hardware_decode,
    ),
    OptionalInput(
        key="keyframe_index", label="Keyframe index (PyAV)", spec_ref="§4.1",
        stage="ingest",
        fallback="uniform frame treatment",
        impact="cannot prefer I-frames, so selected frames may carry more compression damage",
        resolve=_resolve_keyframes,
    ),
    OptionalInput(
        key="semantic_masking", label="Semantic dynamic masking", spec_ref="§5.5",
        stage="condition",
        fallback="geometric consistency only",
        impact="movers are found by geometry alone; stationary vehicles are not masked",
        resolve=_resolve_semantic_masking,
    ),
)


def describe_optional_inputs(manifest: Any) -> list[dict[str, Any]]:
    """The ledger for one run, in registry order.

    ``manifest`` is a ``RunManifest``. Stages that did not complete yield
    ``unknown`` rather than ``absent`` — the pipeline never got far enough to
    find out, and reporting that as a missing input would be a lie.
    """
    from src.core.manifest import StageStatus

    ledger: list[dict[str, Any]] = []
    for spec in OPTIONAL_INPUTS:
        record = manifest.stages.get(spec.stage)
        metrics = None
        if record is not None and record.status is StageStatus.DONE:
            metrics = record.metrics or {}
        ledger.append(spec.describe(metrics))
    return ledger


def missing_inputs(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in ledger if row["status"] == ABSENT]


def summarize(ledger: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(1 for row in ledger if row["status"] == status)
        for status in (PRESENT, ABSENT, UNKNOWN)
    }
