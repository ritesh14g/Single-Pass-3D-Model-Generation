"""Pipeline orchestration.

Every stage follows the same contract, which is what makes spec §2.4 rule 1
("every stage writes to disk and is independently resumable") hold uniformly
rather than stage by stage:

    if manifest.should_run("name"):
        with budget.stage("name") as sb, manifest.stage("name") as st:
            ... work, writing outputs under st.dir ...
            st.add_artifact(key, path)
            st.add_metrics({...})

``should_run`` decides reuse from the recorded status, the artifacts still
being on disk, and a fingerprint of the config the stage depends on — so a
resumed run never silently reuses a frame set computed under different
thresholds. ``budget.stage`` gives the body a deadline it can degrade against,
and ``manifest.stage`` records timing, artifacts, metrics and failures.

Stage functions take and return plain paths and dataclasses, never the
manifest, so each is independently testable.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src import __version__
from src.condition.artifacts import assess_artifacts, suppress_block_artifacts
from src.condition.blur import profile_video_blur, sharpen
from src.condition.dynamic_mask import DynamicMasker, summarize_masks
from src.condition.gps_filter import filter_telemetry, track_length_m
from src.condition.illumination import ExposureChain, condition_illumination, summarize_illumination
from src.core.budget import Budget
from src.core.config import Config
from src.core.device import chunk_frames_for_memory, device_info
from src.core.logging import get_logger, log_event, setup_logging
from src.core.manifest import RunManifest, StageStatus
from src.stages import STAGES, built_manifest_stages
from src.ingest.frame_selector import FrameSelection, load_selection, select_frames
from src.ingest.telemetry import TelemetryTable, load_telemetry
from src.ingest.video_reader import VideoReader

log = get_logger(__name__)

CONDITIONED_IMAGE_QUALITY = 95


@dataclass
class RunInputs:
    """Everything the pipeline was given, plus what it found alongside."""

    video: Path
    srt: Path | None = None
    csv: Path | None = None
    gcp: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": str(self.video),
            "srt": str(self.srt) if self.srt else None,
            "csv": str(self.csv) if self.csv else None,
            "gcp": str(self.gcp) if self.gcp else None,
        }


@dataclass
class RunResult:
    """Outcome of a pipeline run."""

    run_dir: Path
    manifest: RunManifest
    budget: Budget
    completed_stages: list[str] = field(default_factory=list)
    skipped_stages: dict[str, str] = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        return self.budget.summary()["within_budget"]


# --------------------------------------------------------------------------
# Stage 1 — Ingest
# --------------------------------------------------------------------------
def run_ingest(
    inputs: RunInputs, cfg: Config, out_dir: Path, budget_stage: Any = None
) -> dict[str, Any]:
    """Decode, profile, select frames, and parse telemetry (spec §4).

    Telemetry is written *as parsed*. GPS outlier rejection and smoothing are
    §5.6 — Stage 2 — and run in the conditioning stage, so Stage 1 output is a
    faithful record of the input that Stage 2 improvements can be measured
    against.

    Returns a dict of artifact paths and metrics for the caller to register.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ingest_cfg = cfg.get_path("ingest")
    timing: dict[str, float] = {}

    with VideoReader(
        inputs.video,
        hardware_decode=bool(ingest_cfg["video"]["hardware_decode"]),
        max_width=ingest_cfg["video"]["max_width"],
    ) as reader:
        metadata = reader.metadata

        keyframes = None
        if bool(ingest_cfg["frame_selection"]["prefer_keyframes"]):
            keyframes = reader.keyframe_indices()

        started = time.monotonic()
        profile = profile_video_blur(reader, cfg)
        timing["profile_s"] = time.monotonic() - started
        started = time.monotonic()
        selection = select_frames(reader, cfg, profile, keyframes=keyframes, budget=budget_stage)
        timing["select_s"] = time.monotonic() - started
        # Seconds inside VideoCapture during profiling + selection (a subset of the two above).
        timing["decode_s"] = reader.decode_s

    selection_path = selection.save(out_dir / "frames.parquet")
    evaluations_path = out_dir / "blur_evaluations.parquet"
    selection.evaluations_frame().to_parquet(evaluations_path, index=False)

    # Telemetry comes after selection so EXIF, the lowest-priority source, can
    # be read from the frames that were actually kept.
    started = time.monotonic()
    telemetry = load_telemetry(
        inputs.video, cfg, srt_path=inputs.srt, csv_path=inputs.csv,
        frame_timestamps=selection.timestamps, video_duration_s=metadata.duration_s,
    )
    telemetry_path = telemetry.to_parquet(out_dir / "telemetry.parquet")

    # Telemetry resampled onto the kept frames — this is what every later stage
    # actually consumes, so it is materialised once here.
    per_frame = telemetry.at_times(selection.timestamps)
    per_frame.insert(0, "frame_index", selection.indices)
    per_frame_path = out_dir / "frame_telemetry.parquet"
    per_frame.to_parquet(per_frame_path, index=False)
    timing["telemetry_s"] = time.monotonic() - started

    metrics = {
        "video": metadata.to_dict(),
        "selection": selection.summary(),
        "telemetry": telemetry.summary(),
        "track_length_m": round(track_length_m(telemetry), 1),
        "scale_free": telemetry.scale_free,
        "timing": {k: round(v, 3) for k, v in timing.items()},
    }
    return {
        "artifacts": {
            "frames": selection_path,
            "blur_evaluations": evaluations_path,
            "telemetry": telemetry_path,
            "frame_telemetry": per_frame_path,
        },
        "metrics": metrics,
        "selection": selection,
        "telemetry_table": telemetry,
    }


# --------------------------------------------------------------------------
# Stage 2 — Conditioning
# --------------------------------------------------------------------------
def run_condition(
    inputs: RunInputs,
    cfg: Config,
    out_dir: Path,
    selection: FrameSelection,
    telemetry: TelemetryTable,
    budget_stage: Any = None,
) -> dict[str, Any]:
    """Apply the §5 conditioning layer to the selected frames.

    Frames are read in one forward pass, because the exposure chain is
    inherently sequential — each frame's transform is composed from its
    predecessor's — and because a forward pass is the only cheap way to read a
    long-GOP video.

    Every frame leaves with an image, up to two masks, and a fusion weight. The
    weight is the product of what each sub-module learned: mild blur halves it,
    low light halves it again. Downstream, that weight decides how much this
    frame's geometry is allowed to influence the result.
    """
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    enabled = bool(cfg.get_path("condition.enabled"))
    ingest_cfg = cfg.get_path("ingest")

    # §5.6 GPS conditioning on the raw telemetry Stage 1 parsed.
    filtered, gps_report = filter_telemetry(telemetry, cfg)
    filtered_path = filtered.to_parquet(out_dir / "telemetry_filtered.parquet")
    per_frame_telemetry = filtered.at_times(selection.timestamps)
    per_frame_telemetry.insert(0, "frame_index", selection.indices)
    exposure_chain = ExposureChain(cfg)
    masker = DynamicMasker(cfg)

    records: list[dict[str, Any]] = []
    illumination_results = []
    mask_results = []
    blockiness_scores: list[float] = []
    wanted = selection.indices
    total = max(len(wanted), 1)

    with VideoReader(
        inputs.video,
        hardware_decode=bool(ingest_cfg["video"]["hardware_decode"]),
        max_width=ingest_cfg["video"]["max_width"],
    ) as reader:
        for position, frame in enumerate(reader.read_indices(wanted)):
            selected = selection.frames[position] if position < len(selection.frames) else None
            base_weight = selected.weight if selected else 1.0
            image = frame.image

            degraded = False
            if budget_stage is not None and budget_stage.should_degrade(progress=position / total):
                action = budget_stage.degrade(
                    "reduce_resolution", "conditioning projected to overrun its budget",
                    frames_done=position, frames_total=total,
                )
                degraded = action is not None

            if enabled:
                if selected and selected.blur_verdict == "correct":
                    # Mild blur only. Severe blur never reaches here — it was
                    # rejected at selection, because sharpening destroyed detail
                    # manufactures edges rather than recovering them.
                    image = sharpen(image, cfg)

                artifact_assessment = assess_artifacts(image, cfg)
                blockiness_scores.append(artifact_assessment.blockiness)
                image = suppress_block_artifacts(image, cfg, artifact_assessment)

                exposure = exposure_chain.push(
                    frame.index, image, selected.transform_prev if selected else None
                )
                illumination = condition_illumination(image, cfg, exposure, base_weight)
                image = illumination.image
                illumination_results.append(illumination)

                mask = masker.mask(image) if not degraded else _empty_mask(image)
                mask_results.append(mask)
                weight = illumination.weight
                shadow_fraction = illumination.shadow_fraction
                low_light = illumination.low_light
                blockiness = artifact_assessment.blockiness
                masked_fraction = mask.fraction
            else:
                weight = base_weight
                shadow_fraction, low_light, blockiness, masked_fraction = 0.0, False, 1.0, 0.0
                mask = _empty_mask(image)
                illumination = None

            name = f"frame_{frame.index:06d}"
            image_path = image_dir / f"{name}.jpg"
            cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), CONDITIONED_IMAGE_QUALITY])

            exclusion = np.zeros(image.shape[:2], dtype=bool)
            if mask.mask.any():
                exclusion |= mask.mask
            if illumination is not None and illumination.shadow_mask is not None:
                shadow_path = mask_dir / f"{name}_shadow.png"
                cv2.imwrite(str(shadow_path), (illumination.shadow_mask * 255).astype(np.uint8))
            if exclusion.any():
                cv2.imwrite(str(mask_dir / f"{name}_exclude.png"), (exclusion * 255).astype(np.uint8))

            records.append(
                {
                    "frame_index": frame.index,
                    "timestamp_s": frame.timestamp_s,
                    "image": image_path.name,
                    "weight": float(weight),
                    "blur_verdict": selected.blur_verdict if selected else "unknown",
                    "blur_score": selected.blur_score if selected else float("nan"),
                    "blockiness": float(blockiness),
                    "shadow_fraction": float(shadow_fraction),
                    "low_light": bool(low_light),
                    "dynamic_fraction": float(masked_fraction),
                    "conditioning_degraded": degraded,
                }
            )

    frame_table = pd.DataFrame(records)
    table_path = out_dir / "conditioned.parquet"
    frame_table.to_parquet(table_path, index=False)

    geo_path = _write_geo_txt(out_dir / "geo.txt", frame_table, per_frame_telemetry)

    metrics = {
        "frames_conditioned": len(records),
        "enabled": enabled,
        "exposure": exposure_chain.summary(),
        "illumination": summarize_illumination(illumination_results),
        "dynamic": summarize_masks(mask_results),
        "dynamic_model_available": masker.available,
        "dynamic_unavailable_reason": masker.unavailable_reason,
        "blockiness_mean": round(float(np.mean(blockiness_scores)), 3) if blockiness_scores else None,
        "mean_weight": round(float(frame_table["weight"].mean()), 3) if len(frame_table) else None,
        "gps_filter": gps_report.to_dict(),
    }
    artifacts = {
        "images": image_dir, "masks": mask_dir, "conditioned": table_path,
        "telemetry_filtered": filtered_path,
    }
    if geo_path is not None:
        artifacts["geo"] = geo_path
    return {"artifacts": artifacts, "metrics": metrics}


def _empty_mask(image: np.ndarray):
    from src.condition.dynamic_mask import DynamicMaskResult

    return DynamicMaskResult(mask=np.zeros(image.shape[:2], dtype=bool), method="none")


def _write_geo_txt(path: Path, frames: pd.DataFrame, telemetry: pd.DataFrame) -> Path | None:
    """Write an OpenDroneMap ``geo.txt`` from the per-frame telemetry.

    Format is ``EPSG`` on line one, then ``image_name longitude latitude
    altitude`` per image. Returns ``None`` when there is no GPS — a scale-free
    run must not emit a file that claims georeferencing it does not have.
    """
    if telemetry.empty or not telemetry["lat"].notna().any():
        log_event(log, logging.WARNING,
                  "no GPS available; skipping geo.txt — Track A will run without position priors",
                  event="downgrade")
        return None

    merged = frames.merge(telemetry, on="frame_index", how="left")
    lines = ["EPSG:4326"]
    written = 0
    for row in merged.itertuples():
        if not (np.isfinite(row.lat) and np.isfinite(row.lon)):
            continue
        altitude = row.alt_gps if np.isfinite(row.alt_gps) else 0.0
        lines.append(f"{row.image} {row.lon:.9f} {row.lat:.9f} {altitude:.3f}")
        written += 1

    if written == 0:
        return None
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_event(log, logging.INFO, f"wrote geo.txt for {written} images", images=written)
    return path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def make_run_id(video: Path) -> str:
    return f"{video.stem}_{time.strftime('%Y%m%d_%H%M%S')}"


def environment_info(cfg: Config) -> dict[str, Any]:
    """Captured into the manifest so a result can be traced to what produced it."""
    info: dict[str, Any] = {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    info.update(device_info(cfg.get_path("device.prefer", "auto")))
    return info


def run_pipeline(
    inputs: RunInputs,
    cfg: Config,
    run_dir: Path | None = None,
    stages: list[str] | None = None,
    force: bool = False,
) -> RunResult:
    """Run the pipeline end to end, resuming whatever is already done."""
    video = Path(inputs.video)
    if not video.is_file():
        raise FileNotFoundError(f"input video not found: {video}")

    workdir = Path(cfg.get_path("run.workdir"))
    run_dir = Path(run_dir) if run_dir else workdir / make_run_id(video)
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        run_dir=run_dir,
        level=cfg.get_path("logging.level", "INFO"),
        jsonl=bool(cfg.get_path("logging.jsonl", True)),
        console_color=bool(cfg.get_path("logging.console_color", True)),
    )

    manifest = RunManifest.open(run_dir, cfg.to_dict(), resume=bool(cfg.get_path("run.resume", True)))
    manifest.inputs = inputs.to_dict()
    manifest.environment = environment_info(cfg)
    manifest.save()

    budget = Budget.from_config(cfg)
    result = RunResult(run_dir=run_dir, manifest=manifest, budget=budget)

    log_event(
        log, logging.INFO, f"run {manifest.run_id} starting",
        run_dir=str(run_dir), preset=cfg.get_path("preset", "default"),
        mode=cfg.get_path("run.mode"), budget_s=budget.total_s,
        device=manifest.environment.get("device"),
    )

    # Without an explicit request, run only what the stage registry marks as
    # built, so half-finished stages never contaminate a run.
    wanted = set(stages) if stages else set(built_manifest_stages())

    def should(name: str) -> bool:
        if wanted is not None and name not in wanted:
            return False
        return manifest.should_run(name, force=force)

    # -- Ingest -------------------------------------------------------------
    selection: FrameSelection | None = None
    if should("ingest"):
        with budget.stage("ingest") as sb, manifest.stage("ingest") as st:
            outcome = run_ingest(inputs, cfg, st.dir, budget_stage=sb)
            for key, path in outcome["artifacts"].items():
                st.add_artifact(key, path)
            st.add_metrics(outcome["metrics"])
            st.record_degradations(sb.degradations)
            selection = outcome["selection"]
            if outcome["metrics"]["scale_free"]:
                st.warn(
                    "no usable GPS: the model will be scale-free and cannot be georeferenced "
                    "or measured in metres"
                )
        result.completed_stages.append("ingest")
    elif manifest.stages["ingest"].status is StageStatus.DONE:
        log_event(log, logging.INFO, "reusing completed ingest stage", stage="ingest")

    # -- Conditioning -------------------------------------------------------
    if should("condition"):
        if manifest.stages["ingest"].status is not StageStatus.DONE:
            raise RuntimeError("conditioning needs a completed ingest stage")
        if selection is None:
            selection = load_selection(manifest.artifact("ingest", "frames"))
        raw_telemetry = TelemetryTable.from_parquet(manifest.artifact("ingest", "telemetry"))

        with budget.stage("condition") as sb, manifest.stage("condition") as st:
            outcome = run_condition(inputs, cfg, st.dir, selection, raw_telemetry, budget_stage=sb)
            for key, path in outcome["artifacts"].items():
                st.add_artifact(key, path)
            st.add_metrics(outcome["metrics"])
            st.record_degradations(sb.degradations)
            if not outcome["metrics"]["dynamic_model_available"]:
                st.warn(
                    "semantic dynamic-object masking unavailable "
                    f"({outcome['metrics']['dynamic_unavailable_reason']}); "
                    "relying on geometric consistency alone"
                )
        result.completed_stages.append("condition")

    # -- Reconstruction, fusion, geo, export, QA ----------------------------
    # These stages are wired but not yet implemented; each records why it did
    # not run so the manifest and QA report stay truthful about what produced
    # the outputs rather than silently showing fewer stages.
    implemented = {"ingest", "condition"}
    for spec in STAGES:
        for name in spec.manifest_stages:
            if name in result.completed_stages or manifest.stages[name].status is StageStatus.DONE:
                continue
            if name in wanted and name in implemented:
                continue
            reason = (
                f"Stage {spec.number} ({spec.title}, {spec.spec_ref}) is {spec.status.value}"
                + ("" if name in implemented else "; not implemented yet")
            )
            if not spec.is_built and name in implemented:
                reason += " — not run by default; request it with --stage"
            manifest.mark_skipped(name, reason)
            result.skipped_stages[name] = reason

    manifest.summary = {
        "budget": budget.summary(),
        "stages": manifest.status_table(),
        "completed": result.completed_stages,
        "skipped": result.skipped_stages,
    }
    manifest.save()

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(manifest.summary, indent=2, default=str), encoding="utf-8")

    log_event(
        log, logging.INFO, f"run {manifest.run_id} finished",
        elapsed_s=round(budget.elapsed_s, 1), budget_s=budget.total_s,
        within_budget=result.within_budget, completed=len(result.completed_stages),
        degradations=len(budget.degradations),
    )
    return result
