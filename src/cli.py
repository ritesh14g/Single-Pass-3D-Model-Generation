"""Command-line interface.

    python -m src.cli run data/raw/flight.mp4 --preset fast
    python -m src.cli run data/raw/flight.mp4 --resume data/interim/flight_20260912_101500
    python -m src.cli inspect data/raw/flight.mp4
    python -m src.cli status data/interim/flight_20260912_101500
    python -m src.cli config --preset accurate --set budget.total_s=1800

``inspect`` exists because the most expensive mistake on competition day is
discovering at minute twelve of a fifteen-minute budget that the telemetry was
in a dialect nothing parsed. It answers "what did you actually find in this
input?" in a few seconds, without reconstructing anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from src import __version__
from src.core.config import load_config
from src.core.logging import setup_logging
from src.core.manifest import STAGE_ORDER, RunManifest


def _common_config_options(function):
    """Config selection options shared by every command that loads config."""
    function = click.option(
        "--preset", default=None,
        help="Config preset: fast, accurate, or a path to a YAML overlay.",
    )(function)
    function = click.option(
        "--config", "config_files", multiple=True, type=click.Path(exists=True, dir_okay=False),
        help="Extra YAML overlay(s), merged after the preset.",
    )(function)
    function = click.option(
        "--set", "overrides", multiple=True, metavar="KEY.PATH=VALUE",
        help="Override a single config value. Repeatable.",
    )(function)
    return function


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="ps17")
def cli() -> None:
    """Single-pass drone video to a georeferenced 3D model (SIH26158 / PS-17)."""


@cli.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_common_config_options
@click.option("--srt", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="DJI SRT sidecar. Found automatically next to the video when omitted.")
@click.option("--csv", "csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Flight-log CSV. Found automatically next to the video when omitted.")
@click.option("--gcp", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Optional GCP file. The system is designed to work with zero GCPs.")
@click.option("--out", "run_dir", type=click.Path(file_okay=False, path_type=Path),
              help="Run directory. Defaults to <workdir>/<video stem>_<timestamp>.")
@click.option("--resume", "resume_dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Resume an existing run directory, re-running only what is stale.")
@click.option("--stage", "stages", multiple=True, type=click.Choice(STAGE_ORDER),
              help="Run only these stages. Repeatable.")
@click.option("--force", is_flag=True, help="Re-run stages even when their outputs are still valid.")
@click.option("--mode", type=click.Choice(["A", "B", "hybrid", "auto"]),
              help="Reconstruction mode (spec §7.4). Overrides the config.")
def run(
    video: Path, preset, config_files, overrides, srt, csv_path, gcp,
    run_dir, resume_dir, stages, force, mode,
) -> None:
    """Run the pipeline on VIDEO."""
    from src.pipeline import RunInputs, run_pipeline

    override_list = list(overrides)
    if mode:
        override_list.append(f"run.mode={mode}")
    cfg = load_config(preset=preset, overrides=override_list, extra_files=list(config_files))

    if resume_dir and run_dir:
        raise click.UsageError("--resume and --out are mutually exclusive")

    inputs = RunInputs(video=video, srt=srt, csv=csv_path, gcp=gcp)
    result = run_pipeline(
        inputs=inputs,
        cfg=cfg,
        run_dir=resume_dir or run_dir,
        stages=list(stages) or None,
        force=force,
    )

    click.echo()
    _echo_status(result.manifest)
    summary = result.budget.summary()
    verdict = "within" if summary["within_budget"] else "OVER"
    click.echo(
        f"\n  elapsed {summary['elapsed_s']:.1f}s of {summary['total_s']:.0f}s budget ({verdict})"
    )
    if summary["degradations"]:
        click.echo(f"  {len(summary['degradations'])} degradation(s) applied under time pressure:")
        for item in summary["degradations"]:
            click.echo(f"    - {item['stage']}: {item['action']} ({item['reason']})")
    click.echo(f"  run directory: {result.run_dir}")

    if result.skipped_stages:
        click.echo("\n  stages not run:")
        for name, reason in result.skipped_stages.items():
            click.echo(f"    - {name}: {reason}")


@cli.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_common_config_options
@click.option("--srt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--csv", "csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a readable summary.")
def inspect(video: Path, preset, config_files, overrides, srt, csv_path, as_json) -> None:
    """Report what the pipeline can read from VIDEO, without reconstructing."""
    from src.ingest.telemetry import load_telemetry
    from src.ingest.video_reader import VideoReader

    cfg = load_config(preset=preset, overrides=list(overrides), extra_files=list(config_files))
    setup_logging(None, level="WARNING", jsonl=False)

    with VideoReader(video, hardware_decode=bool(cfg.get_path("ingest.video.hardware_decode"))) as reader:
        metadata = reader.metadata.to_dict()
        keyframes = reader.keyframe_indices()

    telemetry = load_telemetry(video, cfg, srt_path=srt, csv_path=csv_path)
    report = {
        "video": metadata,
        "keyframes": len(keyframes) if keyframes is not None else None,
        "telemetry": telemetry.summary(),
    }

    if as_json:
        click.echo(json.dumps(report, indent=2, default=str))
        return

    click.echo(f"\n  {video.name}")
    click.echo(f"    {metadata['width']}x{metadata['height']} @ {metadata['fps']:.2f} fps, "
               f"{metadata['frame_count']} frames, {metadata['duration_s']:.1f}s, codec {metadata['fourcc']}")
    click.echo(f"    hardware decode: {'yes' if metadata['hardware_decode'] else 'no'}")
    click.echo(f"    keyframes: {report['keyframes'] if report['keyframes'] is not None else 'unknown (PyAV absent)'}")

    telemetry_summary = report["telemetry"]
    click.echo(f"\n  telemetry: {telemetry_summary['source']} ({telemetry_summary['rows']} rows)")
    for label, key in (
        ("GPS", "has_gps"), ("barometric altitude", "has_baro"),
        ("attitude", "has_attitude"), ("focal length", "has_focal"), ("RTK/PPK", "has_rtk"),
    ):
        click.echo(f"    {'yes' if telemetry_summary[key] else ' no'}  {label}")

    if telemetry_summary["scale_free"]:
        click.secho(
            "\n  WARNING: no usable GPS. The reconstruction will be scale-free — "
            "shape without real-world size, and not georeferenced.",
            fg="yellow",
        )
    for note in telemetry_summary.get("notes", []):
        click.echo(f"    note: {note}")

    duration = metadata["duration_s"]
    budget = float(cfg.get_path("budget.total_s"))
    if duration > 0:
        # The spec's target is 15 minutes for a 10-minute video; state the
        # implied ratio for this input rather than a generic claim.
        click.echo(f"\n  budget: {budget:.0f}s for {duration:.0f}s of video "
                   f"({budget / duration:.2f}x realtime)")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True)
def status(run_dir: Path, as_json: bool) -> None:
    """Show the state of a run directory."""
    manifest = RunManifest.load(run_dir)
    if as_json:
        click.echo(json.dumps(manifest.to_dict(), indent=2, default=str))
        return
    click.echo(f"\n  run {manifest.run_id}")
    _echo_status(manifest)


@cli.command(name="config")
@_common_config_options
@click.option("--key", help="Print only this dotted key path.")
def show_config(preset, config_files, overrides, key) -> None:
    """Print the effective configuration after presets and overrides."""
    cfg = load_config(preset=preset, overrides=list(overrides), extra_files=list(config_files))
    if key:
        click.echo(json.dumps(cfg.get_path(key), indent=2, default=str))
        return
    click.echo(json.dumps(cfg.to_dict(), indent=2, default=str))


def _echo_status(manifest: RunManifest) -> None:
    symbols = {"done": "+", "failed": "x", "skipped": "-", "pending": ".", "running": ">"}
    for row in manifest.status_table():
        record = manifest.stages[row["stage"]]
        duration = f"{row['duration_s']:.1f}s" if row["duration_s"] else "-"
        line = f"  {symbols.get(row['status'], '?')} {row['stage']:<12} {row['status']:<8} {duration:>8}"
        if row["warnings"]:
            line += f"   {row['warnings']} warning(s)"
        click.echo(line)
        if record.status.value == "failed" and record.error:
            click.secho(f"      {record.error}", fg="red")


def main() -> int:
    try:
        cli(standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("aborted", err=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
