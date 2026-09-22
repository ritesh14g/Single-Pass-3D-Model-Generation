"""Run the Stage 0 input check over a folder of clips and print one line each.

    python scripts/check_corpus.py "D:/Drone Video Dataset" [--json out.json]

Every video found (by content, whatever the extension) is checked, so a new dataset can be
triaged in one pass before anything is reconstructed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config  # noqa: E402
from src.core.logging import setup_logging
from src.ingest.formats import probe_container
from src.preflight import analyze_input

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".ts", ".mts", ".m2ts", ".mxf", ".flv",
                  ".3gp", ".wmv", ".mpg", ".mpeg", ".mpeg4", ".h264", ".h265", ".hevc", ".lrv"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--json", type=Path, help="write the full reports here")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    setup_logging(None, level="ERROR", jsonl=False)
    videos = sorted(p for p in args.folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    print(f"{len(videos)} candidate video file(s) under {args.folder}\n")
    out = {}
    for path in videos:
        probe = probe_container(path)
        if probe.video is None:
            print(f"{path.name:34s} not a video ({probe.error or 'no video stream'})")
            continue
        report = analyze_input(path, cfg)
        out[str(path)] = report.to_dict()
        problems = [c for c in report.checks if c.status in ("block", "warn")]
        head = f"{path.name:34s} {report.verdict:20s} {report.timing_s.get('total', 0):5.1f}s"
        print(f"{head}  {probe.video.width}x{probe.video.height} {probe.video.codec}, "
              f"{(report.telemetry.get('summary') or {}).get('source', 'no telemetry')}")
        for c in problems:
            print(f"{'':36s}{c.status.upper():5s} {c.label}: {c.detail[:150]}")
        print()
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
