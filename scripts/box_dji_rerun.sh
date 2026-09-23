#!/usr/bin/env bash
# Re-run DJI_0047 from Track A on (Stages 0-2 reused) after the Track A fixes of 2026-09-23:
# the nominal camera-table focal held fixed, one process per run folder. Run it ONCE:
#
#   cd ~/single-pass && nohup bash scripts/box_dji_rerun.sh > box_dji.log 2>&1 &
#   bash scripts/box_status.sh --watch dji47_s3      # Ctrl+C stops watching only
#
# About 25 min. Ends with dji47_after.json (Stages 3-5 report) and dji47_diag_after.json.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
RUN=data/interim/dji47_s3
[ -f "$RUN/manifest.json" ] || { echo "no $RUN: run scripts/box_stage3_runs.sh first"; exit 1; }
$PY -m src.core.runlock "$RUN" || exit 1
echo "== $(date -u '+%H:%M:%S') DJI_0047: track_a, geo, fusion, export -> dji47_rerun.log"
$PY -m src.cli run data/raw/DJI_0047/DJI_0047.mp4 --csv data/raw/DJI_0047/telemetry.csv --resume "$RUN" \
    --stage track_a --stage geo --stage fusion --stage export --force > dji47_rerun.log 2>&1 \
    || echo "   run failed: see dji47_rerun.log"
grep -E "elapsed .* budget" dji47_rerun.log | tail -1
echo "== $(date -u '+%H:%M:%S') report"
$PY scripts/stage4_report.py "$RUN" > dji47_after.json
$PY scripts/box_recon_diag.py "$RUN" > dji47_diag_after.json
echo "== $(date -u '+%H:%M:%S') ALL DONE. Paste the summary command's output back to Claude."
