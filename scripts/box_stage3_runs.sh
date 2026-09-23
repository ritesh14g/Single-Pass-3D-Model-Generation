#!/usr/bin/env bash
# The Stage 3 box runs (CLOUD_GPU_GUIDE.md §10.4) in one go, meant to run in the background:
#
#   cd ~/single-pass && nohup bash scripts/box_stage3_runs.sh > box_runs.log 2>&1 &
#   tail -f box_runs.log          # Ctrl+C stops watching only; the runs carry on
#
# 1. Esri, every built stage (hybrid VGGT-Omega; Stage 3 reuses Track B's saved depth)
# 2. DJI_0047 (Stage 0 applies the camera FOV prior and the measured telemetry offset)
# 3. One JSON report of Stages 3-5 for both, also saved to box_runs_report.json
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
echo "== $(date -u '+%H:%M:%S') 1/3 Esri -> data/interim/esri_s3"
bash scripts/box_full_run.sh data/raw/Esri_multiplexer_1.mp4 esri_s3 > /dev/null
grep -E "elapsed .* budget" esri_s3.log | tail -1
echo "== $(date -u '+%H:%M:%S') 2/3 DJI_0047 -> data/interim/dji47_s3"
$PY -m src.core.runlock data/interim/dji47_s3 || exit 1   # never delete a folder a live run is using
rm -rf data/interim/dji47_s3
$PY -m src.cli run data/raw/DJI_0047/DJI_0047.mp4 --csv data/raw/DJI_0047/telemetry.csv \
    --out data/interim/dji47_s3 > dji47_s3.log 2>&1 || echo "   DJI_0047 run failed: see dji47_s3.log"
grep -E "elapsed .* budget" dji47_s3.log | tail -1
echo "== $(date -u '+%H:%M:%S') 3/3 report"
$PY scripts/stage4_report.py data/interim/esri_s3 data/interim/dji47_s3 | tee box_runs_report.json
echo "== $(date -u '+%H:%M:%S') done. Paste box_runs_report.json back to Claude."
