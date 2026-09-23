#!/usr/bin/env bash
# Every BUILT stage on one video (default preset: VGGT-Ω hybrid), then one report with the
# Stage 4 and Stage 5 scorecards.
#
#   bash scripts/box_full_run.sh [video] [run-name]
set -uo pipefail
VIDEO=${1:-data/raw/Esri_multiplexer_1.mp4}
NAME=${2:-esri_full}
PY=${PY:-.venv/bin/python}
RUN=data/interim/$NAME

if ! ls tools/blender-*/blender > /dev/null 2>&1; then
  echo "== note: no Blender in tools/ -> FBX will be skipped (see CLOUD_GPU_GUIDE.md)"
fi
echo "== 1/2 Stages 1-5 on $VIDEO -> $RUN"
$PY -m src.core.runlock "$RUN" || exit 1   # never delete a folder a live run is using
rm -rf "$RUN"
$PY -m src.cli run "$VIDEO" --out "$RUN" > "${NAME}.log" 2>&1 || echo "   run failed: see ${NAME}.log"
grep -E "elapsed .* budget" "${NAME}.log" | tail -1
echo "== 2/2 report"
$PY scripts/stage4_report.py "$RUN"
