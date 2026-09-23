#!/usr/bin/env bash
# Re-run only Stage 3 (fusion) + Stage 5 (export) on the two existing box runs, after a code update.
# Stages 0-4 are reused from the runs' folders, so this takes minutes, not the 30 of a full run.
#
#   cd ~/single-pass && nohup bash scripts/box_stage3_rerun.sh > box_rerun.log 2>&1 &
#   tail -f box_rerun.log         # Ctrl+C stops watching only; the runs carry on
#   bash scripts/box_status.sh --watch esri_s3 dji47_s3
#
# 1. the "before" report (only once: an existing before_report.json is kept)
# 2. esri_s3 and dji47_s3: --resume ... --stage fusion --stage export --force
# 3. the "after" report -> after_fix_report.json
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
RUNS=()
for r in esri_s3 dji47_s3; do
  [ -f "data/interim/$r/manifest.json" ] && RUNS+=("data/interim/$r") || echo "   (no data/interim/$r: skipped)"
done
[ ${#RUNS[@]} -gt 0 ] || { echo "no runs to re-run"; exit 1; }

echo "== $(date -u '+%H:%M:%S') 1/3 before report -> before_report.json"
if [ -s before_report.json ]; then
  echo "   kept the existing before_report.json"
else
  $PY scripts/stage4_report.py "${RUNS[@]}" > before_report.json || echo "   before report failed"
fi

echo "== $(date -u '+%H:%M:%S') 2/3 Stage 3 + export"
if [ -d data/interim/esri_s3 ]; then
  echo "   esri_s3 -> esri_rerun.log"
  $PY -m src.cli run data/raw/Esri_multiplexer_1.mp4 --resume data/interim/esri_s3 \
      --stage fusion --stage export --force > esri_rerun.log 2>&1 || echo "   esri_s3 failed: see esri_rerun.log"
  grep -E "elapsed .* budget|degradation|fusion" esri_rerun.log | tail -4
fi
if [ -d data/interim/dji47_s3 ]; then
  echo "   dji47_s3 -> dji47_rerun.log"
  $PY -m src.cli run data/raw/DJI_0047/DJI_0047.mp4 --csv data/raw/DJI_0047/telemetry.csv \
      --resume data/interim/dji47_s3 --stage fusion --stage export --force > dji47_rerun.log 2>&1 \
      || echo "   dji47_s3 failed: see dji47_rerun.log"
  grep -E "elapsed .* budget|degradation|fusion" dji47_rerun.log | tail -4
fi

echo "== $(date -u '+%H:%M:%S') 3/3 after report -> after_fix_report.json"
$PY scripts/stage4_report.py "${RUNS[@]}" | tee after_fix_report.json
echo "== $(date -u '+%H:%M:%S') ALL DONE. Paste before_report.json and after_fix_report.json back to Claude."
