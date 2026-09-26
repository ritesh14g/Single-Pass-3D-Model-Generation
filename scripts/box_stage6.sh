#!/usr/bin/env bash
# Stage 6 on the box (2026-09-26 code): full re-runs of both clips (S5-1 altitude, S4-11 textures,
# rendered ortho, LAS classes and now the viewer + QA report), then the §8.5 degradation bench
# on Esri at one severity. Run it ONCE:
#
#   cd ~/single-pass && nohup bash scripts/box_stage6.sh > box_stage6.log 2>&1 &
#   bash scripts/box_status.sh --watch esri_s6        # Ctrl+C stops watching only
#
# Runs:  esri_s6 (~6 min), dji47_s6 (~40 min), bench_esri (13 Esri runs, ~70 min).
# BENCH=0 skips the bench; SEVERITY=light|moderate|severe picks its severity (default moderate).
# Bring back: data/interim/{esri_s6,dji47_s6}/{export,qa} and data/interim/bench_esri/degradation.*
# (scripts/box_collect.sh esri_s6 dji47_s6). Accuracy against the USGS lidar runs on the laptop:
#   python -m src.cli qa data/box/runs/esri_s6 --reference data/reference/esri_usgs3dep_2018_utm17n_egm96.laz \
#       --bench data/box/runs/bench_esri
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
BENCH=${BENCH:-1}
SEVERITY=${SEVERITY:-moderate}

run() {  # name, then the cli arguments
    local name=$1; shift
    local dir=data/interim/$name
    $PY -m src.core.runlock "$dir" || return 1
    echo "== $(date -u '+%H:%M:%S') $name -> ${name}_run.log"
    $PY -m src.cli run "$@" --out "$dir" --accept-input > "${name}_run.log" 2>&1 || echo "   $name failed: see ${name}_run.log"
    grep -E "elapsed .* budget" "${name}_run.log" | tail -1
    [ -f "$dir/qa/report.html" ] && echo "   report: $dir/qa/report.html  viewer: $dir/qa/viewer/"
}

run esri_s6 data/raw/Esri_multiplexer_1.mp4
run dji47_s6 data/raw/DJI_0047/DJI_0047.mp4 --csv data/raw/DJI_0047/telemetry.csv

if [ "$BENCH" = "1" ]; then
    echo "== $(date -u '+%H:%M:%S') degradation bench (Esri, $SEVERITY) -> bench_esri.log"
    $PY -m src.cli bench degrade data/raw/Esri_multiplexer_1.mp4 --out data/interim/bench_esri \
        --severity "$SEVERITY" > bench_esri.log 2>&1 || echo "   bench failed: see bench_esri.log"
    [ -f data/interim/bench_esri/degradation.md ] && cat data/interim/bench_esri/degradation.md
fi
echo "== $(date -u '+%H:%M:%S') ALL DONE. Run scripts/box_collect.sh esri_s6 dji47_s6, bring the zip back."
