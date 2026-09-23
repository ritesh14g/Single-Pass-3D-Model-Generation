#!/usr/bin/env bash
# Which pipeline stage is running right now, per run, from each run's manifest.json.
#
#   bash scripts/box_status.sh                 # once
#   bash scripts/box_status.sh --watch         # refresh every 20 s (Ctrl+C stops watching only)
#   bash scripts/box_status.sh --watch esri_s3 dji47_s3
#
# Run names default to every folder in data/interim. Read-only: safe while runs are going.
cd "${PROJECT:-$HOME/single-pass}" 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
WATCH=0
[ "${1:-}" = "--watch" ] && { WATCH=1; shift; }
RUNS="$*"
PY=${PY:-.venv/bin/python}; [ -x "$PY" ] || PY=python3

show() {
  "$PY" - $RUNS <<'PY'
import json, sys, time
from pathlib import Path
names = sys.argv[1:] or sorted(p.name for p in Path("data/interim").iterdir() if (p / "manifest.json").is_file())
icon = {"done": "OK ", "running": ">> ", "failed": "XX ", "skipped": " - ", "pending": " . "}
now = time.time()
print(time.strftime("%H:%M:%S"), "-" * 60)
for name in names:
    m = Path("data/interim") / name / "manifest.json"
    if not m.is_file():
        print(f"{name}: not started yet")
        continue
    try:
        stages = json.loads(m.read_text())["stages"]
    except Exception:                                    # being written right now
        print(f"{name}: (manifest being written, try again)")
        continue
    total = sum(s.get("duration_s") or 0 for s in stages.values())
    running = [n for n, s in stages.items() if s.get("status") == "running"]
    state = f"RUNNING {running[0]}" if running else ("FINISHED" if all(
        s.get("status") in ("done", "skipped") for s in stages.values()) else "stopped / waiting")
    print(f"{name}: {state}")
    for n, s in stages.items():
        st = s.get("status", "pending")
        if st == "running" and s.get("started_at"):
            t = f"{now - s['started_at']:6.0f} s so far"
        elif s.get("duration_s") is not None:
            t = f"{s['duration_s']:6.1f} s"
        else:
            t = ""
        extra = f"  ({len(s.get('warnings', []))} warnings)" if s.get("warnings") else ""
        if st == "failed":
            extra += f"  ERROR: {str(s.get('error', ''))[:120]}"
        if st != "skipped":
            print(f"   {icon.get(st, '   ')}{n:10s} {t}{extra}")
    print(f"   total so far {total / 60:.1f} min")
PY
  for f in box_runs.log box_rerun.log box_dji.log; do [ -f "$f" ] && grep -q "ALL DONE" "$f" && echo "ALL DONE ($f): paste the report(s) it names back"; done
  jobs_left=$(ps aux 2>/dev/null | grep -c "[s]rc.cli run")
  echo "pipeline processes running: $jobs_left"
  LOG=$(ls -t *_s3.log *_rerun.log 2>/dev/null | head -1)
  [ -n "$LOG" ] && { echo "last line of $LOG:"; tail -1 "$LOG" | cut -c1-150; }
}

if [ "$WATCH" = 1 ]; then
  while true; do clear; show; sleep 20; done
else
  show
fi
