#!/usr/bin/env bash
# DJI_0047 regression (2026-09-28): the first-pass SfM folded (camera path chord/length 0.33 vs GPS
# 1.00, 248 m from GPS) and the GPS-prior pass found no initial pair, so the folded model was kept
# (coverage 8.1%, was 87.8% on 2026-09-23). Two controlled re-runs, one after the other:
#
#   A  dji47_expold : preflight -> geo with the exposure chain as it was before S2-9
#                     (per-pixel OLS, no blocks, no leak). Straight now = S2-9 changed the features.
#   B  dji47_rep    : Track A + geo again on dji47_s6's own frames (same code, same inputs).
#                     Straight now = the mapper is not repeatable on this clip.
#
#   cd ~/single-pass && nohup bash scripts/box_dji_diag.sh > dji_diag.log 2>&1 &
#
# ~45 min. Prints one line per run: registered, cameras vs GPS, GPS-prior outcome, SfM straightness.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
VIDEO=data/raw/DJI_0047/DJI_0047.mp4
CSV=data/raw/DJI_0047/telemetry.csv

summary() {  # run folder
    $PY - "$1" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
run = Path(sys.argv[1])
try:
    m = json.loads((run / "manifest.json").read_text())["stages"]["track_a"]["metrics"]
except Exception as exc:
    print(f"{run.name}: no Track A metrics ({exc})"); sys.exit()
ref = m.get("gps_refinement") or {}
line = (f"{run.name}: registered {m.get('registered')}/{m.get('frames_in')}, cameras vs GPS "
        f"{m.get('cam_vs_gps_rms_m')} m, GPS-prior pass: {ref.get('kept')}, sparse_map "
        f"{round((m.get('timings_s') or {}).get('sparse_map', 0))} s")
try:
    import pycolmap
    sparse = run / "track_a" / "sparse"
    model = sparse / "gps_refined" if (sparse / "gps_refined").exists() else sparse / "0"
    rec = pycolmap.Reconstruction(str(model))
    c = np.array([im.projection_center() for im in sorted((i for i in rec.images.values() if i.has_pose),
                                                           key=lambda i: i.name)])
    line += f", SfM path chord/length {np.linalg.norm(c[-1] - c[0]) / np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)):.2f} (GPS 1.00)"
except Exception as exc:
    line += f", straightness n/a ({type(exc).__name__})"
print(line)
PY
}

echo "== $(date -u '+%H:%M:%S') A: exposure chain as before S2-9 -> dji47_expold"
$PY -m src.core.runlock data/interim/dji47_expold || exit 1
rm -rf data/interim/dji47_expold
$PY -m src.cli run "$VIDEO" --csv "$CSV" --out data/interim/dji47_expold --accept-input \
    --stage preflight --stage ingest --stage condition --stage track_a --stage geo \
    --set condition.illumination.exposure_chain.fit_block_px=1 \
    --set condition.illumination.exposure_chain.fit_method=ols \
    --set condition.illumination.exposure_chain.leak=0 > dji47_expold_run.log 2>&1 \
    || echo "   A failed: see dji47_expold_run.log"

echo "== $(date -u '+%H:%M:%S') B: same code, Track A again on dji47_s6's frames -> dji47_rep"
$PY -m src.core.runlock data/interim/dji47_rep || exit 1
rm -rf data/interim/dji47_rep
mkdir -p data/interim/dji47_rep
for d in preflight ingest condition; do
    [ -d data/interim/dji47_s6/$d ] && cp -r data/interim/dji47_s6/$d data/interim/dji47_rep/
done
cp data/interim/dji47_s6/manifest.json data/interim/dji47_s6/optional_inputs.json data/interim/dji47_rep/ 2>/dev/null
$PY -m src.cli run "$VIDEO" --csv "$CSV" --resume data/interim/dji47_rep \
    --stage track_a --stage geo --force > dji47_rep_run.log 2>&1 || echo "   B failed: see dji47_rep_run.log"

echo "== $(date -u '+%H:%M:%S') results"
summary data/interim/dji47_s6
summary data/interim/dji47_expold
summary data/interim/dji47_rep

# Finish the variant whose cameras agree with GPS (the geometry gate: <= 25 m), best first, so the
# prototype has a usable DJI_0047 model: Stage 3 fill, export, viewer + QA, then the handover.
BEST=$($PY - <<'PY'
import json
from pathlib import Path
best = None
for name in ("dji47_expold", "dji47_rep"):
    try:
        m = json.loads(Path(f"data/interim/{name}/manifest.json").read_text())["stages"]["track_a"]["metrics"]
        rms = float(m.get("cam_vs_gps_rms_m"))
    except Exception:
        continue
    if rms <= 25 and (best is None or rms < best[1]):
        best = (name, rms)
print(best[0] if best else "")
PY
)
if [ -z "$BEST" ]; then
    echo "== neither variant agrees with GPS (all > 25 m): nothing finished; paste this log to the laptop session"
    exit 1
fi
echo "== $(date -u '+%H:%M:%S') finishing $BEST: fusion, export, qa"
$PY -m src.cli run "$VIDEO" --csv "$CSV" --resume data/interim/$BEST     --stage fusion --stage export --stage qa > ${BEST}_finish.log 2>&1 || echo "   finish failed: see ${BEST}_finish.log"
grep -E "elapsed .* budget|scores:" ${BEST}_finish.log | tail -2
bash scripts/box_collect.sh "$BEST"
echo "== $(date -u '+%H:%M:%S') DONE: download the newest ~/box_handover_*.tar.gz ($BEST)"
