#!/usr/bin/env bash
# Dense-stereo speed sweep for Stage 4 Track A (DEVLOG S4-8).
#
# Every variant reuses one GPU sparse model, so only the PatchMatch settings differ.
# The baseline (COLMAP's quality defaults: 1920 px, 20 source views, 5 iterations,
# geometric pass) took 1133 s for 45 frames on the 2g.20gb MIG slice; the spec's MVS
# budget is 4 min for a 10-min video.
#
#   bash scripts/dense_sweep.sh data/interim/esri_gpu data/outputs/recon_probe/esri_gpu [variants.txt]
#
# variants.txt (optional) replaces the built-in list: one "name|probe flags" per line,
# '#' comments allowed. SKIP_BASELINE=1 skips re-measuring the baseline cloud. The
# table lists every variant already in <base>_sweep, so runs accumulate.
set -euo pipefail

RUN_DIR=${1:?run dir of src.cli run}
BASE=${2:?probe output that holds the sparse model}
VARIANTS_FILE=${3:-}
SWEEP=${BASE}_sweep
OMVS=${OMVS:-tools/openmvs/bin}
PY=${PY:-.venv/bin/python}

# name | probe flags
VARIANTS=(
  "s1280_v8_geom|--dense-size 1280 --pm-src-images 8"
  "s1280_v8_nogeom|--dense-size 1280 --pm-src-images 8 --pm-no-geom"
  "s960_v8_geom|--dense-size 960 --pm-src-images 8"
  "s1280_v8_i3_w2|--dense-size 1280 --pm-src-images 8 --pm-iterations 3 --pm-window-step 2"
  "s960_v6_i3_nogeom|--dense-size 960 --pm-src-images 6 --pm-iterations 3 --pm-no-geom"
)

if [[ -n "$VARIANTS_FILE" ]]; then
  mapfile -t VARIANTS < <(grep -Ev '^[[:space:]]*(#|$)' "$VARIANTS_FILE")
fi

if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
  echo "== baseline: measure the existing dense cloud (no recompute)"
  "$PY" scripts/recon_probe.py --run-dir "$RUN_DIR" --out "$BASE" --openmvs-bin "$OMVS" \
    --reuse --no-texture > /dev/null 2>&1 || echo "baseline measurement failed (see $BASE/probe.log)"
fi

mkdir -p "$SWEEP"
for entry in "${VARIANTS[@]}"; do
  name=${entry%%|*}
  flags=${entry#*|}
  out="$SWEEP/$name"
  echo "== $name: $flags"
  rm -rf "$out" && mkdir -p "$out"
  cp -r "$BASE/sparse" "$out/"
  # shellcheck disable=SC2086
  "$PY" scripts/recon_probe.py --run-dir "$RUN_DIR" --out "$out" --openmvs-bin "$OMVS" \
    --reuse --redo-dense --no-texture $flags > "$out.stdout" 2>&1 || echo "   FAILED (see $out.stdout)"
done

"$PY" - "$BASE" "$SWEEP" <<'EOF'
import json, sys
from pathlib import Path

base, sweep = Path(sys.argv[1]), Path(sys.argv[2])
rows = [("baseline_1920_v20", base)] + [(d.name, d) for d in sorted(sweep.iterdir()) if d.is_dir()]
baseline_patchmatch_s = 1133.3  # measured 2026-09-21; the baseline measurement above reuses its cloud
print("\n===== PASTE EVERYTHING BELOW BACK TO CLAUDE =====")
print(f"{'variant':22s} {'frames':>6s} {'patchmatch_s':>12s} {'fusion_s':>9s} {'points':>9s} {'footprint_m2':>12s} {'mesh_faces':>10s}  notes")
for name, d in rows:
    f = d / "probe_summary.json"
    if not f.exists():
        print(f"{name:22s} {'missing':>12s}")
        continue
    s = json.loads(f.read_text())
    t, dense, mesh = s["timings_s"], s.get("dense", {}), s.get("mesh", {})
    pm = t.get("dense_patchmatch_gpu", baseline_patchmatch_s if d == base else float("nan"))
    notes = "; ".join(n for n in s.get("notes", []) if "DOWNGRADE" in n)
    params = dense.get("params") if isinstance(dense.get("params"), dict) else {}
    print(f"{name:22s} {str(params.get('frames', '-')):>6s} {pm:12.1f}{t.get('dense_fusion', float('nan')):9.1f} "
          f"{dense.get('points', 0):9d} {dense.get('footprint_m2', float('nan')):12.0f} "
          f"{mesh.get('faces', 0):10d}  {notes}")
EOF
