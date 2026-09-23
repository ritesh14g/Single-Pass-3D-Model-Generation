#!/usr/bin/env bash
# Package everything worth keeping from the GPU box into one tarball to download.
#
#   bash scripts/box_collect.sh                 # code + evidence (small, a few MB)
#   bash scripts/box_collect.sh esri_full2      # ... plus that run's six output formats
#   bash scripts/box_collect.sh esri_full2 all  # ... plus every run's exports (large)
#
# Always collected (small):
#   * any work not yet pushed: a patch of uncommitted changes and untracked files
#   * every run's manifest, metrics, scorecards, logs, telemetry and frame tables
#   * each run's sparse reconstruction + georeferencing, so Stage 5 can be re-run without a GPU
#   * what the machine was: GPU, CPU, memory, driver, CUDA, installed packages, tool versions
# Optionally (named run): that run's export/ folder — OBJ/PLY/LAS/GeoTIFF/glb/FBX (~400 MB on Esri).
#
# Nothing is deleted and nothing is uploaded; the tarball lands in the home directory.
set -uo pipefail

RUN_EXPORTS=${1:-}
ALL_EXPORTS=${2:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
STAMP=$(date +%Y%m%d_%H%M)
STAGE=$(mktemp -d)/box_handover_$STAMP
OUT=$HOME/box_handover_$STAMP.tar.gz
mkdir -p "$STAGE"/{code,runs,env}

say() { printf '== %s\n' "$*"; }

# ---------------------------------------------------------------- code
say "code: anything not yet in git"
{
  echo "# branch and recent history"
  git rev-parse --abbrev-ref HEAD 2>/dev/null
  git log --oneline -20 2>/dev/null
  echo
  echo "# working tree status"
  git status --porcelain 2>/dev/null
} > "$STAGE/code/git_state.txt" 2>&1

git diff HEAD > "$STAGE/code/uncommitted.patch" 2>/dev/null
UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null | grep -v '^data/' || true)
if [ -n "$UNTRACKED" ]; then
  # Untracked files the repo does not ignore: scripts or configs written on the box.
  echo "$UNTRACKED" > "$STAGE/code/untracked_list.txt"
  tar czf "$STAGE/code/untracked_files.tar.gz" $UNTRACKED 2>/dev/null
fi
if [ -s "$STAGE/code/uncommitted.patch" ] || [ -n "$UNTRACKED" ]; then
  say "   NOTE: this box has work that is not committed — it is in code/ in the tarball"
else
  say "   clean: everything here is already pushed"
  rm -f "$STAGE/code/uncommitted.patch"
fi

# ---------------------------------------------------------------- environment
say "env: what this machine was"
{
  echo "=== date"; date -u
  echo; echo "=== host"; uname -a
  echo; echo "=== cpu"; lscpu 2>/dev/null | head -25
  echo; echo "=== memory"; free -g 2>/dev/null
  echo; echo "=== disk"; df -h "$REPO" "$HOME" 2>/dev/null
  echo; echo "=== gpu"; nvidia-smi 2>/dev/null || echo "no nvidia-smi"
  echo; echo "=== python"; (.venv/bin/python -V 2>/dev/null || python3 -V)
  echo; echo "=== torch / cuda"
  .venv/bin/python - <<'PY' 2>/dev/null || true
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
  echo; echo "=== tools"
  ls -la tools/ 2>/dev/null
  ls tools/openmvs/bin 2>/dev/null
  ls -d tools/blender-* 2>/dev/null
} > "$STAGE/env/machine.txt" 2>&1
(.venv/bin/python -m pip freeze 2>/dev/null || pip freeze 2>/dev/null) > "$STAGE/env/pip_freeze.txt"

# ---------------------------------------------------------------- runs
say "runs: reports, metrics, logs, sparse models"
for RUNDIR in data/interim/*/; do
  [ -d "$RUNDIR" ] || continue
  NAME=$(basename "$RUNDIR")
  DEST="$STAGE/runs/$NAME"
  mkdir -p "$DEST"
  # Every JSON (manifest, input check, scorecards, georef, export metadata) and small tables.
  find "$RUNDIR" -maxdepth 3 -name '*.json' -size -20M -exec cp --parents {} "$STAGE/runs/" \; 2>/dev/null
  find "$RUNDIR" -maxdepth 3 -name '*.parquet' -size -50M -exec cp --parents {} "$STAGE/runs/" \; 2>/dev/null
  find "$RUNDIR" -maxdepth 3 -name '*.txt' -size -20M -exec cp --parents {} "$STAGE/runs/" \; 2>/dev/null
  find "$RUNDIR" -maxdepth 2 -name '*.csv' -size -20M -exec cp --parents {} "$STAGE/runs/" \; 2>/dev/null
  # The structured per-run log: every downgrade, degradation and timing the QA report cites.
  find "$RUNDIR" -maxdepth 1 -name '*.jsonl' -size -20M -exec cp --parents {} "$STAGE/runs/" \; 2>/dev/null
  # The sparse reconstruction: small, and enough to re-run Stage 5 on a laptop.
  if [ -d "$RUNDIR/track_a/sparse" ]; then
    SIZE=$(du -sm "$RUNDIR/track_a/sparse" 2>/dev/null | cut -f1)
    if [ "${SIZE:-999}" -lt 400 ]; then
      mkdir -p "$DEST/track_a"
      cp -r "$RUNDIR/track_a/sparse" "$DEST/track_a/" 2>/dev/null
    else
      echo "sparse model skipped: ${SIZE} MB" > "$DEST/track_a_sparse_SKIPPED.txt"
    fi
  fi
done
# The find --parents copies land under runs/data/interim/<run>/...; flatten that away.
if [ -d "$STAGE/runs/data/interim" ]; then
  for RUNDIR in "$STAGE/runs/data/interim"/*/; do
    NAME=$(basename "$RUNDIR")
    mkdir -p "$STAGE/runs/$NAME"
    cp -r "$RUNDIR"/. "$STAGE/runs/$NAME/" 2>/dev/null
  done
  rm -rf "$STAGE/runs/data"
fi

say "runs: console logs"
mkdir -p "$STAGE/runs/_logs"
find . -maxdepth 1 -name '*.log' -size -50M -exec cp {} "$STAGE/runs/_logs/" \; 2>/dev/null
find . -maxdepth 1 -name 'nohup.out' -size -50M -exec cp {} "$STAGE/runs/_logs/" \; 2>/dev/null
find data/interim -maxdepth 1 -name '*.log' -size -50M -exec cp {} "$STAGE/runs/_logs/" \; 2>/dev/null

# ---------------------------------------------------------------- exports (optional, large)
copy_exports() {
  local name="$1"
  local src="data/interim/$name/export"
  [ -d "$src" ] || { say "   no export/ in run '$name'"; return; }
  local size; size=$(du -sm "$src" | cut -f1)
  say "   $name/export: ${size} MB"
  mkdir -p "$STAGE/runs/$name"
  cp -r "$src" "$STAGE/runs/$name/"
}
if [ -n "$RUN_EXPORTS" ]; then
  say "exports: the finished 3-D outputs"
  if [ "$ALL_EXPORTS" = "all" ]; then
    for RUNDIR in data/interim/*/; do copy_exports "$(basename "$RUNDIR")"; done
  else
    copy_exports "$RUN_EXPORTS"
  fi
else
  say "exports: skipped (pass a run name to include them, e.g. 'bash scripts/box_collect.sh esri_full2')"
fi

# ---------------------------------------------------------------- pack
cat > "$STAGE/README.txt" <<EOF
Handover from the GPU box, $(date -u '+%Y-%m-%d %H:%M UTC').

code/     git state, a patch of anything uncommitted, and untracked files written on the box.
          Apply with:  git apply code/uncommitted.patch
env/      what the machine was: GPU, CPU, memory, CUDA, installed packages, tool versions.
runs/     one folder per pipeline run: manifest, input check, scorecards, georeferencing,
          export metadata, telemetry/frame tables, sparse reconstruction, console logs.
          With a run named on the command line, that run's export/ (OBJ, PLY, LAS, GeoTIFF,
          glb, FBX) is included too.

Not included, because they are re-downloaded rather than kept: OpenMVS binaries, portable
Blender, the VGGT-Omega weights (Hugging Face cache), and the raw input videos.
EOF

say "packing"
tar czf "$OUT" -C "$(dirname "$STAGE")" "$(basename "$STAGE")"
rm -rf "$(dirname "$STAGE")"

echo
say "DONE: $OUT"
ls -lh "$OUT" | awk '{print "   size: " $5}'
echo "   Download it from the Jupyter file browser (right-click the file in your home folder -> Download)."
echo "   Then on the laptop:  tar xzf box_handover_$STAMP.tar.gz"
