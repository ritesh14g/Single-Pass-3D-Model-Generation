#!/usr/bin/env bash
# Restore the project on a GPU box from a box_pack.py upload (CLOUD_GPU_GUIDE.md §10).
#
#   bash box_restore.sh ~/box_upload_<stamp>.zip [~/single-pass] [--no-tests] [--unpack-only]
#
# --unpack-only refreshes the code (and clips) from a newer bundle and verifies it, nothing else.
# Every step is safe to re-run: it skips what is already there, so after a culled session
# just run it again. It never writes a token to disk; Hugging Face login stays in your shell.
#   1 unpack    code + clips into the project folder (keeps .venv, tools/, data/interim/)
#   2 verify    every file against BUNDLE.json's sha256
#   3 machine   GPU / MIG slice, CPU quota, memory, disk
#   4 python    .venv (--system-site-packages), requirements, pycolmap-cuda12, GPU extras
#   5 tools     portable Blender 4.2.3, OpenMVS 2.4.0 (Linux), VGGT + VGGT-Omega code
#   6 weights   VGGT-Omega checkpoint (gated: needs HF_TOKEN in this shell)
#   7 check     torch sees CUDA, device budget, NVDEC probe, test suite
set -uo pipefail

ZIP=${1:?usage: bash box_restore.sh <box_upload.zip> [project-dir] [--no-tests] [--unpack-only]}
DEST=${2:-$HOME/single-pass}
case "$DEST" in --*) DEST=$HOME/single-pass ;; esac
RUN_TESTS=1
UNPACK_ONLY=0
for a in "$@"; do
  [ "$a" = "--no-tests" ] && RUN_TESTS=0
  [ "$a" = "--unpack-only" ] && UNPACK_ONLY=1
done
PY3=${PY3:-python3}
BLENDER_URL=https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz
OPENMVS_TAG=v2.4.0

step() { printf '\n== %s\n' "$*"; }
warn() { printf '   !! %s\n' "$*"; }

# ---------------------------------------------------------------- 0 join (box_upload.py parts)
if [ ! -f "$ZIP" ] && ls "$ZIP".part* >/dev/null 2>&1; then
  step "0/7 join $(ls "$ZIP".part* | wc -l) uploaded parts -> $ZIP"
  cat $(ls "$ZIP".part* | sort) > "$ZIP.joining" || { warn "join failed (disk full?)"; exit 1; }
  if [ -f "$ZIP.sha256" ]; then
    WANT=$(head -1 "$ZIP.sha256" | cut -d' ' -f1)
    GOT=$(sha256sum "$ZIP.joining" | cut -d' ' -f1)
    if [ "$WANT" != "$GOT" ]; then
      rm -f "$ZIP.joining"
      # Name the damaged parts from their own checksums and delete only those, so a re-run of
      # box_upload.py (which skips parts already present) sends exactly what is missing.
      BAD=0
      while read -r SUM NAME; do
        case "$NAME" in *.part[0-9]*) ;; *) continue ;; esac
        P="$(dirname "$ZIP")/$NAME"
        if [ ! -f "$P" ] || [ "$(sha256sum "$P" | cut -d' ' -f1)" != "$SUM" ]; then
          rm -f "$P"; BAD=$((BAD + 1)); echo "   bad or missing: $NAME (deleted)"
        fi
      done < "$ZIP.sha256"
      warn "checksum mismatch after joining: $BAD part(s) removed. Re-run the same box_upload.py command on"
      warn "the laptop (it re-sends only those), then run this again"
      exit 1
    fi
    echo "   sha256 OK"
  else
    warn "no $ZIP.sha256 next to the parts: joined without a whole-file check (step 2 still checks every file)"
  fi
  mv "$ZIP.joining" "$ZIP" && rm -f "$ZIP".part*
fi
[ -f "$ZIP" ] || { warn "$ZIP not found (nor its .partNNN pieces)"; exit 1; }

# ---------------------------------------------------------------- 1 unpack
step "1/7 unpack $ZIP -> $DEST"
mkdir -p "$DEST"
"$PY3" - "$ZIP" "$DEST" <<'PY' || { warn "unpack failed"; exit 1; }
import sys, zipfile
from pathlib import Path
zf, dest = zipfile.ZipFile(sys.argv[1]), Path(sys.argv[2])
prefix = "single-pass/"
n = 0
for info in zf.infolist():
    if info.is_dir() or not info.filename.startswith(prefix):
        continue
    target = dest / info.filename[len(prefix):]
    if target.exists() and target.stat().st_size == info.file_size and target.suffix.lower() in (".mp4", ".mov", ".ts"):
        continue  # a clip that is already here: skip the big copy on a re-run
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, open(target, "wb") as out:
        while chunk := src.read(1 << 20):
            out.write(chunk)
    n += 1
print(f"   {n} files written")
PY
cd "$DEST" || exit 1
sed -i 's/\r$//' scripts/*.sh 2>/dev/null   # a bundle packed from a Windows checkout may carry CRLF
chmod +x scripts/*.sh 2>/dev/null

# ---------------------------------------------------------------- 2 verify
step "2/7 verify checksums (BUNDLE.json)"
"$PY3" - <<'PY' || warn "some files did not verify: re-upload the zip"
import hashlib, json, sys
from pathlib import Path
bundle = json.loads(Path("BUNDLE.json").read_text())
bad = []
for f in bundle["files"]:
    h = hashlib.sha256()
    with open(f["path"], "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != f["sha256"]:
        bad.append(f["path"])
print(f"   {len(bundle['files']) - len(bad)} of {len(bundle['files'])} files OK; packed {bundle['created']} "
      f"from {bundle['branch']} @ {bundle['head'][:8]} (+{len(bundle['uncommitted'])} uncommitted)")
for p in bad:
    print("   BAD", p)
sys.exit(1 if bad else 0)
PY
if [ "$UNPACK_ONLY" = 1 ]; then
  echo "   --unpack-only: done (code refreshed in $DEST)"
  exit 0
fi

# ---------------------------------------------------------------- 3 machine
step "3/7 machine"
nvidia-smi -L 2>/dev/null || warn "no nvidia-smi: no GPU visible (everything will run on CPU, slowly)"
echo "   cpu quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo 'n/a')   nproc: $(nproc)"
free -g 2>/dev/null | sed -n 2p | awk '{print "   memory: " $2 " GB"}'
df -h "$DEST" | tail -1 | awk '{print "   disk free here: " $4}'

# ---------------------------------------------------------------- 4 python
step "4/7 python environment"
if [ ! -x .venv/bin/python ]; then
  "$PY3" -m venv .venv --system-site-packages || { warn "venv failed"; exit 1; }
fi
PY=.venv/bin/python
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt || warn "requirements.txt did not install cleanly (see above)"
# Track A on CUDA: pycolmap-cuda12, never together with the CPU wheel (same module name).
if ! "$PY" -c "import pycolmap, sys; sys.exit(0 if pycolmap.has_cuda else 1)" 2>/dev/null; then
  "$PY" -m pip uninstall -y -q pycolmap 2>/dev/null
  "$PY" -m pip install -q pycolmap-cuda12==4.2.0 || warn "pycolmap-cuda12 failed: Track A will run on CPU"
fi
"$PY" -m pip install -q PyNvVideoCodec av ultralytics huggingface_hub || warn "a GPU extra failed; its stage logs a CPU fallback"
"$PY" -c "import torch; print('   torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---------------------------------------------------------------- 5 tools
step "5/7 tools (git-ignored, fetched per machine)"
mkdir -p tools
if ls tools/blender-*-linux-x64/blender >/dev/null 2>&1; then
  echo "   Blender: present"
else
  echo "   Blender 4.2.3 (FBX export)"
  curl -fsSL "$BLENDER_URL" -o /tmp/blender.tar.xz && tar xf /tmp/blender.tar.xz -C tools && rm -f /tmp/blender.tar.xz \
    || warn "Blender download failed: FBX will be skipped with the reason (everything else works)"
fi
if [ -x tools/openmvs/bin/TextureMesh ]; then
  echo "   OpenMVS: present"
else
  echo "   OpenMVS $OPENMVS_TAG (Linux build: mesh + texture)"
  URL=$("$PY3" - "$OPENMVS_TAG" <<'PY'
import json, sys, urllib.request
tag = sys.argv[1]
try:
    rel = json.load(urllib.request.urlopen(f"https://api.github.com/repos/cdcseacave/openMVS/releases/tags/{tag}", timeout=30))
except Exception as exc:
    sys.exit(f"github api: {exc}")
assets = [a for a in rel.get("assets", []) if any(k in a["name"].lower() for k in ("ubuntu", "linux"))]
print(assets[0]["browser_download_url"] if assets else "")
PY
)
  if [ -n "$URL" ]; then
    mkdir -p /tmp/openmvs && curl -fsSL "$URL" -o "/tmp/openmvs/$(basename "$URL")" || warn "OpenMVS download failed"
    ( cd /tmp/openmvs && for f in *; do case "$f" in *.zip) "$PY3" -m zipfile -e "$f" . ;; *.tar*|*.tgz) tar xf "$f" ;; esac; done )
    BIN=$(dirname "$(find /tmp/openmvs -type f -name TextureMesh | head -1)" 2>/dev/null)
    if [ -n "$BIN" ] && [ -f "$BIN/TextureMesh" ]; then
      mkdir -p tools/openmvs/bin && cp "$BIN"/* tools/openmvs/bin/ && chmod +x tools/openmvs/bin/* && rm -rf /tmp/openmvs
      echo "   OpenMVS: installed into tools/openmvs/bin"
    else
      warn "OpenMVS archive had no TextureMesh; put the Linux binaries in tools/openmvs/bin by hand"
    fi
  else
    warn "no Linux asset found for OpenMVS $OPENMVS_TAG: download it from github.com/cdcseacave/openMVS/releases"
    warn "and copy the binaries into tools/openmvs/bin (without it: Poisson mesh, no texture)"
  fi
fi
[ -d tools/vggt/.git ] || git clone -q --depth 1 https://github.com/facebookresearch/vggt tools/vggt \
  || warn "VGGT code clone failed"
[ -d tools/vggt_omega/.git ] || git clone -q --depth 1 https://github.com/facebookresearch/vggt-omega tools/vggt_omega \
  || warn "VGGT-Omega code clone failed"
echo "   tools/: $(ls tools | tr '\n' ' ')"

# ---------------------------------------------------------------- 6 weights
step "6/7 VGGT-Omega weights (gated)"
if [ -z "${HF_TOKEN:-}" ] && ! "$PY" -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
  warn "not logged in to Hugging Face. In THIS terminal only (never in a file):"
  warn "    export HF_TOKEN=hf_...      # a fresh, read-only token; the old one was rotated"
  warn "then re-run this script. Without it Track B falls back to VGGT-1B, and Stage 3's fill uses"
  warn "the depth maps Track B saved — or reports the gaps unfilled."
else
  "$PY" - <<'PY' || warn "VGGT-Omega checkpoint download failed (account approved for facebook/VGGT-Omega?)"
from huggingface_hub import hf_hub_download
path = hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt")
print("   checkpoint:", path)
PY
fi

# ---------------------------------------------------------------- 7 check
step "7/7 check"
"$PY" -c "from src.core.device import device_info, cpu_thread_budget; print('  ', device_info(), 'threads', cpu_thread_budget())"
CLIP=$(ls data/raw/*.mp4 2>/dev/null | head -1)
[ -n "$CLIP" ] && "$PY" -m src.cli inspect "$CLIP" 2>&1 | grep -iE "decode|resolution|duration" | head -5
if [ "$RUN_TESTS" = 1 ]; then
  "$PY" -m pytest tests -q -x 2>&1 | tail -3
fi

cat <<EOF

== restored in $DEST
Next, the Stage 3 box runs (CLOUD_GPU_GUIDE.md §10.4), in the background:
  cd $DEST && nohup bash scripts/box_stage3_runs.sh > box_runs.log 2>&1 &
  tail -f box_runs.log      # Ctrl+C stops watching only; at the end, paste box_runs_report.json back
Before handing the box back: bash scripts/box_collect.sh esri_s3, then scripts/box_wipe_credentials.sh.
EOF
