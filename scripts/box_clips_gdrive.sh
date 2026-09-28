#!/usr/bin/env bash
# On the GPU box: the flight clips from a box_pack.py bundle on Google Drive, into a git checkout.
#
#   bash scripts/box_clips_gdrive.sh <drive file id or link>
#   bash scripts/box_clips_gdrive.sh 1nLA99rY1DJoSQlw-HZIQT_olUKiPe3SH     # the team bundle (2026-09-23)
#
# Only ``single-pass/data/raw/*`` is extracted (Esri, DJI_0047 + telemetry.csv in the team bundle), so
# the bundle's older code never overwrites the checkout. The file must be shared "Anyone with the
# link". The zip is deleted afterwards. Environment: PY3 (default python3).
set -uo pipefail
ID=${1:?usage: bash scripts/box_clips_gdrive.sh <drive file id or link>}
cd "$(dirname "$0")/.." || exit 1
PY3=${PY3:-python3}
ZIP=${ZIP:-$HOME/clips_bundle.zip}

GD_PY=$PY3
if ! "$GD_PY" -c "import gdown" 2>/dev/null; then
  "$PY3" -m pip install -q --user --upgrade gdown 2>/dev/null
  if ! "$PY3" -c "import gdown" 2>/dev/null; then
    "$PY3" -m venv "$HOME/.cache/gdown-venv" && "$HOME/.cache/gdown-venv/bin/python" -m pip install -q gdown \
      && GD_PY=$HOME/.cache/gdown-venv/bin/python || { echo "!! could not install gdown"; exit 1; }
  fi
fi

"$GD_PY" - "$ID" "$ZIP" <<'PY' || exit 1
import re, sys, zipfile
from pathlib import Path
import gdown
link, out = sys.argv[1], Path(sys.argv[2])
m = re.search(r"(?:/d/|id=)([A-Za-z0-9_-]{20,})", link)
file_id = m.group(1) if m else link
if not (out.exists() and zipfile.is_zipfile(out)):
    gdown.download(id=file_id, output=str(out), quiet=False)
if not zipfile.is_zipfile(out):
    sys.exit("!! not a zip (is the Drive file shared 'Anyone with the link'?)")
n = 0
with zipfile.ZipFile(out) as zf:
    for info in zf.infolist():
        if info.is_dir() or not info.filename.startswith("single-pass/data/raw/"):
            continue
        dest = Path(info.filename[len("single-pass/"):])
        if dest.exists() and dest.stat().st_size == info.file_size:
            print("   already here:", dest); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(dest, "wb") as fh:
            while chunk := src.read(1 << 20):
                fh.write(chunk)
        n += 1
        print(f"   {dest}  {info.file_size / 1e6:.0f} MB")
print(f"   {n} file(s) extracted into data/raw")
PY
rm -f "$ZIP"
ls -la data/raw data/raw/DJI_0047 2>/dev/null
