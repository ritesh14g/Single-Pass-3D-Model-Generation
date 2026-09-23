#!/usr/bin/env bash
# On the GPU box: download the bundle from Google Drive with gdown, verify it, then restore.
#
#   bash box_fetch_gdrive.sh "<Drive share link or file id>" [sha256] [box_restore.sh options]
#
#   bash box_fetch_gdrive.sh "https://drive.google.com/file/d/1AbC.../view?usp=sharing" 3f9c...e1
#   bash box_fetch_gdrive.sh 1AbC... "" --unpack-only        # code refresh, no checksum
#
# The file must be shared "Anyone with the link" (Viewer). The download resumes: after a drop,
# run the same command again. With a sha256 (scripts/box_gdrive.py prints it) a wrong or partial
# file is caught before anything is unpacked. Then it hands over to box_restore.sh, from the zip.
# Environment: OUT (default ~/box_upload.zip), PY3 (default python3).
set -uo pipefail

LINK=${1:?usage: bash box_fetch_gdrive.sh <drive link or file id> [sha256] [restore options]}
SHA=${2:-}
shift; [ $# -gt 0 ] && shift
OUT=${OUT:-$HOME/box_upload.zip}
PY3=${PY3:-python3}

say()  { printf '\n== %s\n' "$*"; }
warn() { printf '   !! %s\n' "$*"; }

# ---------------------------------------------------------------- gdown
say "gdown"
GD_PY=$PY3
if ! "$GD_PY" -c "import gdown" 2>/dev/null; then
  "$PY3" -m pip install -q --user --upgrade gdown 2>/dev/null
  if ! "$PY3" -c "import gdown" 2>/dev/null; then
    # The system Python refuses user installs (PEP 668): a small private venv just for gdown.
    "$PY3" -m venv "$HOME/.cache/gdown-venv" && "$HOME/.cache/gdown-venv/bin/python" -m pip install -q gdown \
      && GD_PY=$HOME/.cache/gdown-venv/bin/python || { warn "could not install gdown"; exit 1; }
  fi
fi
"$GD_PY" -c "import gdown; print('   gdown', gdown.__version__)"

# ---------------------------------------------------------------- download
say "download -> $OUT"
ok=0
for attempt in 1 2 3 4 5; do
  if "$GD_PY" - "$LINK" "$OUT" <<'PY'
import inspect, re, sys
import gdown
link, out = sys.argv[1], sys.argv[2]
# The file id from any Drive link form (/file/d/<id>/, ?id=<id>) or a bare id. Passing the plain
# uc?id= URL works in every gdown version (6.x dropped the "fuzzy" link parsing).
m = re.search(r"/file/d/([\w-]{20,})", link) or re.search(r"[?&]id=([\w-]{20,})", link)
file_id = m.group(1) if m else link
if not re.fullmatch(r"[\w-]{20,}", file_id):
    sys.exit(f"no Google Drive file id in {link!r}")
params = inspect.signature(gdown.download).parameters
kw = {"output": out, "quiet": False}
if "resume" in params:
    kw["resume"] = True                                   # continue a partial file
if "retries" in params:
    kw["retries"] = 3
try:
    result = gdown.download(f"https://drive.google.com/uc?id={file_id}", **kw)
except Exception as exc:  # noqa: BLE001
    if type(exc).__name__ == "FileURLRetrievalError":
        # Drive would not hand out the file: sharing or the daily quota. Retrying cannot help.
        print(f"   Drive refused the file: {str(exc).splitlines()[2].strip() if len(str(exc).splitlines()) > 2 else exc}")
        sys.exit(3)
    print(f"   {type(exc).__name__}: {exc}")
    sys.exit(1)
sys.exit(0 if result else 1)
PY
  then ok=1; break; else code=$?; fi
  [ "$code" = 3 ] && break
  warn "attempt $attempt failed; retrying in $((attempt * 10)) s (the partial file is kept)"
  sleep $((attempt * 10))
done
if [ "$ok" != 1 ]; then
  warn "download failed. Usual causes:"
  warn " - sharing: in Drive, Share -> General access -> 'Anyone with the link' (Viewer)"
  warn " - 'Too many users have viewed or downloaded this file': Drive's daily quota for the file;"
  warn "   in Drive right-click -> Make a copy, share the copy, and use its link"
  exit 1
fi

# Drive answers a permission problem with an HTML page, which gdown may save under the zip's name.
if ! "$PY3" -c "import sys, zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" "$OUT"; then
  warn "$OUT is not a zip (first bytes: $(head -c 60 "$OUT" | tr -d '\n'))"
  warn "that is Drive's web page, not the file: check the sharing setting, delete $OUT, and run again"
  exit 1
fi

# ---------------------------------------------------------------- verify
if [ -n "$SHA" ]; then
  say "sha256"
  GOT=$(sha256sum "$OUT" | cut -d' ' -f1)
  if [ "$GOT" != "$SHA" ]; then
    warn "checksum mismatch: got $GOT"
    warn "the file on Drive is not the bundle you packed (or the download is damaged): rm $OUT and run again"
    exit 1
  fi
  echo "   OK"
fi

# ---------------------------------------------------------------- restore
say "restore (box_restore.sh from the zip)"
"$PY3" - "$OUT" "$HOME/box_restore.sh" <<'PY' || { warn "no box_restore.sh in the zip"; exit 1; }
import sys, zipfile
data = zipfile.ZipFile(sys.argv[1]).read("single-pass/scripts/box_restore.sh").replace(b"\r\n", b"\n")
open(sys.argv[2], "wb").write(data)
PY
exec bash "$HOME/box_restore.sh" "$OUT" "$@"
