"""Laptop side of the Google Drive route: checksum the bundle, write the box's paste-in commands.

    python scripts/box_gdrive.py                       # before uploading: size, sha256, what to do
    python scripts/box_gdrive.py --link "https://drive.google.com/file/d/<id>/view?usp=sharing"

1. Upload ``data/box_upload/box_upload_<stamp>.zip`` at drive.google.com (New -> File upload).
2. Share it: Share -> General access -> "Anyone with the link" (Viewer). Copy the link.
3. Run this with ``--link``. It writes ``data/box_upload/box_fetch_paste.txt``: one block to paste
   into the box's Jupyter terminal. The block creates ``~/box_fetch_gdrive.sh`` (gdown download with
   resume, sha256 check) and runs it, which then runs the full restore.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "scripts" / "box_fetch_gdrive.sh"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def drive_id(link: str) -> str | None:
    """The file id from any Drive link form (/file/d/<id>/, ?id=<id>, open?id=) or a bare id."""
    for pattern in (r"/file/d/([\w-]{20,})", r"[?&]id=([\w-]{20,})"):
        m = re.search(pattern, link)
        if m:
            return m.group(1)
    return link if re.fullmatch(r"[\w-]{20,}", link) else None


def paste_block(file_id: str, digest: str, extra: str) -> str:
    script = FETCH.read_text(encoding="utf-8").replace("\r\n", "\n")
    return (
        "# ---- paste everything below into the box's Jupyter terminal ----\n"
        "export HF_TOKEN=hf_...   # EDIT: your fresh read-only token (this shell only), or delete this line\n"
        "cat > ~/box_fetch_gdrive.sh <<'BOX_FETCH_EOF'\n"
        f"{script.rstrip()}\n"
        "BOX_FETCH_EOF\n"
        "# Runs in the background (no tmux needed): a closed tab does not stop it.\n"
        f"cd ~ && nohup bash ~/box_fetch_gdrive.sh {file_id} {digest}{extra} > ~/box_setup.log 2>&1 &\n"
        "sleep 2; tail -f ~/box_setup.log   # Ctrl+C stops watching only; reopen with: tail -f ~/box_setup.log\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", nargs="?", help="bundle (default: newest in data/box_upload)")
    ap.add_argument("--link", help="the Drive share link (or file id) after uploading")
    ap.add_argument("--unpack-only", action="store_true", help="code refresh on a box that is already set up")
    ap.add_argument("--skip-sha", action="store_true",
                    help="the zip on Drive is an earlier bundle than the newest local one: skip the whole-zip "
                         "checksum (every file inside is still checked when it is unpacked)")
    args = ap.parse_args()

    if args.zip:
        zip_path = Path(args.zip)
    else:
        found = sorted((ROOT / "data" / "box_upload").glob("box_upload_*.zip"))
        if not found:
            sys.exit("no bundle in data/box_upload: run scripts/box_pack.py first")
        zip_path = found[-1]
    digest = '""' if args.skip_sha else sha256(zip_path)
    if args.skip_sha:
        print("whole-zip checksum skipped (--skip-sha); box_restore.sh still verifies every file in the zip")
    else:
        (zip_path.parent / f"{zip_path.name}.sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(f"bundle: {zip_path}  ({zip_path.stat().st_size / 1e6:,.0f} MB)")
    print(f"sha256: {digest}")

    if not args.link:
        print("\nNext:")
        print("  1. drive.google.com -> New -> File upload -> pick the bundle above")
        print("  2. right-click it -> Share -> General access: 'Anyone with the link' (Viewer) -> Copy link")
        print('  3. .venv\\Scripts\\python scripts\\box_gdrive.py --link "<the link>"')
        return 0

    file_id = drive_id(args.link)
    if not file_id:
        sys.exit(f"no Drive file id in {args.link!r}; copy the link from Share -> Copy link")
    block = paste_block(file_id, digest, " --unpack-only" if args.unpack_only else "")
    out = zip_path.parent / "box_fetch_paste.txt"
    out.write_text(block, encoding="utf-8", newline="\n")
    print(f"\nWrote {out}")
    print("Open it, edit the HF_TOKEN line, copy everything, and paste it into the box's Jupyter terminal")
    print("It runs in the background and logs to ~/box_setup.log. If the box already has the script:")
    print(f"  nohup bash ~/box_fetch_gdrive.sh {file_id} {digest}" + (" --unpack-only" if args.unpack_only else "")
          + " > ~/box_setup.log 2>&1 &  then  tail -f ~/box_setup.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
