"""Pack one zip with everything a (fresh) GPU box needs; unpack it there with box_restore.sh.

    python scripts/box_pack.py                          # code + default clips
    python scripts/box_pack.py --with-box-runs          # ... + the last handover's evidence (no exports)
    python scripts/box_pack.py --clip path/to/clip.mp4 --clip path/to/clip_folder --no-default-clips

What goes in (under ``single-pass/`` in the zip):
  * the code **as it is in the working tree** — tracked files plus new files git does not ignore,
    so uncommitted work travels too (the box needs no GitHub token to get it);
  * clips into ``data/raw/``: a file as is, a folder (video + its telemetry) as a sub-folder;
  * optionally ``data/box/runs`` without the ``export/`` folders (manifests, georef, sparse models);
  * ``BUNDLE.json``: git state, and a size + sha256 for every file, which box_restore.sh checks.

What never goes in: tokens or credentials (the pack stops if a code file looks like one),
model weights, OpenMVS / Blender binaries (fetched on the box), and the Python environment.
Videos are stored, not deflated (they are already compressed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS = [ROOT / "data" / "raw" / "Esri_multiplexer_1.mp4",
                 ROOT.parent / "Drone Video Dataset" / "QGISFMV_Samples" / "DJI" / "DJI_0047"]
STORED = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".ts", ".mpg", ".mpeg", ".h264", ".jpg", ".jpeg", ".png",
          ".zip", ".gz", ".xz", ".tif", ".tiff", ".glb", ".las", ".laz"}
SECRET = re.compile(rb"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
                    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
PREFIX = "single-pass"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def code_files() -> list[Path]:
    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
    files = [ROOT / n for n in names if n and (ROOT / n).is_file()]
    return [f for f in files if not f.name.startswith(".env")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plan(args) -> list[tuple[Path, str]]:
    """(source file, path inside the zip) for everything that will be packed."""
    items = [(f, f"{PREFIX}/{f.relative_to(ROOT).as_posix()}") for f in code_files()]
    clips = ([] if args.no_default_clips else [c for c in DEFAULT_CLIPS if c.exists()]) + [Path(c) for c in args.clip]
    for clip in clips:
        clip = clip.resolve()
        if clip.is_file():
            items.append((clip, f"{PREFIX}/data/raw/{clip.name}"))
            for sidecar in clip.parent.glob(clip.stem + ".*"):          # DJI .SRT next to the video
                if sidecar != clip and sidecar.suffix.lower() in (".srt", ".csv", ".txt", ".gpx", ".kml"):
                    items.append((sidecar, f"{PREFIX}/data/raw/{sidecar.name}"))
        elif clip.is_dir():
            for f in sorted(clip.rglob("*")):
                if f.is_file():
                    items.append((f, f"{PREFIX}/data/raw/{clip.name}/{f.relative_to(clip).as_posix()}"))
        else:
            sys.exit(f"clip not found: {clip}")
    if args.with_box_runs:
        runs = ROOT / "data" / "box" / "runs"
        for f in sorted(runs.rglob("*")):
            rel = f.relative_to(runs)
            if f.is_file() and "export" not in rel.parts:
                items.append((f, f"{PREFIX}/data/box/runs/{rel.as_posix()}"))
    seen: dict[str, Path] = {}
    for src, arc in items:
        seen.setdefault(arc, src)
    return [(src, arc) for arc, src in seen.items()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", default=[], help="video file or clip folder to include (repeatable)")
    ap.add_argument("--no-default-clips", action="store_true", help="leave out Esri and DJI_0047")
    ap.add_argument("--with-box-runs", action="store_true", help="include data/box/runs (no export/ folders)")
    ap.add_argument("--out", default=str(ROOT / "data" / "box_upload"), help="output folder")
    args = ap.parse_args()

    items = plan(args)
    code = [src for src, arc in items if src.is_relative_to(ROOT) and not arc.startswith(f"{PREFIX}/data/")]
    leaks = [str(f.relative_to(ROOT)) for f in code if f.stat().st_size < 5_000_000 and SECRET.search(f.read_bytes())]
    if leaks:
        print("STOP: these files look like they contain a token or private key; nothing was packed:")
        print("\n".join(f"  {p}" for p in leaks))
        return 2

    status = git("status", "--porcelain")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    target = out_dir / f"box_upload_{stamp}.zip"
    total = sum(src.stat().st_size for src, _ in items)
    print(f"packing {len(items)} files, {total / 1e6:,.0f} MB -> {target}")

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head": git("rev-parse", "HEAD").strip(), "uncommitted": status.splitlines(),
        "note": "code is the working tree at pack time, uncommitted changes included",
        "not_included": ["model weights (Hugging Face cache on the box)", "OpenMVS and Blender binaries (fetched by "
                         "box_restore.sh)", "the Python environment", "tokens and credentials"],
        "files": [],
    }
    with zipfile.ZipFile(target, "w", allowZip64=True) as zf:
        for n, (src, arc) in enumerate(items, 1):
            method = zipfile.ZIP_STORED if src.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            if src.suffix == ".sh":
                # Shell scripts run on Linux: a Windows checkout (core.autocrlf) gives them CRLF, and bash
                # then fails with "$'\r': command not found" (it did, on the box, 2026-09-23).
                data = src.read_bytes().replace(b"\r\n", b"\n")
                zf.writestr(zipfile.ZipInfo.from_file(src, arc), data, compress_type=method)
                digest, size = hashlib.sha256(data).hexdigest(), len(data)
            else:
                zf.write(src, arc, compress_type=method)
                digest, size = sha256(src), src.stat().st_size
            manifest["files"].append({"path": arc[len(PREFIX) + 1:], "bytes": size, "sha256": digest})
            if src.stat().st_size > 50_000_000:
                print(f"  [{n}/{len(items)}] {arc} ({src.stat().st_size / 1e6:,.0f} MB)")
        zf.writestr(f"{PREFIX}/BUNDLE.json", json.dumps(manifest, indent=1))

    print(f"\nDONE: {target} ({target.stat().st_size / 1e6:,.0f} MB)")
    if status.strip():
        print(f"  includes {len(status.splitlines())} uncommitted change(s) from the working tree")
    print("\nNext:")
    print("  1. upload it to drive.google.com, share 'Anyone with the link', copy the link")
    print('  2. .venv\\Scripts\\python scripts\\box_gdrive.py --link "<the link>"   -> box_fetch_paste.txt')
    print("  3. paste that file into the box's Jupyter terminal")
    print("  (no Drive? scripts/box_upload.py sends it straight to the Jupyter server)")
    print("  Full steps: CLOUD_GPU_GUIDE.md section 10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
