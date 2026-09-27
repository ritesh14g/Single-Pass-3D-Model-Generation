"""Drive Kaggle GPU runs from the laptop (CLOUD_GPU_GUIDE.md §11). The laptop stays the only source.

    python scripts/kaggle_ctl.py whoami                 # the Kaggle CLI is set up and logged in
    python scripts/kaggle_ctl.py data                   # upload the clips once (single-pass-data, ~1.2 GB)
    python scripts/kaggle_ctl.py code                   # upload the working tree (single-pass-code), wait until ready
    python scripts/kaggle_ctl.py run check              # a preset job (see PRESETS), or:
    python scripts/kaggle_ctl.py run myjob --cmd "bash scripts/box_stage6.sh" --collect esri_s6 dji47_s6
    python scripts/kaggle_ctl.py status | logs | wait   # the latest run of the job notebook
    python scripts/kaggle_ctl.py pull myjob             # outputs -> data/box/kaggle/myjob/

``run`` uploads the code first unless ``--no-code``. Every job is a new version of ONE private notebook,
``<user>/single-pass-gpu``, so the HF_TOKEN secret is attached once (Kaggle editor -> Add-ons -> Secrets).
The Kaggle CLI (>= 2.x, for --accelerator) is found on PATH or in ~/.local/bin (``uv tool install kaggle``);
credentials are the CLI's own (~/.kaggle/kaggle.json or KAGGLE_API_TOKEN), never read or printed here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "kaggle"
KERNEL = "single-pass-gpu"
DATA_DS, CODE_DS = "single-pass-data", "single-pass-code"
DJI_DIR = ROOT.parent / "Drone Video Dataset" / "QGISFMV_Samples" / "DJI" / "DJI_0047"
CLIPS = [ROOT / "data/raw/Esri_multiplexer_1.mp4", DJI_DIR / "DJI_0047.mp4", DJI_DIR / "telemetry.csv"]

PRESETS = {
    # setup + test suite + a quick look at a clip: proves the machine before a long run (~20 min)
    "check": {"cmd": "$PY -m src.cli inspect data/raw/Esri_multiplexer_1.mp4", "tests": True, "collect": []},
    # the Stage 6 box run: Esri + DJI_0047 fresh, then the degradation bench (~4 h on a T4)
    "stage6": {"cmd": "bash scripts/box_stage6.sh", "collect": ["esri_s6", "dji47_s6"]},
    "stage6-nobench": {"cmd": "BENCH=0 bash scripts/box_stage6.sh", "collect": ["esri_s6", "dji47_s6"]},
}


def kaggle_exe() -> str:
    for cand in (shutil.which("kaggle"), str(Path.home() / ".local/bin/kaggle.exe"), str(Path.home() / ".local/bin/kaggle")):
        if cand and Path(cand).exists():
            out = subprocess.run([cand, "--version"], capture_output=True, text=True).stdout
            if "CLI 2." in out or "CLI 3." in out:
                return cand
    sys.exit("Kaggle CLI 2.x not found: `uv tool install kaggle --python 3.12` (the 1.7 CLI has no --accelerator)")


def kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [kaggle_exe(), *args]
    print("$ kaggle", " ".join(args), flush=True)
    res = subprocess.run(cmd, text=True, capture_output=capture)
    if capture and res.stdout:
        print(res.stdout.strip())
    if check and res.returncode != 0:
        sys.exit(f"kaggle {args[0]} {args[1] if len(args) > 1 else ''} failed ({res.returncode}): {(res.stderr or '').strip()}")
    return res


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cred = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if cred.is_file():
        return json.loads(cred.read_text())["username"]
    # New-style access token (~/.kaggle/access_token or KAGGLE_API_TOKEN): ask the CLI who it is.
    out = subprocess.run([kaggle_exe(), "config", "view"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().lower().startswith("- username:") or line.strip().lower().startswith("username:"):
            name = line.split(":", 1)[1].strip()
            if name and name.lower() != "none":
                return name
    sys.exit("no Kaggle username: put kaggle.json in ~/.kaggle (Kaggle -> Settings -> API -> Create New Token)"
             " or set KAGGLE_USERNAME")


def dataset_exists(ref: str) -> bool:
    return kaggle("datasets", "status", ref, check=False, capture=True).returncode == 0


def wait_ready(ref: str, timeout_s: int = 1800) -> None:
    t = time.time()
    while time.time() - t < timeout_s:
        out = kaggle("datasets", "status", ref, check=False, capture=True).stdout.strip().lower()
        if "ready" in out:
            return
        if "error" in out:
            sys.exit(f"{ref}: {out}")
        time.sleep(15)
    sys.exit(f"{ref} not ready after {timeout_s} s")


def publish(folder: Path, slug: str, message: str) -> None:
    ref = f"{username()}/{slug}"
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": ref, "licenses": [{"name": "other"}]}, indent=2))
    if dataset_exists(ref):
        kaggle("datasets", "version", "-p", str(folder), "-m", message, "-t", "-r", "skip")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-t", "-r", "skip")
    wait_ready(ref)
    print(f"{ref}: ready")


def cmd_data(_: argparse.Namespace) -> None:
    folder = STAGE / "data"
    folder.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        if not clip.is_file():
            sys.exit(f"missing clip {clip}")
        dest = folder / clip.name
        if not dest.exists():
            try:
                os.link(clip, dest)  # same volume: no 1.2 GB copy
            except OSError:
                shutil.copy2(clip, dest)
    publish(folder, DATA_DS, "clips")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def cmd_code(_: argparse.Namespace) -> None:
    folder = STAGE / "code"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    files = [f for f in git("ls-files", "-co", "--exclude-standard").splitlines()
             if f and not f.startswith("data/") and (ROOT / f).is_file()]
    info = {"branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(), "head": git("rev-parse", "HEAD").strip(),
            "uncommitted": git("status", "--porcelain").splitlines(), "created": dt.datetime.now().isoformat(timespec="seconds")}
    with zipfile.ZipFile(folder / "code.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(ROOT / f, f)
        zf.writestr("CODE.json", json.dumps(info, indent=2))
    print(f"code.zip: {len(files)} files, {(folder / 'code.zip').stat().st_size / 1e6:.1f} MB, "
          f"{info['branch']} @ {info['head'][:8]} +{len(info['uncommitted'])} uncommitted")
    publish(folder, CODE_DS, f"{info['head'][:8]} {info['created']}")


def cmd_run(a: argparse.Namespace) -> None:
    job = dict(PRESETS.get(a.name, {}))
    if a.cmd:
        job["cmd"] = a.cmd
    if a.collect:
        job["collect"] = a.collect
    if a.tests:
        job["tests"] = True
    if not job.get("cmd") and not job.get("tests"):
        sys.exit(f"unknown preset {a.name!r} and no --cmd (presets: {', '.join(PRESETS)})")
    job.update({"name": a.name, "env": dict(e.split("=", 1) for e in a.env), "pushed": dt.datetime.now().isoformat(timespec="seconds")})
    if not a.no_code:
        cmd_code(a)
    user = username()
    folder = STAGE / "kernel"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    body = (ROOT / "scripts/kaggle_job.py").read_text(encoding="utf-8")
    (folder / "job.py").write_text(f"JOB = {job!r}\n" + body, encoding="utf-8")   # a Python literal, not JSON
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL}", "title": KERNEL, "code_file": "job.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATA_DS}", f"{user}/{CODE_DS}"], "competition_sources": [],
        "kernel_sources": [], "machine_shape": a.accelerator}, indent=2))
    kaggle("kernels", "push", "-p", str(folder), "--accelerator", a.accelerator)
    (STAGE / "last_job.json").write_text(json.dumps(job, indent=2))
    print(f"pushed {a.name!r} on {a.accelerator}: https://www.kaggle.com/code/{user}/{KERNEL}")


def status_text() -> str:
    return kaggle("kernels", "status", f"{username()}/{KERNEL}", check=False, capture=True).stdout.lower()


def cmd_status(_: argparse.Namespace) -> None:
    status_text()


def cmd_logs(_: argparse.Namespace) -> None:
    kaggle("kernels", "logs", f"{username()}/{KERNEL}", check=False)


def cmd_wait(a: argparse.Namespace) -> None:
    while True:
        s = status_text()
        if any(k in s for k in ("complete", "error", "cancel")):
            return
        time.sleep(a.every)


def cmd_pull(a: argparse.Namespace) -> None:
    dest = ROOT / "data/box/kaggle" / a.name
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{username()}/{KERNEL}", "-p", str(dest), "-o")
    for tar in dest.glob("box_handover_*.tar.gz"):
        print(f"handover: {tar}  (unpack into data/box/ as after a box session)")
    print(f"outputs in {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd_name", required=True)
    sub.add_parser("whoami").set_defaults(fn=lambda a: print(username(), kaggle_exe()))
    sub.add_parser("data").set_defaults(fn=cmd_data)
    sub.add_parser("code").set_defaults(fn=cmd_code)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--cmd")
    r.add_argument("--collect", nargs="*")
    r.add_argument("--tests", action="store_true")
    r.add_argument("--env", nargs="*", default=[], help="KEY=VALUE for the job's command")
    r.add_argument("--accelerator", default="NvidiaTeslaT4", help="NvidiaTeslaT4 | NvidiaL4 (if your quota has it)")
    r.add_argument("--no-code", action="store_true", help="reuse the last uploaded code")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("logs").set_defaults(fn=cmd_logs)
    w = sub.add_parser("wait")
    w.add_argument("--every", type=int, default=60)
    w.set_defaults(fn=cmd_wait)
    pl = sub.add_parser("pull")
    pl.add_argument("name")
    pl.set_defaults(fn=cmd_pull)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
