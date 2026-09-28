"""The GPU job that runs on Kaggle (pushed by ``scripts/kaggle_ctl.py``, never run by hand).

Kaggle stands in for the institute box (CLOUD_GPU_GUIDE.md §11). The laptop stays the only source:
``kaggle_ctl.py`` uploads the working tree as the private dataset ``single-pass-code`` and the clips
as ``single-pass-data``, then pushes this file with a ``JOB = {...}`` line prepended. Here it:

  1 records the machine (GPU, CPU, memory, disk)
  2 unpacks the code onto scratch disk (only /kaggle/working is kept, max 20 GB) and links the clips
  3 installs the requirements into the job's own interpreter (Kaggle's, with CUDA torch): no .venv,
    Kaggle's python3 cannot build one with pip and the container is thrown away after the job
  4 fetches Blender 4.2.3, OpenMVS 2.4.0 (Ubuntu), VGGT and VGGT-Omega code
  5 pre-fetches the VGGT-Omega checkpoint with the HF_TOKEN Kaggle secret (attached to this
    notebook once, in the Kaggle editor: Add-ons -> Secrets); without it Track B logs VGGT-1B
  6 optionally runs the test suite, then JOB["cmd"] with bash in the project folder
  7 collects: scripts/box_collect.sh <runs> -> /kaggle/working, plus every *.log

Each step logs and carries on when it fails, like the pipeline: GPU first, CPU fallback, never a crash.
"""

# No `from __future__` import: kaggle_ctl.py prepends the JOB line above this docstring. PEP 604
# annotations (Path | None) work at runtime on Kaggle's Python (>= 3.10).
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

JOB: dict = globals().get("JOB") or {"name": "check", "cmd": "", "collect": [], "tests": True}

OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
BLENDER_URL = "https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz"
OPENMVS_URL = "https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Ubuntu_x64.zip"
REPOS = {"tools/vggt": "https://github.com/facebookresearch/vggt",
         "tools/vggt_omega": "https://github.com/facebookresearch/vggt-omega"}
CLIPS = {"Esri_multiplexer_1.mp4": "data/raw/Esri_multiplexer_1.mp4",
         "DJI_0047.mp4": "data/raw/DJI_0047/DJI_0047.mp4",
         "telemetry.csv": "data/raw/DJI_0047/telemetry.csv"}

OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "job.log"
T0 = time.time()
RESULT: dict = {"job": JOB, "steps": {}}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sh(cmd: str, cwd: Path | None = None, env: dict | None = None, tee: Path | None = None) -> int:
    """Run a bash command, streaming its output to the Kaggle log (and ``tee``)."""
    log(f"$ {cmd}")
    proc = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    fh = tee.open("a", encoding="utf-8") if tee else None
    for line in proc.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        if fh:
            fh.write(line)
    if fh:
        fh.close()
    return proc.wait()


def step(name: str):
    def wrap(fn):
        t = time.time()
        log(f"== {name}")
        try:
            value = fn()
            RESULT["steps"][name] = {"ok": True, "s": round(time.time() - t, 1), "value": value}
        except Exception as exc:  # noqa: BLE001 - every step logs and the job carries on
            log(f"   !! {name} failed: {type(exc).__name__}: {exc}")
            RESULT["steps"][name] = {"ok": False, "s": round(time.time() - t, 1), "error": str(exc)}
        (OUT / "job_result.json").write_text(json.dumps(RESULT, indent=2, default=str))
        return fn
    return wrap


def scratch() -> Path:
    """The roomiest writable disk that is not /kaggle/working (whose 20 GB is kept as output)."""
    best, free = Path("/tmp"), 0
    for cand in (Path("/kaggle/tmp"), Path("/tmp"), Path.home()):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            f = shutil.disk_usage(cand).free
        except OSError:
            continue
        if f > free:
            best, free = cand, f
    return best / "single-pass"


PROJ = scratch()
PY = Path(sys.executable)


def _clips_from_drive(file_id: str) -> dict:
    """The clips from a box_pack.py bundle on Google Drive ("Anyone with the link"): at datacenter
    speed instead of a slow home upload. Only ``single-pass/data/raw/*`` members are extracted."""
    bundle = Path("/tmp/clips_bundle.zip")
    if not bundle.exists():
        sh(f"{PY} -m pip install -q gdown")
        rc = sh(f"{PY} -c \"import gdown; gdown.download(id='{file_id}', output='{bundle}', quiet=False)\"")
        if rc != 0 or not bundle.exists() or not zipfile.is_zipfile(bundle):
            log("   !! Drive download failed (is the file shared 'Anyone with the link'?)")
            bundle.unlink(missing_ok=True)
            return {}
    got = {}
    with zipfile.ZipFile(bundle) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.startswith("single-pass/data/raw/"):
                continue
            dest = PROJ / info.filename[len("single-pass/"):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            got[dest.name] = f"drive:{file_id}"
    bundle.unlink(missing_ok=True)
    log(f"   clips from Drive: {sorted(got)}")
    return got


@step("machine")
def _machine():
    sh("nvidia-smi; nvidia-smi -L; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; free -g; df -h /kaggle/working "
       f"{PROJ.parent}; python3 -V", tee=OUT / "machine.txt")
    return {"project": str(PROJ)}


@step("code")
def _code():
    stages = [p for p in INPUT.rglob("stages.py") if p.parent.name == "src"]
    if PROJ.exists():
        shutil.rmtree(PROJ)
    if stages:  # Kaggle unpacked code.zip into a folder
        shutil.copytree(stages[0].parent.parent, PROJ)
    else:
        archive = next(INPUT.rglob("code.zip"))
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(PROJ)
    sh(f"chmod -R u+w {PROJ}")                       # /kaggle/input is read-only and copytree kept that
    for sh_file in (PROJ / "scripts").glob("*.sh"):  # a Windows checkout carries CRLF
        sh_file.write_bytes(sh_file.read_bytes().replace(b"\r\n", b"\n"))
        sh_file.chmod(0o755)
    info = PROJ / "CODE.json"
    code = json.loads(info.read_text()) if info.is_file() else {}
    log(f"   code {code.get('branch')} @ {str(code.get('head'))[:8]} (+{len(code.get('uncommitted', []))} uncommitted),"
        f" packed {code.get('created')}")
    linked = {}
    drive_id = JOB.get("clips_gdrive")
    if drive_id:
        linked.update(_clips_from_drive(str(drive_id)))
    for name, rel in CLIPS.items():
        if name in linked:
            continue
        hits = sorted(INPUT.rglob(name))
        if hits:
            dest = PROJ / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(hits[0])
            linked[name] = str(hits[0])
        else:
            log(f"   !! clip {name} not found under {INPUT}")
    return {"code": code, "clips": linked}


@step("python")
def _python():
    pip = f"{PY} -m pip install -q"
    sh(f"{pip} -r requirements.txt", cwd=PROJ)
    ok = sh(f"{PY} -c \"import pycolmap, sys; sys.exit(0 if pycolmap.has_cuda else 1)\"", cwd=PROJ) == 0
    if not ok:
        sh(f"{PY} -m pip uninstall -y -q pycolmap; {pip} pycolmap-cuda12==4.2.0", cwd=PROJ)
    for extra in ("av", "huggingface_hub", "ultralytics", "PyNvVideoCodec"):
        sh(f"{pip} {extra}", cwd=PROJ)
    sh(f"{PY} -c \"import torch, pycolmap; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),"
       f" torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'pycolmap cuda', pycolmap.has_cuda)\"",
       cwd=PROJ, tee=OUT / "machine.txt")
    missing = [m for m in ("torch", "pycolmap", "click", "pytest", "cv2", "laspy", "rasterio", "trimesh")
               if sh(f"{PY} -c \"import {m}\"", cwd=PROJ) != 0]
    if missing:
        raise RuntimeError(f"not importable after setup: {missing}")
    return str(PY)


def _openmvs_runtime(bin_dir: Path) -> str:
    """OpenMVS 2.4.0's Ubuntu build needs glibc 2.38 / GLIBCXX_3.4.32 (Ubuntu 24.04); Kaggle's image
    is older, so every OpenMVS step exited 1 there. When a binary fails that way, Ubuntu 24.04's libc6,
    libstdc++6 and libgcc-s1 are unpacked beside it and each binary is replaced by a wrapper that runs
    the real one through the newer loader (the binaries link only glibc and libstdc++ dynamically)."""
    import re
    import urllib.request

    probe = bin_dir / "InterfaceCOLMAP"
    if not probe.exists():
        return "missing"
    out = subprocess.run([str(probe), "--help"], capture_output=True, text=True).stderr
    if "GLIBC" not in out and "GLIBCXX" not in out:
        return "native"
    root = bin_dir.parent / "noble-runtime"
    root.mkdir(exist_ok=True)
    mirror = "http://archive.ubuntu.com/ubuntu/pool/main/g"
    wanted = [("glibc", r"libc6_2\.39-0ubuntu8(?:\.\d+)*_amd64\.deb"),
              ("gcc-14", r"libstdc\+\+6_14\.2\.0-4ubuntu2~24\.04(?:\.\d+)*_amd64\.deb"),
              ("gcc-14", r"libgcc-s1_14\.2\.0-4ubuntu2~24\.04(?:\.\d+)*_amd64\.deb")]
    for folder, pattern in wanted:
        listing = urllib.request.urlopen(f"{mirror}/{folder}/", timeout=60).read().decode("utf-8", "replace")
        names = sorted(set(re.findall(pattern, listing.replace("%2B", "+").replace("%7E", "~"))))
        if not names:
            raise RuntimeError(f"no {pattern} in {mirror}/{folder}/")
        deb = names[-1]
        url = f"{mirror}/{folder}/{deb.replace('+', '%2B').replace('~', '%7E')}"
        if sh(f"curl -fsSL '{url}' -o /tmp/{deb} && dpkg-deb -x /tmp/{deb} {root} && rm /tmp/{deb}") != 0:
            raise RuntimeError(f"could not unpack {deb}")
    loader = next(iter(sorted(root.rglob("ld-linux-x86-64.so.2"))), None)
    if loader is None:
        raise RuntimeError("no ld-linux-x86-64.so.2 in the unpacked runtime")
    libs = ":".join(sorted({str(p.parent) for p in root.rglob("libc.so.6")} | {str(p.parent) for p in root.rglob("libstdc++.so.6")}))
    real = bin_dir.parent / "real"
    real.mkdir(exist_ok=True)
    for exe in sorted(bin_dir.iterdir()):
        if not exe.is_file() or exe.suffix in (".so", ".txt", ".md"):
            continue
        target = real / exe.name
        shutil.move(str(exe), target)
        exe.write_text(f'#!/bin/sh\nexec "{loader}" --library-path "{libs}" "{target}" "$@"\n')
        exe.chmod(0o755)
    check = subprocess.run([str(probe), "--help"], capture_output=True, text=True)
    status = "wrapped (Ubuntu 24.04 runtime)" if "GLIBC" not in check.stderr else "wrapped but still failing"
    log(f"   OpenMVS: {status}; loader {loader}")
    return status


@step("tools")
def _tools():
    tools = PROJ / "tools"
    tools.mkdir(exist_ok=True)
    sh(f"curl -fsSL {BLENDER_URL} -o /tmp/blender.tar.xz && tar xf /tmp/blender.tar.xz -C {tools} && rm /tmp/blender.tar.xz")
    sh(f"rm -rf /tmp/openmvs && mkdir -p /tmp/openmvs {tools}/openmvs/bin && curl -fsSL {OPENMVS_URL} -o /tmp/openmvs/o.zip"
       f" && cd /tmp/openmvs && python3 -m zipfile -e o.zip . && "
       f"cp \"$(dirname \"$(find /tmp/openmvs -type f -name TextureMesh | head -1)\")\"/* {tools}/openmvs/bin/ && "
       f"chmod +x {tools}/openmvs/bin/*")
    openmvs = _openmvs_runtime(tools / "openmvs" / "bin")
    for rel, url in REPOS.items():
        if not (PROJ / rel / ".git").exists():
            sh(f"git clone -q --depth 1 {url} {PROJ / rel}")
    return {"tools": sorted(p.name for p in tools.iterdir()), "openmvs": openmvs}


def _hf_token() -> tuple[str | None, str]:
    """The Hugging Face token and where it came from; the token itself is never logged.

    Kaggle Secrets reach runs started in the editor, not versions pushed through the API (the
    attached HF_TOKEN gave ConnectionError on 2026-09-27), so the private dataset that
    ``kaggle_ctl.py hf-token`` publishes (``hf_token.txt``) is tried second."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token.strip(), "Kaggle secret HF_TOKEN"
    except Exception as exc:  # noqa: BLE001
        log(f"   Kaggle secret HF_TOKEN not available to this run ({type(exc).__name__})")
    for path in sorted(INPUT.rglob("hf_token.txt")):
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token, f"private dataset file {path.parent.name}/hf_token.txt"
    return None, "none"


@step("weights")
def _weights():
    token, source = _hf_token()
    if token is None:
        log("   !! NO HF TOKEN: Track B would fall back to VGGT-1B. Laptop: kaggle_ctl.py hf-token")
        return "no token"
    os.environ["HF_TOKEN"] = token
    log(f"   HF token detected from {source} ({len(token)} characters, not printed)")
    probe = (f"{PY} -c \"import os; from huggingface_hub import hf_hub_download as d; "
             f"p = d('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt'); "
             f"print('VGGT_OMEGA_CHECKPOINT', p, round(os.path.getsize(p) / 1e9, 2), 'GB')\"")
    rc = sh(probe, cwd=PROJ, tee=OUT / "weights.log")
    text = (OUT / "weights.log").read_text(encoding="utf-8", errors="replace") if (OUT / "weights.log").exists() else ""
    if rc != 0 or "VGGT_OMEGA_CHECKPOINT" not in text:
        log("   !! VGGT-Omega checkpoint download FAILED with this token (account approved for facebook/VGGT-Omega?)")
        return "download failed"
    line = next(ln for ln in text.splitlines() if ln.startswith("VGGT_OMEGA_CHECKPOINT"))
    log(f"   ==> VGGT-OMEGA READY: {line.split(' ', 1)[1]} -> Track B runs VGGT-Omega")
    return "ok"


def env() -> dict:
    e = dict(os.environ)
    e.update({"PY": str(PY), "PATH": f"{PY.parent}:{e.get('PATH', '')}", "PYTHONUNBUFFERED": "1"})
    e.update({k: str(v) for k, v in (JOB.get("env") or {}).items()})
    return e


if JOB.get("tests"):
    @step("tests")
    def _tests():
        rc = sh(f"{PY} -m pytest tests -q -p no:cacheprovider", cwd=PROJ, env=env(), tee=OUT / "tests.log")
        return {"exit": rc}

weights_ok = (RESULT["steps"].get("weights") or {}).get("value") == "ok"
if JOB.get("require_hf") and not weights_ok:
    log("   !! require_hf: the VGGT-Omega checkpoint is not available, so the command is skipped "
        "(a long run would silently use VGGT-1B)")
elif JOB.get("cmd"):
    @step("cmd")
    def _cmd():
        rc = sh(JOB["cmd"], cwd=PROJ, env=env(), tee=OUT / "cmd.log")
        return {"exit": rc}


@step("collect")
def _collect():
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    for f in list(PROJ.glob("*.log")) + list(PROJ.glob("*.json")):
        shutil.copy2(f, logs / f.name)
    for bench in (PROJ / "data/interim").glob("bench_*"):
        for f in list(bench.glob("*.md")) + list(bench.glob("*.json")):
            (logs / bench.name).mkdir(exist_ok=True)
            shutil.copy2(f, logs / bench.name / f.name)
    before = set(Path.home().glob("box_handover_*.tar.gz"))
    sh(f"bash scripts/box_collect.sh {' '.join(JOB.get('collect') or [])}", cwd=PROJ, env=env())
    moved = []
    for tar in set(Path.home().glob("box_handover_*.tar.gz")) - before:
        shutil.move(str(tar), OUT / tar.name)
        moved.append(tar.name)
    return moved


log(f"done in {(time.time() - T0) / 60:.1f} min: "
    + ", ".join(f"{k} {'ok' if v['ok'] else 'FAILED'}" for k, v in RESULT["steps"].items()))
sys.exit(0)
