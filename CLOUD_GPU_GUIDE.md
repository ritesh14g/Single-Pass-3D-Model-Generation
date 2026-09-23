# Cloud GPU guide — PS-17 Single-Pass

For whoever picks up Stages 2–6 on the GPU machine. Read `DEVLOG.md` first (as always);
this file only covers the machine.

---

## 0. The team's machine (institute-provided)

| Resource | What we get | Spec reference point |
|----------|-------------|----------------------|
| GPU | **20 GB MIG slice of an NVIDIA H100** | 24 GB card (RTX 4090 / A5000), spec §7.2 |
| CPU | **3 cores** of Intel Xeon Platinum 8480+ | 8+ cores recommended (§2 below) |
| RAM | **56 GB** | Pipeline streams frames; no hard minimum |
| Environment | **Notebook profile**, "PyTorch Environment with CUDA" | Full Linux VM with Docker, spec §3 |

### Is it enough?

**Yes for development, and for Stages 1–2 and Track B. Tight for the 15-minute target. At risk for
Track A.** In detail:

- **GPU memory (20 GB): enough for Track B at the spec's default chunk size.** With the 4 GB headroom
  from `device.gpu_headroom_gb`, the budget is 16 GB. From the VGGT-Ω table that is about
  **128–135 frames per chunk**, and the spec's default of 128 frames with 16 overlap fits. This is
  pinned by `tests/test_core.py::test_institute_mig_slice_fits_the_default_chunk`. The consequences:
  - There is less room than on the reference 24 GB card (~180 frames). Do **not** raise
    `chunk_frames` above 128 on this box.
  - A MIG slice can report slightly under 20 GB. At about 19.5 GB detected, the safe chunk lands
    right at 128. If Track B hits out-of-memory errors, lower the headroom to 3 GB only when nothing
    else is using the GPU at the same time.
  - **Run Track A and Track B one after the other, not side by side.** There is no room for both.
- **GPU compute: likely fine, but measure it.** A 20 GB H100 slice has only a fraction of the
  card's streaming multiprocessors. Tensor-core throughput should still be in the reference card's
  class for FP16 inference, but this hasn't been measured yet. FP16 is on by default
  (`device.half_precision`).
- **CPU (3 cores): this is the real bottleneck.** The spec's budget assumes a workstation.
  - Software 4K decode on 3 cores cannot meet the 1-minute ingest budget (S1-4 projected ~740 s on
    the dev laptop). **NVDEC decode is required, not optional**, and the code now tries it first.
  - The CPU parts of Stage 2 (bilateral deblocking, non-local-means low-light denoise, exposure
    fitting) and the frame-selection analysis all share those 3 cores.
  - ODM's feature extraction and matching are CPU-heavy even in the GPU build, so expect Track A
    to overrun its budget here.
  - Numbers from this box are the honest ones to report. Record them as they are.
- **RAM (56 GB): enough.** Everything streams, and nothing loads a whole video into memory.
- **Notebook profile: the biggest unknown.** Notebook containers usually have **no Docker, no sudo,
  and idle timeouts**. Track A in spec §7.1 is `opendronemap/odm:gpu`, which runs in Docker. Check
  on day 1 (§3 step 2 below). If Docker, Podman and Apptainer are all unavailable, ask the
  institute for a VM profile, or run Track A elsewhere (a teammate's machine, or a short rented VM)
  and treat this box as the Track B and Stages 1–2 machine.
- **MIG and decoders:** each H100 MIG slice gets its own share of the card's hardware video
  decoders, but how many depends on the slice profile. `nvidia-smi -L` shows the profile. The NVDEC
  path proves itself when the reader opens, and falls back to CPU decode if the slice has no
  decoder engine.

### GPU first, CPU fallback — what the code does

Every GPU user asks `src/core/device.py::resolve_device(device.prefer)` and gets `cuda` or `cpu`.
When it wanted the GPU and didn't get it, it logs a downgrade. It never crashes.

| Where | GPU path | Fallback |
|-------|----------|----------|
| Stage 1 decode (`src/ingest/video_reader.py`) | NVDEC through **PyNvVideoCodec** (`ingest.video.nvdec`) | OpenCV hardware flag, then OpenCV software decode. An NVDEC fault **mid-stream** re-reads that frame through OpenCV, so there are no gaps |
| Stage 2 dynamic masking (`src/condition/dynamic_mask.py`) | YOLOv8-seg on `cuda:0`, FP16 | On a GPU error such as out-of-memory, the frame is retried on the CPU and later frames stay there. If the model is missing, it uses geometric masking only |
| Thread pools (`device.cpu_threads: auto`) | — | OpenCV and torch threads are capped to the container's **cgroup CPU quota**. Notebook containers often see every host core, and would run 100+ threads on 3 cores |
| Track B chunking | Reads the detected GPU memory | Uses `device.gpu_memory_gb` (now **20**) when no GPU is visible |
| Stage 5 export (`src/export/`) | — (CPU) | FBX needs headless Blender in `tools/blender-4.2.3-linux-x64/` (portable tarball, no install); without it FBX is skipped with the reason. EGM96 heights need PROJ's network grid fetch (cdn.proj.org) once; without it heights stay as given and a downgrade is logged |
| Track A (`src/recon/track_a_colmap.py`) | pycolmap-cuda12: SIFT, matching, PatchMatch dense on CUDA | pycolmap on CPU; dense via OpenMVS `DensifyPointCloud` (CPU); OpenMVS mesh → Poisson; texture → untextured mesh. Needs `pip install pycolmap-cuda12==4.2.0` and OpenMVS 2.4.0 in `tools/openmvs/bin` on the box |
| Stage 3 Zone 2 fill (`src/fusion/`) | Reuses Track B's saved VGGT depth (`track_a/track_b_depth/`), else runs the Track B predictor per frame on CUDA | A GPU error moves the predictor to the CPU (`fusion.mono_depth.cpu_max_frames` frames, logged). No model or weights: the gaps are reported and not filled. Zone classification itself is CPU (~13 s on DJI_0047) |

The run manifest records `device`, `gpu_name`, `gpu_memory_gb`, `mig`, `cpu_threads` and the
`decoder` that was actually used. Check them after every run.

**Rule for new code (Stages 3–6):** get the device from `resolve_device`, and never hard-code
`"cuda"`. Wrap GPU work so a failure logs a downgrade and continues on the CPU. Put any new knob in
`configs/default.yaml`.

---

## 1. What actually needs the GPU

| Work | Needs GPU? | Why |
|------|-----------|-----|
| Stage 1 ingest, Stage Lab UI, most of the test suite | No (but decode wants NVDEC) | Runs on a laptop today (320+ tests, CPU) |
| **S1-3 / S1-4**: hardware video decode and the §4.3 speed target | **Yes** | Decode is the bottleneck. pip OpenCV never engages NVDEC, so the code uses PyNvVideoCodec |
| **Track A**: OpenDroneMap (§7.1) | **Yes** (GPU image) | `opendronemap/odm:gpu` runs in Docker, which the notebook may not have (§0) |
| **Track B**: VGGT-Ω feed-forward (§7.2) | **Yes** | ~6 GB for 1 frame, 20.8 GB for 200 frames |
| Refinement BA, MVS (§7.3, §9) | **Yes** | GPU PatchMatch / hybrid depth per the §9 budget |
| Stage 3 Zone 2 fill (§6.3) | **Yes** for the fill, no for zones and gaps | Monocular depth comes from VGGT: saved by Track B on hybrid runs, or predicted. The laptop has no VGGT-Ω weights (gated), so there the fill is skipped and logged |
| Dynamic-object masking, S2-4 (§5.5) | Recommended | YOLOv8-seg via `ultralytics`, which runs on the GPU when one is visible |

Rule of thumb: write and unit-test on your laptop, measure on the GPU box.

---

## 2. If you ever rent a machine instead

**GPU memory is what decides this.** From the spec's VGGT-Ω table (§7.2):

| Card | VRAM | Safe Track B chunk | Notes |
|------|------|--------------------|-------|
| **H100 MIG slice (institute)** | **20 GB** | **128 frames (default), ceiling ~135** | Our box. No concurrent Track A + B |
| RTX 4090 / RTX A5000 | 24 GB | 128 frames, 16 overlap (spec default); ceiling ~180 | The spec's reference card |
| RTX A6000 / L40S | 48 GB | ~300+ frames | Comfortable; Track A + B can run together |
| A100 / H100 80 GB | 80 GB | 500 frames fits (43.2 GB) | Only if the budget allows |

Also check:

- **OS:** Ubuntu 22.04 (ships Python 3.10, which the repo targets). **CUDA 12.1+** driver.
- **Disk:** ≥ 100 GB. The ODM Docker image is large, and ODM intermediates for a 10-minute clip add
  up fast.
- **CPU:** 8+ cores. ODM's feature extraction and matching are CPU-heavy even on the GPU image.
- **Docker with GPU access.** Track A needs it, and many pod or notebook-style GPU offerings don't
  allow it. Run the Docker check in §3 before relying on a machine.

---

## 3. First hour on the box

Linux paths from here on: `.venv/bin/python`, not the Windows `.venv\Scripts\python` used in
`CLAUDE.md`. In the notebook profile, run these from a **Jupyter terminal**, not from notebook cells.

```bash
# 1. GPU, driver and MIG slice are visible
nvidia-smi
nvidia-smi -L                 # shows the MIG profile, e.g. "MIG 2g.20gb"
nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null   # expect a 3-core quota (300000 100000)

# 2. Container tooling. Track A depends on one of these; if all fail, see §0
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi \
  || which podman apptainer singularity

# 3. Code
git clone <repo-url> single-pass && cd single-pass

# 4. Python env. --system-site-packages reuses the notebook's CUDA PyTorch instead of
#    downloading a second 2 GB copy. No sudo is needed.
python3 -m venv .venv --system-site-packages
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# 5. GPU extras (optional in requirements.txt; each has a logged fallback)
.venv/bin/python -m pip install PyNvVideoCodec av ultralytics
#    Only if the profile's torch has no CUDA:
#    .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Verify
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -c "from src.core.device import device_info, cpu_thread_budget; print(device_info(), 'threads', cpu_thread_budget())"
.venv/bin/python -m src.cli inspect data/raw/<clip>.mp4   # "hardware decode: yes (nvdec)" is the goal
.venv/bin/python -m pytest tests -q          # must be green before you change anything
```

If `inspect` shows `(opencv)` instead of `(nvdec)`, the run log has a `NVDEC decode -> OpenCV decode`
line with the reason. The `NvdecCapture` adapter was written against the PyNvVideoCodec 2.x
`SimpleDecoder` API **without GPU hardware to test on**. If the reason is an API mismatch, fix the
adapter; the fallback keeps runs working in the meantime.

`device.gpu_memory_gb` is already **20** in `configs/default.yaml`, and a visible card's real memory
overrides it anyway. Track B chunk sizing (`src/core/device.py`) reads `device.gpu_memory_gb` and
`device.gpu_headroom_gb`. When you profile the real card, **replace the numbers in the memory table
and keep the method**.

---

## 4. Getting footage onto the box

`data/` is git-ignored, so videos never go through git.

- **Notebook profile:** use the Jupyter file browser's upload for small files. For large clips,
  use `scp`/`rsync` if the institute gives SSH access, or fetch them from shared storage (Drive,
  OneDrive or an institute share) with `wget`/`curl` inside the Jupyter terminal.
- **With SSH:**

```bash
# from your laptop
rsync -avP "data/lab/uploads/LineVision-VideoGeoTagging.mp4" \
           "data/lab/uploads/DJIFlightRecord_2019-01-09_[13-16-53].txt" \
           user@<gpu-host>:~/single-pass/data/raw/
```

On Windows without rsync, use `scp` with the same paths. Keep the quotes, because the flight-record
name has square brackets.

**Gotcha for the LineVision clip:** it was trimmed after recording, so the flight log can't be matched
automatically. Always pass the offset, or the run has no GPS:

```bash
.venv/bin/python -m src.cli run data/raw/LineVision-VideoGeoTagging.mp4 \
  --csv "data/raw/DJIFlightRecord_2019-01-09_[13-16-53].txt" \
  --set ingest.telemetry.flight_record.offset_s=438.3
```

---

## 5. Using the Stage Lab UI remotely

Never expose Streamlit on a public port. Bind it to localhost:

```bash
# on the GPU box, in the background so it survives a closed tab (the box has no tmux)
nohup .venv/bin/python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501 > lab.log 2>&1 &
```

- **Notebook profile:** if the hub has `jupyter-server-proxy`, open
  `<your-notebook-url>/proxy/8501/`. If pages load blank, add
  `--server.baseUrlPath proxy/8501 --server.enableCORS false --server.enableXsrfProtection false`.
  If there's no proxy, run the pipeline on the box with the CLI and view results locally after
  copying the run folder back.
- **With SSH:** `ssh -N -L 8501:127.0.0.1:8501 user@<gpu-host>`, then open http://localhost:8501.

**The box has no `tmux`.** Run every long job (setup, full pipeline, benchmarks) in the background:
`nohup <command> > job.log 2>&1 &`, then watch it with `tail -f job.log` (Ctrl+C stops the watching,
not the job; `ps aux | grep python` shows it is still running; `kill <pid>` stops it). A foreground
process dies with a closed browser tab or an idle-culled kernel.

---

## 6. What to close first on the GPU box, in order

These are open issues from `DEVLOG.md` §5 that can only be settled on real hardware.

1. **Licensing (spec §3.3), day 1.** Check VGGT-Ω weight access on Hugging Face. Read the actual
   `LICENSE`, and record the outcome in `README.md`. Log in with a token in an environment variable
   (`export HF_TOKEN=...` or `huggingface-cli login`). **Never commit tokens or weights**; `weights/`
   and `*.pt` / `*.pth` are already git-ignored.
2. **Docker / container check (§3 step 2).** This decides where Track A runs. Record the answer in
   DEVLOG.
3. **S1-3: hardware decode.** Install `PyNvVideoCodec` and confirm `inspect` reports `nvdec`. Measure
   decode frames/s against `--set ingest.video.nvdec=false` on the same clip. Don't assume the speedup.
4. **S1-4: §4.3 speed target (< 60 s for a 10-min 4K clip).** Re-measure on real 4K footage once
   decode is on NVDEC. On 3 cores, the frame-selection analysis may become the bottleneck next.
5. **S1-2: I-frame preference.** Now testable, since PyAV is installed.
6. **S2-4: semantic masking.** Now testable. Confirm the run log says `device=cuda half=True`, and
   record ms/frame.
7. **Track A (§7.1).** Get `opendronemap/odm:gpu` producing a textured mesh on a sample dataset, then
   **tag that commit** (spec §10 day-3 freeze). Check the ODM docs for current flags. The basic shape
   is `docker run --rm --gpus all -v <datasets>:/datasets opendronemap/odm:gpu --project-path /datasets <project>`.

---

## 7. Data hygiene

- Notebook sessions get culled when idle. Keep the repo, venv, weights and datasets under the
  **persistent home directory**, not `/tmp`, so a restarted session doesn't mean re-downloading.
- It's a shared institute GPU, so shut the kernel down when you're done. Out-of-memory errors can
  come from your own leftover notebook kernels holding GPU memory; check `nvidia-smi`.
- Stage Lab runs go to `data/lab/` and pipeline runs go to `data/interim/`. Both are git-ignored. If
  a number matters, write it into your DEVLOG entry; the run folder may not survive.
- Copy results back before your storage is cleared:
  `rsync -avP user@<gpu-host>:~/single-pass/data/outputs/ data/outputs/`, or zip them and download
  through Jupyter.

---

## 8. Before you hand work back

- [ ] `.venv/bin/python -m pytest tests -q` is green (xfails are tracked open issues, not failures)
- [ ] `tests/test_ui_smoke.py` passes: every Stage Lab panel still loads
- [ ] DEVLOG session entry written. On GPU sessions, also record the **GPU model and MIG profile,
      VRAM, driver version, CUDA version, CPU quota** and the measured timings, because numbers
      without the hardware are not comparable
- [ ] Status board in `DEVLOG.md` §2 matches `src/stages.py`
- [ ] Any new tunable lives in `configs/default.yaml`, not in code
- [ ] Any new GPU code follows the §0 rule: `resolve_device`, logged downgrade, CPU fallback

---

## 9. Giving the box back (it goes to someone else)

Two scripts, in this order. The first keeps the work; the second keeps your accounts safe.

```bash
cd ~/single-pass && git pull                 # make sure the box is on the latest code
git status --porcelain                       # anything listed here was written on the box
git add -A && git commit -m "box: <what>" && git push    # push it, if it is work worth keeping

bash scripts/box_collect.sh esri_full2       # package results -> ~/box_handover_<stamp>.tar.gz
```

`box_collect.sh` takes the things that cannot be recreated off a machine you are losing:
a patch of anything uncommitted, every run's manifest, input check, scorecards, georeferencing,
logs and telemetry tables, each run's **sparse reconstruction** (so Stage 5 can be re-run on a
laptop), and a record of **what the machine was** (GPU, CUDA, packages, tool versions) — numbers
are not comparable without it. Naming a run adds that run's `export/`: OBJ, PLY, LAS, GeoTIFF,
glb and FBX, about 400 MB on Esri, which is what Stage 6's viewer is built against offline.

It deliberately skips what a fresh box re-downloads: OpenMVS binaries, portable Blender, the
VGGT-Ω weights, and the raw footage.

Download the tarball from the Jupyter file browser (right-click → Download), unpack it, and
check it opens before moving on.

```bash
bash scripts/box_wipe_credentials.sh         # reports what it finds, changes nothing
bash scripts/box_wipe_credentials.sh --yes   # removes it
```

It looks for the Hugging Face token, saved git credentials and tokens embedded in a remote URL,
SSH private keys, secrets in shell profiles, and shell/notebook history — and offers to delete
the working copy with its footage and reconstructions. It never prints a secret's value.

**Treat every token that was on that machine as exposed and rotate it**, even after wiping: you
cannot prove what the next user restores from a snapshot.

---

## 10. Getting the box back: the upload bundle

The box comes back empty (§9 wiped it). Instead of cloning with a GitHub token on a shared machine,
upload **one zip** that holds everything the box needs. Two scripts do it:

| Script | Runs on | Does |
|--------|---------|------|
| `scripts/box_pack.py` | laptop | Zips the code **as it is in your working tree** (uncommitted work included), the clips, optionally the last handover's evidence, and `BUNDLE.json` (git state, sha256 per file). Stops if any code file looks like a token or private key |
| `scripts/box_gdrive.py` | laptop | **Recommended.** After you upload the zip to Google Drive: checksums it and writes `box_fetch_paste.txt`, one block to paste into the box terminal |
| `scripts/box_fetch_gdrive.sh` | box | gdown download (resumes, retries), catches Drive's web-page / quota answers, checks the sha256, then runs `box_restore.sh` from the zip |
| `scripts/box_upload.py` | laptop | Sends the zip to the box's Jupyter server in resumable, checksummed parts (the browser upload fails at this size) |
| `scripts/box_restore.sh` | box | Unpacks, verifies every checksum, checks the machine, builds `.venv`, fetches Blender / OpenMVS / VGGT code, gets the VGGT-Ω checkpoint (with your token), runs the tests, and prints the next commands. Safe to re-run; `--unpack-only` just refreshes the code |

What is **not** in the bundle, and why: model weights (gated, re-downloaded with a fresh token),
OpenMVS and Blender binaries (Linux builds, fetched by the restore script), the Python environment,
and any credential.

### 10.1 On the laptop

```powershell
# code + Esri (475 MB) + DJI_0047 (video + telemetry.csv, 778 MB): about 1.25 GB
.venv\Scripts\python scriptsox_pack.py
# also the last handover's evidence (manifests, georef, sparse models; no exports), ~40 MB
.venv\Scripts\python scriptsox_pack.py --with-box-runs
# other footage: files or folders (a folder keeps its telemetry next to the video)
.venv\Scripts\python scriptsox_pack.py --clip "..\Drone Video Dataset\QGISFMV_Samples\MISB\<clip>.ts"
```

The zip lands in `dataox_upload\` (git-ignored). The script prints the exact box commands. Code only,
for a quick refresh: `--no-default-clips` (about 2 MB).

### 10.2 Sending it to the box: Google Drive + gdown (recommended)

The Jupyter browser upload fails on a 1.3 GB file. Google Drive's browser upload does not (it resumes
by itself), and the box pulls the file down with `gdown`. You paste one block into the box terminal.

```powershell
# 1. Laptop: pack (if not done) and checksum
.venv\Scripts\python scripts\box_pack.py --with-box-runs
.venv\Scripts\python scripts\box_gdrive.py
# 2. Browser: drive.google.com -> New -> File upload -> data\box_upload\box_upload_<stamp>.zip
#    When it finishes: right-click -> Share -> General access: "Anyone with the link" (Viewer) -> Copy link
# 3. Laptop: turn the link into the box's commands
.venv\Scripts\python scripts\box_gdrive.py --link "https://drive.google.com/file/d/<id>/view?usp=sharing"
#    -> data\box_upload\box_fetch_paste.txt
```

4. Open `box_fetch_paste.txt`, put your fresh Hugging Face token on the `export HF_TOKEN=` line (or
   delete that line), then copy all of it.
5. On the box, open a Jupyter terminal and paste. The block writes `~/box_fetch_gdrive.sh` and starts
   it in the background (`nohup`, log in `~/box_setup.log`, shown with `tail -f`; Ctrl+C stops only the
   watching, and closing the tab does not stop the setup): install gdown, download to `~/box_upload.zip`, check the sha256,
   then the full `box_restore.sh` (§10.3).

If the download is interrupted, run `bash ~/box_fetch_gdrive.sh <id> <sha256>` again (the paste file's last
line); gdown continues the partial file. Messages you may see:

| Message | Meaning | Fix |
|---------|---------|-----|
| `Drive refused the file: Cannot retrieve the public link` | not shared, or Drive's daily download quota for the file | Share → "Anyone with the link"; for the quota, Drive → right-click → Make a copy, share the copy, `box_gdrive.py --link <copy link>` |
| `... is not a zip (first bytes: <!DOCTYPE html>` | Drive sent its web page instead of the file | same as above; delete `~/box_upload.zip`, run again |
| `checksum mismatch` | a different file on Drive than the one you packed, or a damaged download | `rm ~/box_upload.zip`, run again; check you shared the newest bundle |

Keep the Drive file private again (or delete it) once the box has it: the bundle holds the code and the
clips. Code refreshes later: `box_pack.py --no-default-clips` (about 2 MB), upload, share,
`box_gdrive.py --link ... --unpack-only`, paste.

**Alternative without Drive:** `scripts/box_upload.py` pushes the zip straight to the box's Jupyter server
in resumable 100 MB parts:
`.venv\Scripts\python scripts\box_upload.py --url "<your Jupyter tab address, with ?token=...>"`, then
`bash ~/box_restore.sh ~/box_upload_<stamp>.zip` on the box (it joins and checks the parts first).

### 10.3 Restoring on the box (Jupyter terminal)

```bash
export HF_TOKEN=hf_...            # fresh read-only token, this shell only; never write it to a file
cd ~ && nohup bash box_restore.sh ~/box_upload_<stamp>.zip > ~/box_setup.log 2>&1 &   # -> ~/single-pass
tail -f ~/box_setup.log           # Ctrl+C stops watching only; the restore keeps running
```

15–25 min the first time (pip, Blender, OpenMVS, the 4 GB checkpoint). Read the `!!` lines it prints:
each one names what is missing and what the pipeline does without it. Then record the machine (its
step 3 output) in your DEVLOG entry, as §8 asks. If it reports a bad part, re-run the laptop's upload
command, then this again.

Later code changes: `box_pack.py --no-default-clips` (about 2 MB), `box_upload.py` again, then
`bash ~/box_restore.sh ~/box_upload_<new>.zip --unpack-only`.

### 10.4 What to run on the box now (Stage 3 and the open box items)

In order. Each writes a run folder under `data/interim/`; `stage4_report.py` prints Stages 3, 4 and 5
together as JSON to paste back.

```bash
cd ~/single-pass && nohup bash scripts/box_stage3_runs.sh > box_runs.log 2>&1 &
tail -f box_runs.log       # Ctrl+C stops watching only
```

`scripts/box_stage3_runs.sh` runs, in order: (1) Esri, every built stage (hybrid VGGT-Ω, so Stage 3
reuses Track B's saved depth for the fill); (2) DJI_0047 (Stage 0 applies the camera FOV prior and the
measured telemetry offset); (3) `stage4_report.py` for both, saved to `box_runs_report.json`, which is
what to paste back.

What to look for in Stage 3 (`fusion/fusion_report.json`, or the Stage Lab's Stage 3 page):
`fill.frames_anchored` vs `frames_tried` and the refusal reasons, `fill.holdout_error_median_pct` (Zone 2
accuracy: pass ≤ 2% of depth), `coverage_pct` vs `measured_coverage_pct` (how much the fill closed),
`rejected_near_camera_pct` (26% on the laptop's DJI_0047 run), and `timings_s.fill` against the 120 s
fusion allotment. To tune Stage 3 without re-running Stage 4, use the Stage 3 page's
"re-run Stage 3 alone on an existing run".

Before handing the box back: `bash scripts/box_collect.sh esri_s3` (now also collects each run's
`fusion/`), download the tarball, then `bash scripts/box_wipe_credentials.sh --yes`, and rotate the token.

