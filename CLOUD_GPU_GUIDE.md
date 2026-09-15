# Cloud GPU guide — PS-17 Single-Pass

For whoever picks up Stages 2–6 on a rented GPU machine. Read `DEVLOG.md` first (as always);
this file only covers the machine.

---

## 1. What actually needs the GPU

| Work | Needs GPU? | Why |
|------|-----------|-----|
| Stage 1 ingest, Stage Lab UI, most of the test suite | No | Runs on a laptop today (145+ tests, CPU) |
| **S1-3 / S1-4** — hardware video decode and the §4.3 speed target | **Yes** | Decode is the bottleneck; pip OpenCV never engages NVDEC |
| **Track A** — OpenDroneMap (§7.1) | **Yes** (GPU image) | `opendronemap/odm:gpu`, runs in Docker |
| **Track B** — VGGT-Ω feed-forward (§7.2) | **Yes** | ~6 GB for 1 frame, 20.8 GB for 200 frames |
| Refinement BA, MVS, TSDF fusion (§7.3, §9) | **Yes** | GPU OpenMVS / TSDF per §9 budget |
| Dynamic-object masking, S2-4 (§5.5) | Recommended | YOLOv8-seg via `ultralytics` |

Rule of thumb: write and unit-test on your laptop, measure on the GPU box. Don't pay for a GPU while
you're writing a parser.

---

## 2. Picking a machine

**GPU memory is what decides this.** From the spec's VGGT-Ω table (§7.2):

| Card | VRAM | Safe Track B chunk | Notes |
|------|------|--------------------|-------|
| RTX 4090 / RTX A5000 | 24 GB | 128 frames, 16 overlap (spec default); ceiling ~180 | The spec's reference card |
| RTX A6000 / L40S | 48 GB | ~300+ frames | Comfortable; Track A + B can run together |
| A100 / H100 80 GB | 80 GB | 500 frames fits (43.2 GB) | Only if the budget allows |

Also check:

- **OS:** Ubuntu 22.04 (ships Python 3.10, which the repo targets). **CUDA 12.1+** driver.
- **Disk:** ≥ 100 GB. The ODM Docker image is large, and ODM intermediates for a 10-minute clip add
  up fast.
- **CPU:** 8+ cores. ODM's feature extraction and matching are CPU-heavy even on the GPU image.
- **Docker with GPU access.** Track A needs it. **Many "pod"/container-style GPU rentals do not let
  you run Docker inside them.** Before you pay for hours, the first thing to run is the Docker check
  in §3. If it fails, pick a full VM instance instead.

Providers the team can compare: RunPod, Vast.ai, Lambda, or AWS / GCP / Azure GPU VMs. Prices change
often, so check current rates, and **set a spending alert on day one**.

---

## 3. First hour on a new box

Linux paths from here on: `.venv/bin/python`, not the Windows `.venv\Scripts\python` used in
`CLAUDE.md`.

```bash
# 1. GPU and driver are visible
nvidia-smi

# 2. Docker can reach the GPU (Track A depends on this — if it fails, change instance type)
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# 3. Code
git clone <repo-url> single-pass && cd single-pass

# 4. Python env
sudo apt-get update && sudo apt-get install -y python3.10-venv ffmpeg tmux
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# 5. GPU extras that are commented out in requirements.txt
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install av ultralytics

# 6. Verify
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -c "import av; print('pyav', av.__version__)"
.venv/bin/python -m pytest tests -q          # must be green before you change anything
```

Then set the card's real memory in the config, either in `configs/default.yaml` or per run with `--set`:

```bash
.venv/bin/python -m src.cli config --key device
.venv/bin/python -m src.cli run <video> --set device.gpu_memory_gb=48
```

Track B chunk sizing (`src/core/device.py`) reads `device.gpu_memory_gb` and
`device.gpu_headroom_gb`. When you profile the real card, **replace the numbers in the memory table
and keep the method**.

---

## 4. Getting footage onto the box

`data/` is git-ignored, so videos never go through git. Copy them directly:

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

Never expose Streamlit on a public port. Bind it to localhost and tunnel over SSH:

```bash
# on the GPU box, inside tmux so it survives disconnects
tmux new -s lab
.venv/bin/python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501

# on your laptop
ssh -N -L 8501:127.0.0.1:8501 user@<gpu-host>
# then open http://localhost:8501
```

Run every long job (full pipeline, ODM, benchmarks) inside `tmux`. An SSH drop kills a plain
foreground process and wastes paid GPU time.

---

## 6. What to close first on the GPU box, in order

These are open issues from `DEVLOG.md` §5 that can only be settled on real hardware.

1. **Licensing (spec §3.3), day 1.** Check VGGT-Ω weight access on Hugging Face. Read the actual
   `LICENSE`, and record the outcome in `README.md`. Log in with a token in an environment variable
   (`export HF_TOKEN=...` or `huggingface-cli login`). **Never commit tokens or weights**; `weights/`
   and `*.pt` / `*.pth` are already git-ignored.
2. **S1-3 — hardware decode.** `src/ingest/video_reader.py` asks OpenCV for
   `CAP_PROP_HW_ACCELERATION`. The pip `opencv-python` wheel has no NVDEC, so this always falls back
   (you'll see `hardware decode unavailable -> falling back to software decode` in the run log).
   Options: decode through PyAV/FFmpeg with NVDEC (`ffmpeg -hwaccel cuda`), or an OpenCV build with
   CUDA video. Measure before and after; don't assume.
3. **S1-4 — §4.3 speed target (< 60 s for a 10-min 4K clip).** Re-measure on real 4K footage once
   decode is fixed. The Stage 1 scorecard's "Projected selection time" KPI is an extrapolation; the
   GPU box gives you the real number.
4. **S1-2 — I-frame preference.** Now testable, since PyAV is installed.
5. **Track A (§7.1).** Get `opendronemap/odm:gpu` producing a textured mesh on a sample dataset, then
   **tag that commit** (spec §10 day-3 freeze). Check the ODM docs for current flags. The basic shape
   is `docker run --rm --gpus all -v <datasets>:/datasets opendronemap/odm:gpu --project-path /datasets <project>`.
6. **S2-4 — semantic masking.** Now testable, since `ultralytics` is installed.

---

## 7. Money and data hygiene

- **Stop the instance when you stop working.** An idle GPU bills the same as a busy one.
- Keep weights, datasets and the venv on a **persistent volume**, if the provider separates it from
  the instance disk, so a stopped or recreated instance doesn't mean re-downloading everything.
- Stage Lab runs go to `data/lab/` and pipeline runs go to `data/interim/`. Both are git-ignored. If
  a number matters, write it into your DEVLOG entry; the run folder won't survive the instance.
- Copy results back before destroying a machine:
  `rsync -avP user@<gpu-host>:~/single-pass/data/outputs/ data/outputs/`

---

## 8. Before you hand work back

- [ ] `.venv/bin/python -m pytest tests -q` is green (xfails are tracked open issues, not failures)
- [ ] `tests/test_ui_smoke.py` passes: every Stage Lab panel still loads
- [ ] DEVLOG session entry written. On GPU sessions, also record the **GPU model, VRAM, driver
      version, CUDA version** and the measured timings, because numbers without the hardware are
      not comparable
- [ ] Status board in `DEVLOG.md` §2 matches `src/stages.py`
- [ ] Any new tunable lives in `configs/default.yaml`, not in code
