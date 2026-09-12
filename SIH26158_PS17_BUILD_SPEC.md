# SIH26158 / PS-17 — Single-Pass Drone Video to Accurate 3D Model Generation System

**Build specification for Claude Code**
Version 1.0 · Target submission date: 28 September 2026

---

## 0. How to use this document

This is the authoritative build spec. Read Sections 1–3 before writing any code.
Sections 4–9 define the pipeline stage by stage; each stage lists its inputs, outputs,
acceptance test, and the module path it should live in. Section 10 is the day-by-day
build order. Section 11 defines what "done" means.

**Prime directive:** the classical path (Track A) must work end to end before any
neural component (Track B) is integrated. Track A is the submission floor. Track B is
the score ceiling. Never let Track B breakage block a demo.

---

## 1. Problem statement (verbatim requirements)

**Organisation:** National Technical Research Organisation
**Category:** Software · **Theme:** Drone / Robotics
**Title:** Single-Pass Drone Video to Accurate 3D Model Generation System

### 1.1 What must be built

An AI-enabled system that generates a **georeferenced and metrically accurate 3D model**
of a scene using **only a single-pass drone video stream** captured from a moving UAV.
The system processes video frames captured during **one flight path** and reconstructs:

1. 3D terrain and structures
2. Building facades and rooftops
3. Roads and infrastructure
4. Vegetation and obstacles
5. Textured 3D meshes or point clouds

The generated model must be suitable for **visualization, measurement, and analysis**.

### 1.2 Key challenges (all eight are explicitly scored — none may be ignored)

| # | Challenge | Handled in |
|---|-----------|-----------|
| i | Limited viewing angles due to single flight path | §6 (occlusion engine), §7 (Track B) |
| ii | Motion blur and video compression artifacts | §5.2, §5.3 |
| iii | Variable illumination and shadows | §5.4 |
| iv | Dynamic objects (vehicles, humans, animals) | §5.5 |
| v | GPS inaccuracies and sensor noise | §5.6, §8 |
| vi | Real-time or near-real-time processing requirements | §9 (performance budget) |
| vii | Reconstruction of occluded surfaces | §6 (core innovation) |
| viii | Metric accuracy without extensive GCPs | §8 (GPS/telemetry anchoring) |

### 1.3 Input data

**Mandatory**
- Drone video (1080p / 4K)
- GPS coordinates
- Flight metadata

**Optional (use when present, degrade gracefully when absent)**
- IMU data
- Barometric altitude
- Camera intrinsic parameters
- RTK / PPK corrections

> Note: the official dataset will be **provided at competition time**. The system must
> therefore auto-detect which optional inputs exist and adapt, never hard-require them.

### 1.4 Desired output

| Parameter | Target |
|-----------|--------|
| Reconstruction type | 3D mesh / point cloud |
| Processing time | < 15 minutes for a 10-minute video |
| Spatial accuracy | ≤ 1 m |
| Coverage | Entire visible scene |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb / .gltf, .fbx |
| Visualization | Web-based or desktop viewer |

### 1.5 Evaluation criteria (drives every engineering tradeoff)

| Criteria | Weight | Design implication |
|----------|--------|--------------------|
| Reconstruction Accuracy | 30% | Refinement stage (§7.3) is non-negotiable |
| Model Completeness | 20% | Occlusion engine (§6) is the biggest single lever |
| Processing Speed | 20% | Hard 15-min budget, enforced in code (§9) |
| Innovation | 15% | Confidence-gated hybrid + honest gap reporting |
| Scalability | 10% | Chunked processing, tile-based export |
| User Interface | 5% | Web viewer with confidence overlay (§8.4) |

### 1.6 Potential applications (mention in UI/docs, drives disaster-response robustness focus)

Border and strategic area mapping · Disaster damage assessment · Urban planning and smart
cities · Infrastructure inspection · Construction progress monitoring · Archaeological
documentation · Digital twin generation · Military reconnaissance and mission planning

---

## 2. Architecture overview

### 2.1 Two-track design

```
                        ┌──────────────────────────────┐
   Video + telemetry ──▶│  STAGE 1: Ingest & Condition │──▶ clean frames + poses prior
                        └──────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │ TRACK A (baseline)    │           │ TRACK B (accelerator) │
        │ OpenSfM → OpenMVS     │           │ VGGT-Ω feed-forward   │
        │ via OpenDroneMap      │           │ pose + depth + conf.  │
        └───────────┬───────────┘           └───────────┬───────────┘
                    │                                   │
                    └─────────────┬─────────────────────┘
                                  ▼
                  ┌───────────────────────────────────┐
                  │ STAGE 3: Confidence-Gated Fusion  │
                  │  high conf → BA refinement        │
                  │  low conf  → mono-depth + TSDF    │
                  │  no obs    → flagged gap          │
                  └───────────────┬───────────────────┘
                                  ▼
                  ┌───────────────────────────────────┐
                  │ STAGE 4: Metric & Geo Anchoring   │
                  │  GPS/baro → scale + CRS + datum   │
                  └───────────────┬───────────────────┘
                                  ▼
                  ┌───────────────────────────────────┐
                  │ STAGE 5: Mesh, Texture, Export    │
                  │  OBJ PLY LAS GeoTIFF glTF FBX     │
                  └───────────────┬───────────────────┘
                                  ▼
                  ┌───────────────────────────────────┐
                  │ STAGE 6: Web Viewer + QA Report   │
                  └───────────────────────────────────┘
```

### 2.2 Why this shape (design rationale — keep for the judge pitch)

- **Feed-forward alone is not accurate enough.** VGGT-class models are fast but their raw
  output is incomplete and geometrically loose. Published comparisons put raw VGGT around
  88% on a hard SfM benchmark, while hybrid systems that pair feed-forward priors with
  depth-regularized bundle adjustment reach ~98%. Even the VGGT-Ω authors built their own
  training data by using VGGT to *initialize* camera parameters and then running COLMAP
  bundle adjustment on top.
- **Classical alone is too slow and too gap-prone for single-pass.** OpenSfM/OpenMVS needs
  many overlapping views per surface; a single strip leaves facades under-observed.
- **So: feed-forward for speed and coverage, classical for accuracy, confidence for the
  handoff between them.** On aerial photogrammetric blocks specifically, feed-forward
  models have shown completeness gains up to +50% over COLMAP at an order of magnitude
  less processing time — but with degrading pose reliability on very large image sets,
  which is exactly what the chunking strategy in §7.2 exists to fix.
- **Honest gaps beat silent hallucination.** Never let Poisson reconstruction quietly
  close a hole in a never-observed facade. Flag it. This is both scientifically correct
  and a differentiator in front of judges.

### 2.3 Repository layout

```
sih-ps17/
├── README.md
├── requirements.txt
├── docker/
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── configs/
│   ├── default.yaml            # all tunables, no magic numbers in code
│   ├── fast.yaml               # demo-speed preset
│   └── accurate.yaml           # accuracy preset
├── src/
│   ├── ingest/
│   │   ├── video_reader.py
│   │   ├── telemetry.py        # SRT/CSV/EXIF parsing
│   │   └── frame_selector.py
│   ├── condition/
│   │   ├── blur.py
│   │   ├── artifacts.py
│   │   ├── illumination.py
│   │   └── dynamic_mask.py
│   ├── recon/
│   │   ├── track_a_odm.py
│   │   ├── track_b_vggt.py
│   │   ├── chunking.py
│   │   └── refine_ba.py
│   ├── fusion/
│   │   ├── confidence.py
│   │   ├── tsdf.py
│   │   ├── mono_depth.py
│   │   └── gap_detect.py
│   ├── geo/
│   │   ├── scale.py
│   │   ├── crs.py
│   │   └── georeference.py
│   ├── export/
│   │   ├── mesh.py
│   │   ├── pointcloud.py
│   │   ├── raster.py           # GeoTIFF / DSM / orthophoto
│   │   └── convert.py          # glTF / FBX
│   ├── qa/
│   │   ├── metrics.py
│   │   ├── degrade.py          # synthetic degradation harness
│   │   └── report.py
│   └── cli.py
├── viewer/                     # web viewer (three.js / potree)
├── tests/
└── data/
    ├── raw/
    ├── interim/
    └── outputs/
```

### 2.4 Global engineering rules

1. **Every stage writes to disk and is independently resumable.** A crash at stage 5 must
   not force re-running stage 2. Use a manifest JSON per run.
2. **Every stage emits timing to a structured log.** The 15-minute budget is tracked live.
3. **No hard-coded parameters.** Everything in `configs/default.yaml`.
4. **Graceful degradation is a feature, not error handling.** Missing IMU, missing baro,
   missing intrinsics, garbage GPS — each has a defined fallback, logged loudly.
5. **Every geometric output carries a per-point or per-face confidence value.** This
   propagates all the way to the viewer.

---

## 3. Environment and dependencies

### 3.1 Base

- Ubuntu 22.04, Python 3.10
- CUDA 12.1+ (match to the rented GPU's driver)
- PyTorch with matching CUDA build

### 3.2 Core packages

```
# reconstruction
opendronemap (Docker image: opendronemap/odm:gpu)
pycolmap
opencv-python
open3d

# neural
torch, torchvision
vggt-omega            # github.com/facebookresearch/vggt-omega
transformers          # for Depth-Anything fallback

# geospatial
pyproj
rasterio
laspy
gdal

# export
trimesh
pygltflib

# infra
numpy, scipy, pyyaml, tqdm, click
```

### 3.3 Licensing check — DO THIS FIRST, DAY 1

**Status as of this spec:** weight access has been requested at
`huggingface.co/facebook/VGGT-Omega`. Access requests are reviewed by an automated process,
not manually queued by the authors, so approval is typically fast — but do not block Track A
work waiting on it (see §10 build order, which starts Track A on day 1 regardless).

While waiting for approval:
- The **public Gradio Space** (`huggingface.co/spaces/facebook/vggt-omega`) is available to
  everyone with no gating — use it today for a qualitative sanity check on a sample clip.
  It does not give you local batch inference, but it validates the model on your footage
  before you've written a line of integration code.
- Once approved, read the actual `LICENSE` file in `facebookresearch/vggt-omega` yourself —
  do not rely on secondhand summaries. Third-party integrations of this model describe it as
  research/non-commercial use only. If that holds, it is very likely fine for a hackathon
  demonstration but worth flagging to your mentors given the organisation is NTRO and the
  problem statement itself lists military reconnaissance as a potential application —
  the *original* VGGT project separately ships a `VGGT-1B-Commercial` checkpoint that
  explicitly permits commercial use excluding military applications, which is a useful
  fallback reference point if licensing becomes a submission concern later.
- If approval is delayed past **day 3**, do not wait — switch to Depth-Anything in the same
  architectural slot (some accuracy cost, zero licensing risk) and revisit VGGT-Ω later if
  access comes through before the day-7 go/no-go checkpoint.

**Record the finding — approved/pending/denied, and the license text itself — in
`README.md` regardless of outcome.**

**Caveat on VGGT-Ω's own reported benchmark numbers:** the authors have publicly flagged a
possible benchmark-contamination issue in the released 1B checkpoint's training data,
meaning some of their published accuracy improvements over the original VGGT may currently
be inflated. The model still works for downstream use — just don't cite the authors' own
benchmark deltas in your pitch deck without a caveat; cite *your own* measured numbers from
§8.5 instead.

---

## 4. Stage 1 — Ingest

**Module:** `src/ingest/`

### 4.1 Video reading

- Accept MP4/MOV, 1080p and 4K.
- Decode with hardware acceleration where available (NVDEC via OpenCV/PyAV).
- Never decode every frame into RAM. Stream, score, keep.

### 4.2 Telemetry parsing (`telemetry.py`)

Support, in priority order:
1. **DJI `.SRT` sidecar** — per-frame GPS lat/lon, absolute + relative altitude, gimbal
   pitch/yaw/roll, ISO, shutter, focal length. Parse via `open-telemetry-kit` or a custom
   regex parser (DJI SRT format varies by firmware — write tolerant parsing, unit-test
   against at least two firmware variants).
2. **Separate flight log CSV** — column auto-detection with a mapping config.
3. **EXIF GPS** on extracted frames.
4. **No telemetry at all** — system must still produce a *scale-free* model and say so
   loudly in the QA report. This path must not crash.

**Output:** `telemetry.parquet` — one row per video timestamp with
`t, lat, lon, alt_gps, alt_baro, roll, pitch, yaw, focal_mm, valid_flags`.

### 4.3 Frame selection (`frame_selector.py`)

Do not sample at a fixed interval. Select adaptively:
- Target overlap between consecutive kept frames: 70–80% (estimate via sparse optical flow
  magnitude vs. frame width).
- Hard reject frames failing the blur gate (§5.2).
- Enforce min/max frame count per chunk from config.

**Acceptance test:** on a 10-minute 4K clip, selection completes in < 60 s and yields a
frame set whose consecutive-pair overlap distribution is centred in [0.7, 0.8].

---

## 5. Stage 2 — Conditioning (the noise-robustness layer)

**Module:** `src/condition/`

This stage directly addresses challenges ii, iii, iv, v. Every operation here must be
**measurable** — the QA report (§8.5) reports before/after numbers for each.

### 5.1 Design principle

**Exclude what is destroyed; correct what is merely degraded.** No algorithm recovers
detail that severe motion blur has physically removed. Attempting to "deblur and use
anyway" injects false geometry. Conditioning therefore has two verdicts per frame:
`CORRECT` or `REJECT`.

### 5.2 Motion blur (challenge ii)

- **Metric:** variance-of-Laplacian, computed on a downscaled grayscale frame.
- Compute the distribution across the whole video first, then set the rejection threshold
  **adaptively** (e.g. reject below the 15th percentile, or below an absolute floor,
  whichever is stricter). A fixed global threshold fails across different videos.
- **Also compute directional blur:** estimate blur kernel orientation via FFT; frames with
  strong unidirectional blur are drone-jerk artifacts and should be rejected harder than
  frames with uniform softness (which may just be atmospheric haze).
- **Rejection is safe** because §4.3 maintains overlap — dropping frames costs coverage only
  if too many consecutive frames are dropped. Track consecutive-rejection runs; if a run
  exceeds N frames, log a **coverage warning** for that flight segment and mark the
  corresponding spatial region as reduced-confidence.
- Mild blur (between the reject threshold and a "clean" threshold): apply unsharp masking
  with conservative parameters, and down-weight that frame's contribution in fusion.

### 5.3 Compression artifacts (challenge ii)

Drone video is heavily H.264/H.265 compressed. Blocking and mosquito noise create false
texture that SfM feature detectors latch onto, producing spurious matches.

- **Detect** blockiness: measure gradient energy at 8×8 / 16×16 block boundaries versus
  interior. High ratio = strong blocking.
- **Correct:** light bilateral or guided filtering (edge-preserving) — strong enough to
  suppress block edges, weak enough to keep real geometric edges. Never use Gaussian blur
  here; it destroys the corners SfM depends on.
- **Suppress false features:** in Track A, reject keypoints whose position lands within
  2 px of a detected block boundary grid, when blockiness score is high.
- Prefer keyframes: when the container exposes frame types, bias frame selection toward
  I-frames, which carry fewer inter-frame artifacts.

### 5.4 Variable illumination and shadows (challenge iii)

Two distinct problems — handle separately.

**(a) Exposure drift across the flight.** As the drone turns, auto-exposure shifts. Photo-
consistency in MVS assumes constant appearance.
- Estimate a per-frame gain/bias by matching histograms of overlapping regions between
  consecutive frames; build a global exposure chain and normalize all frames to a common
  reference.
- Apply CLAHE (contrast-limited adaptive histogram equalization) in LAB colour space on
  the L channel only, so colour fidelity for texturing is preserved.

**(b) Cast shadows.** Hard shadows create strong false edges that are *static in the world*
but change appearance with viewing angle, and they hide facade detail.
- **Detect** shadow regions: low luminance + low saturation + high blue-channel ratio
  (shadows are skylight-illuminated, hence bluer). Produce a per-frame shadow mask.
- **Do not aggressively de-shadow before reconstruction** — that introduces artifacts.
  Instead: (1) down-weight shadow pixels in photo-consistency scoring during MVS, (2) apply
  shadow-aware relighting *only* at the texturing stage, so the final model looks clean.
- **For low-light footage** (disaster-response night/dusk scenarios): apply gamma correction
  plus denoising (non-local means or fast bilateral) *before* feature detection, and record
  that this frame is low-SNR so fusion down-weights it. Set expectations honestly: low-light
  reconstruction accuracy will be measurably worse, and the QA report must show this rather
  than hide it.

### 5.5 Dynamic objects (challenge iv)

Cars, people, and animals violate the static-scene assumption and smear into the mesh.

- **Primary method — semantic masking:** run a lightweight segmentation model
  (YOLOv8-seg or similar, `person / car / truck / bus / motorcycle / bicycle / animal`
  classes) on each selected frame. Dilate masks by ~10 px. Exclude masked pixels from
  feature detection and from MVS depth fusion.
- **Secondary method — geometric consistency:** after an initial pose estimate, reproject
  each frame's depth into neighbours. Pixels whose depth disagrees across views beyond a
  threshold, *despite* being in a high-texture region, are dynamic. This catches objects the
  semantic model misses (debris, floodwater, moving vegetation).
- **Hole policy:** masked regions become gaps, not guesses. They are filled at the *ground
  plane* level only if the region is road/terrain and surrounded by confident geometry;
  otherwise flagged.
- Note for the pitch: PS challenge (iv) asks us to *handle* dynamic objects, not to
  reconstruct them. Removing them cleanly is the correct answer and is exactly where
  feed-forward models struggle on their own.

### 5.6 GPS and sensor noise (challenge v)

- **Outlier rejection:** flag GPS fixes implying impossible velocity/acceleration given the
  platform. Use a median filter over a sliding window, then a constant-velocity Kalman
  filter to smooth the trajectory.
- **Altitude:** prefer **barometric altitude** over GPS altitude when available — GPS
  vertical error is typically 2–3× horizontal. Fuse baro (good relative precision, drifting
  absolute) with GPS (noisy but unbiased absolute) via a complementary filter.
- **IMU:** when present, use for short-baseline orientation priors and to detect rotational
  jerk segments that should trigger stricter blur thresholds.
- **Robust cost:** in all bundle adjustment, use Huber loss on GPS position residuals so a
  handful of bad fixes cannot drag the whole block.
- **RTK/PPK:** when present, treat GPS as high-confidence and tighten its weight in the
  adjustment by an order of magnitude. This is the path to sub-metre accuracy — the system
  should automatically detect and exploit it.

**Acceptance test for Stage 2:** on a clean dataset with synthetic degradation applied
(§8.5), reconstruction error with conditioning enabled must be measurably lower than with
it disabled. Report the delta. If a sub-module shows no improvement, say so honestly rather
than keeping it for show.

---

## 6. Stage 3 — Occluded surface reconstruction (challenge vii — CORE INNOVATION)

**Module:** `src/fusion/`

This is the single highest-value component: it targets Model Completeness (20%) and
Innovation (15%) simultaneously, and it is the direct answer to the "single pass" premise.

### 6.1 The three-zone model

Every point in the scene volume is classified into exactly one zone:

| Zone | Definition | Treatment |
|------|-----------|-----------|
| **Zone 1 — Well observed** | Seen in ≥ N views (config, default 4) with sufficient triangulation angle (> 5°) | Direct MVS depth. Highest confidence. Used as the accuracy backbone. |
| **Zone 2 — Thinly observed** | Seen in 1–3 views, or many views at poor angle | Monocular depth prior, **anchored to Zone 1 metric scale**, fused via confidence-weighted TSDF |
| **Zone 3 — Never observed** | Zero valid observations (e.g. the north facade the drone never flew past) | **Flagged as a gap.** Optionally closed with a clearly-marked, visually distinct "inferred surface" that is excluded from accuracy metrics and toggleable in the viewer. |

### 6.2 Zone classification implementation

1. Build a voxel grid over the scene bounding volume (resolution from config; default
   derived from GSD so a voxel ≈ 2× ground sample distance).
2. For each camera, ray-cast into the grid; accumulate per-voxel: observation count,
   max triangulation angle, mean photometric consistency, mean per-view confidence.
3. Classify per the table above. Persist the zone map — the viewer and QA report both use it.

### 6.3 Zone 2 handling — the monocular anchoring procedure

This is the technically delicate part. Monocular depth is *relative*, not metric. Naively
fusing it corrupts the metric accuracy that Zone 1 provides.

```
For each thinly-observed region R:
  1. Find the boundary band B where R touches Zone 1.
  2. Run monocular depth estimation (Depth-Anything / VGGT-Ω depth head) on frames covering R.
  3. Solve for scale s and shift t minimizing || s * d_mono + t - d_mvs || over B only,
     using RANSAC to reject outliers.
  4. If the fit residual exceeds a threshold, DO NOT fuse — reclassify R as Zone 3.
     A bad anchor is worse than a gap.
  5. Apply (s, t) to R's monocular depth. Integrate into TSDF with weight proportional to
     confidence, always strictly below Zone 1 weights.
  6. Enforce C0 continuity at the boundary with a narrow blend band.
```

**Critical rule:** monocular depth may never *override* MVS depth anywhere the two disagree
in Zone 1. It fills, it does not correct.

### 6.4 Zone 3 handling — honest gap reporting

- Do **not** rely on Poisson surface reconstruction's default hole-closing. Run Poisson with
  the density attribute retained, then **trim** faces whose supporting sample density falls
  below a threshold. This removes the silently-hallucinated surfaces Poisson loves to create.
- Produce a `gaps.geojson` listing each unobserved region's footprint with area, and a
  `coverage_percent` scalar for the whole model.
- In the viewer, render Zone 3 surfaces (if generated at all) in a distinct wireframe/
  translucent style with a legend entry. Never present inferred geometry as measured.

### 6.5 Symmetry and plane completion (optional, only if time permits after §10 day 10)

For building facades specifically: detect dominant planes (RANSAC) in Zone 1, and where a
facade is partially observed, extend the fitted plane across the gap. Mark as inferred.
This is cheap and visually impressive, but **it is a stretch goal** — do not build it before
the core pipeline is green.

---

## 7. Stage 4 — Reconstruction tracks

### 7.1 Track A — classical baseline (`src/recon/track_a_odm.py`)

- Wrap OpenDroneMap (GPU image) as a subprocess with a clean Python interface.
- Feed it the conditioned frames plus a GPS-EXIF or geo.txt file from telemetry.
- Capture ODM's intermediate outputs: sparse cloud, dense cloud, mesh, orthophoto, DSM.
- Config presets: `fast` (reduced `--pc-quality`, downscaled images) and `accurate`.
- **This must work standalone.** A green Track A is the submission floor.

### 7.2 Track B — feed-forward accelerator (`src/recon/track_b_vggt.py`)

- **Chunking is mandatory.** Plain VGGT-class models overflow memory on long sequences —
  published work notes VGGT and peers failing with memory overflow on kilometre-scale
  sequences where chunked variants succeed. Process in overlapping windows, configurable,
  sized from the GPU memory budget below.

  **VGGT-Omega-1B-512 peak GPU memory by frame count** (official benchmark, single GPU,
  624×416 inputs — use this to size chunks, don't guess):

  | Frames | 1 | 10 | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
  |--------|---|----|----|----|----|-----|-----|-----|-----|
  | Peak memory (GB) | 6.0 | 6.7 | 7.8 | 9.7 | 13.4 | 20.8 | 28.3 | 35.7 | 43.2 |

  On a 24 GB card (RTX 4090 / A5000), that puts the safe ceiling around **~180–200 frames
  per chunk** before other pipeline processes (conditioning, Track A running concurrently)
  eat into headroom — leave margin, don't run right up to the 20.8 GB line at 200 frames.
  Default to **128 frames, 16-frame overlap** as a conservative starting config and tune up
  only after profiling on the actual rented instance. Note the table assumes downscaled
  ~512px-wide input — Track B always operates on downscaled frames; full-resolution detail
  comes from Track A's MVS pass in Zone 1, not from Track B.
- **Chunk alignment:** register consecutive chunks via their overlapping frames using
  Sim(3) estimation on the shared point clouds (colored ICP as refinement). Accumulate into
  a global pose graph.
- **Loop/drift control:** for flight paths that revisit an area, detect revisits via global
  descriptors and add relative-pose constraints to the pose graph, then optimize.
- **Extract and keep the confidence maps.** These are the key output — they drive §6 and §7.3.
  Do not discard them.
- **VGGT-Ω over VGGT** where licensing permits: it is the newer, scaled-up model, reported
  to improve camera estimation accuracy substantially over the predecessor and to run with
  far lower memory during training, with a dense-prediction architecture simplification.

### 7.3 The refinement bridge (`src/recon/refine_ba.py`) — ACCURACY CRITICAL

Never ship raw feed-forward geometry. The published pattern is: feed-forward for
initialization, classical bundle adjustment for accuracy.

```
1. Take Track B poses + correspondences as INITIALIZATION.
2. Feed into pycolmap / OpenSfM bundle adjustment as a starting point
   (not from scratch — this is what makes it fast).
3. Add GPS position priors as soft constraints with Huber loss.
4. Run BA to convergence with a hard iteration cap (time budget).
5. Compare reprojection error before vs. after.
6. GATE: if post-BA error is worse than pre-BA error, KEEP THE PRE-BA RESULT and log it.
```

**The gate in step 6 is not optional.** There are documented cases where enabling bundle
adjustment on feed-forward output made large scenes noticeably worse or failed outright
depending on reprojection-error parameters. Build the gate before you build the optimism.

### 7.4 Track selection logic

```
run_mode = config.mode  # "A", "B", "hybrid" (default), "auto"

hybrid:
  1. Track B produces poses + depth + confidence   (fast, covers everything)
  2. Refinement bridge tightens poses              (accuracy)
  3. Track A MVS runs on refined poses for Zone 1  (dense, high-accuracy backbone)
  4. Track B depth fills Zone 2                    (completeness)
  5. Zone 3 flagged                                (honesty)

auto:
  Try hybrid. On any Track B failure (OOM, license block, bad output),
  fall back to pure Track A automatically and log the downgrade.
```

---

## 8. Stage 5–6 — Georeferencing, export, viewer, QA

### 8.1 Metric scale and georeferencing (`src/geo/`) — challenge viii

Feed-forward and monocular methods are **scale-ambiguous by construction** — they recover
shape and orientation but not real-world size. The fix is the telemetry we already parse:
use GPS measurements as position priors in the pose graph and solve for a scale factor on
the estimated camera positions.

```
1. Collect estimated camera centres C_est and corresponding GPS positions C_gps
   (projected to a local ENU frame).
2. Solve Umeyama/Horn similarity transform (scale + rotation + translation),
   RANSAC-robust, using only GPS fixes that passed §5.6 filtering.
3. Apply the transform to all geometry.
4. Refine: re-run BA with GPS priors active so scale is jointly optimized, not just fitted.
5. Report the RMS residual between transformed camera centres and GPS — this is the
   headline "how metric are we" number, and it goes in the QA report.
```

- **Vertical datum matters.** GPS altitude is ellipsoidal; most deliverables want
  orthometric. Apply a geoid model (EGM96/EGM2008 via pyproj) and state which datum the
  output uses in the metadata sidecar.
- **CRS:** auto-select the appropriate UTM zone from the flight's mean longitude. Write CRS
  into GeoTIFF/LAS headers properly — a LAS file without a valid CRS header is not
  georeferenced, no matter how correct the coordinates are.
- **GCP support:** accept an optional GCP file. The PS says *without extensive* GCPs, not
  *without any* — supporting a handful is a legitimate accuracy feature, as long as the
  system works with zero.

### 8.2 Export (`src/export/`)

All six required formats, all carrying correct georeferencing where the format supports it:

| Format | Content | Library | Georeferenced? |
|--------|---------|---------|----------------|
| `.obj` + `.mtl` + texture | Textured mesh | trimesh | Offset in sidecar |
| `.ply` | Point cloud (with confidence as scalar field) | open3d | Local + sidecar |
| `.las` / `.laz` | Point cloud, classified | laspy | Yes — CRS in header |
| `.tif` (GeoTIFF) | DSM + orthophoto | rasterio / GDAL | Yes |
| `.glb` / `.gltf` | Web-ready mesh | pygltflib / trimesh | Offset in extras |
| `.fbx` | Interop mesh | Blender headless or FBX SDK | Offset in metadata |

- Write a `metadata.json` sidecar with CRS, datum, scale residual, coverage percent, model
  bounds, processing time, and software versions.
- **Confidence must survive export.** PLY/LAS get it as a per-point scalar; glTF gets it as
  vertex colours on a toggleable second material.
- For large scenes, tile the output (this is the Scalability 10%).

### 8.3 FBX caveat

FBX is the most fragile export. Budget for it failing. Fallback: export OBJ and convert via
headless Blender. Verify the converted file opens. Do not discover this on demo day.

### 8.4 Web viewer (`viewer/`) — 5% of score, but it is what judges *see*

Keep it simple and fast. Priorities in order:
1. Loads the `.glb` and orbits smoothly. Nothing else matters if this is janky.
2. **Confidence overlay toggle** — recolour the model by confidence. This single feature
   visually communicates the entire innovation story in two seconds.
3. Gap highlighting — show Zone 3 regions.
4. Measurement tool — point-to-point distance. The PS explicitly says the model must be
   suitable for *measurement*; demonstrating it closes the loop.
5. Processing stats panel — time taken, coverage %, accuracy estimate.

Stack: three.js + a GLTFLoader, or Potree for point clouds. Single page, no build step if
possible.

### 8.5 QA and benchmarking (`src/qa/`) — this is how you win Accuracy points

**The synthetic degradation harness (`degrade.py`)** is the most persuasive artifact you can
build for judges. It produces measurable before/after numbers instead of claims.

```
Given a clean dataset with ground truth:
  1. Reconstruct → baseline metrics.
  2. Apply controlled degradations, one at a time and combined:
       - motion blur (directional kernel, varying magnitude)
       - H.264 recompression at decreasing bitrates
       - brightness/gamma reduction + sensor noise (low-light simulation)
       - synthetic hard shadows
       - GPS noise injection (Gaussian + occasional gross outliers)
       - dynamic object insertion
  3. Reconstruct each degraded version, WITH and WITHOUT the conditioning layer.
  4. Produce a table: degradation type × severity × [error with / error without].
```

**The single-pass simulation:** take a multi-strip survey dataset, reconstruct with all
strips (this is your pseudo-ground-truth), then **withhold all but one strip** and
reconstruct again. The delta is your honest single-pass accuracy number, measured on real
data, against a real reference.

**Metrics to compute (`metrics.py`):**
- Cloud-to-cloud distance vs. reference (mean, median, RMS, 95th percentile)
- Camera centre RMS vs. GPS (metric accuracy proxy)
- Coverage percentage (Zone 1+2 / total expected surface)
- Per-zone accuracy breakdown — **report Zone 1 and Zone 2 accuracy separately.** Zone 2
  will be worse. Showing that you know which parts are trustworthy is a strength.
- Wall-clock time per stage

**Reporting stance:** publish an accuracy-vs-condition curve, not a single number. For
example: "0.4 m on well-lit stable footage; 1.8 m on the worst degraded case, with 12% of
surface flagged low-confidence." That is more credible and more defensible under questioning
than a flat "≤1 m" claim.

---

## 9. Performance budget (Processing Speed — 20%)

Target: **< 15 minutes** for a 10-minute video. Budget allocation on the reference GPU:

| Stage | Budget | Notes |
|-------|--------|-------|
| Ingest + telemetry | 1 min | Streamed, hardware decode |
| Conditioning | 2 min | Batched on GPU where possible |
| Track B feed-forward | 2 min | Chunked, batched |
| Refinement BA | 3 min | Hard iteration cap |
| MVS (Zone 1) | 4 min | GPU OpenMVS, depth-map resolution from preset |
| Fusion + meshing | 2 min | TSDF on GPU |
| Export + viewer prep | 1 min | Parallel format writing |
| **Total** | **15 min** | |

**Enforcement in code:**
- Each stage receives a time budget from config and a deadline.
- On overrun, the stage degrades rather than blocking: reduce resolution, cap iterations,
  reduce frame count. Log every degradation.
- `--preset fast` for live demos, `--preset accurate` for benchmark runs. **Demo with fast,
  report numbers from accurate, and state which is which.**
- Instrument from day one. A profiler added on day 12 is useless.

---

## 10. Build order (16 days: 12–28 September)

| Days | Milestone | Definition of done |
|------|-----------|--------------------|
| 1 | Env + licensing check + repo skeleton | Docker builds, GPU visible, VGGT-Ω license recorded in README |
| 1–2 | Ingest + telemetry | Parses DJI SRT → parquet; frame selector hits overlap target |
| 2–3 | **Track A end-to-end** | ODM produces a textured mesh from a sample dataset. **Freeze this as the fallback build and tag it in git.** |
| 3–4 | Export layer | All 6 formats emitted with valid CRS; FBX verified opening |
| 4–5 | Conditioning layer | Blur/artifact/illumination/dynamic modules with unit tests |
| 5–6 | Track B + chunking | VGGT-Ω runs on a 60 s clip, chunks align, confidence maps saved |
| 6–7 | **GO/NO-GO checkpoint** | Compare Track B + refinement vs. Track A alone on the same frames. If Track B is not clearly better on speed and at least equal on accuracy, **drop it and reallocate to §6 and QA.** Decide by end of day 7. No later. |
| 7–8 | Refinement bridge + gate | BA-on-feed-forward with the regression gate working |
| 8–10 | Occlusion engine (§6) | Zone classification + anchored monocular fusion + gap reporting |
| 10–11 | Georeferencing + scale | Camera-centre RMS vs. GPS reported; UTM/geoid correct |
| 11–12 | Web viewer | Loads glb, confidence toggle, measurement tool |
| 12–13 | QA harness + degradation runs | Full results table generated |
| 13–14 | Single-pass simulation benchmark | Honest accuracy number on withheld-strip test |
| 14–15 | Integration hardening, `auto` fallback path, dry runs | Full pipeline runs unattended 3× without intervention |
| 15–16 | Demo rehearsal, docs, pitch deck | Demo runs in < 15 min live, on a laptop, from a cold start |

**Non-negotiable checkpoints:** the day-3 Track A freeze and the day-7 go/no-go. Both exist
to guarantee there is always something to submit.

---

## 11. Definition of done

The system is submission-ready when, on a previously unseen drone video with GPS:

- [ ] It runs end to end without manual intervention
- [ ] It completes within 15 minutes for a 10-minute input on the reference GPU
- [ ] It emits all six output formats with valid, verified georeferencing
- [ ] It reports a camera-centre RMS residual against GPS
- [ ] It reports coverage percent and explicitly flags unobserved regions
- [ ] It reports separate accuracy for well-observed and thinly-observed zones
- [ ] The web viewer loads the model and toggles the confidence overlay
- [ ] The measurement tool returns a distance within the stated accuracy on a known baseline
- [ ] It degrades gracefully with missing IMU, missing baro, and noisy GPS
- [ ] It falls back to Track A automatically if Track B fails
- [ ] The QA report contains the degradation table and the single-pass simulation result
- [ ] A README explains the architecture, the licensing position, and the honest limitations

---

## 12. Honest limitations to state openly (do not hide these — state them first)

Judges respect calibration. Volunteering these is stronger than being caught by them.

1. **≤1 m is achievable in good conditions, not guaranteed in all.** Under genuine
   disaster-response degradation — real motion blur, real low light, single pass — no
   current technique, classical or neural, reliably guarantees sub-metre accuracy
   everywhere. We report accuracy per condition and per zone.
2. **Never-observed surfaces cannot be reconstructed, only inferred.** We flag rather than
   fabricate. This is a correctness decision, not a capability shortfall.
3. **Feed-forward models degrade on very large scenes.** We chunk and refine to mitigate;
   we do not claim the problem is eliminated.
4. **Dynamic objects are removed, not reconstructed.** This is the correct behaviour for a
   static-scene mapping product.
5. **Without RTK/PPK, absolute georeferencing is bounded by consumer GPS accuracy.** Our
   relative geometry can be better than our absolute placement, and we report both.
