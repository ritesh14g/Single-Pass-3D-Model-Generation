# Context for the SIH national-round deck — SIH26158 / PS-17

**Who this is for:** team members building the presentation. Everything a slide needs is here:
the exact problem statement, what we built and with what, what we measured, what makes it
different, how it meets each judging criterion, and what is still to be built.

**How to use it**
- Every claim carries a status. Keep the status on the slide, or at least in your speaker notes:
  - ✅ **Built and measured**: the number comes from a real run on real footage.
  - 🟡 **Built, with a known gap**: it works, but a limitation is logged (issue ID given).
  - ⚪ **Planned**: taken from the build spec (`SIH26158_PS17_BUILD_SPEC.md`), not yet built.
- Every number has a source (`DEVLOG.md` issue ID or section). If you round a number, round it
  honestly; do not upgrade a 🟡 to a ✅.
- Judges will ask. Section 12 has the likely questions with answers.
- As of **26 Sept 2026**: Stages 0–5 built, Stage 6 (web viewer + QA report) planned.
  444 automated tests pass.

---

## 1. The problem statement (verbatim, from the spec §1)

| | |
|---|---|
| **ID** | SIH26158 (PS-17) |
| **Organisation** | National Technical Research Organisation (NTRO) |
| **Category / Theme** | Software · Drone / Robotics |
| **Title** | Single-Pass Drone Video to Accurate 3D Model Generation System |

**What must be built:** an AI-enabled system that generates a **georeferenced and metrically
accurate 3D model** of a scene using **only a single-pass drone video stream** captured from a
moving UAV. It processes frames from **one flight path** and reconstructs:
1. 3D terrain and structures
2. Building facades and rooftops
3. Roads and infrastructure
4. Vegetation and obstacles
5. Textured 3D meshes or point clouds

The model must be suitable for **visualization, measurement and analysis**.

**Eight key challenges (all scored):**

| # | Challenge |
|---|---|
| i | Limited viewing angles due to single flight path |
| ii | Motion blur and video compression artifacts |
| iii | Variable illumination and shadows |
| iv | Dynamic objects (vehicles, humans, animals) |
| v | GPS inaccuracies and sensor noise |
| vi | Real-time or near-real-time processing |
| vii | Reconstruction of occluded surfaces |
| viii | Metric accuracy without extensive GCPs |

**Inputs:** mandatory: drone video (1080p/4K), GPS coordinates, flight metadata.
Optional: IMU, barometric altitude, camera intrinsics, RTK/PPK. The official dataset is given
**at competition time**, so the system must detect which optional inputs exist and adapt.

**Desired output:**

| Parameter | Target |
|---|---|
| Reconstruction type | 3D mesh / point cloud |
| Processing time | < 15 minutes for a 10-minute video |
| Spatial accuracy | ≤ 1 m |
| Coverage | Entire visible scene |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx |
| Visualization | Web-based or desktop viewer |

**Evaluation criteria:**

| Criterion | Weight |
|---|---|
| Reconstruction Accuracy | 30% |
| Model Completeness | 20% |
| Processing Speed | 20% |
| Innovation | 15% |
| Scalability | 10% |
| User Interface | 5% |

**Applications named in the PS:** border and strategic area mapping, disaster damage assessment,
urban planning and smart cities, infrastructure inspection, construction progress monitoring,
archaeological documentation, digital twin generation, military reconnaissance and mission planning.

---

## 2. Our solution in one sentence, and in one paragraph

**One sentence:** a GPS-anchored, confidence-graded 3D reconstruction pipeline that turns one
drone pass into a georeferenced mesh and point cloud in all six formats. It fuses classical
photogrammetry (accurate) with a feed-forward neural depth model (fast). Every surface is labelled
as *measured*, *inferred* or *never seen*.

**One paragraph:** single-pass video sees each surface from only a few angles, so any system
either leaves holes or invents geometry. We do neither silently. The pipeline checks the input
first (Stage 0), selects and cleans frames (Stages 1–2), and reconstructs with COLMAP
structure-from-motion. Dense depth comes from Meta's VGGT-Ω network, anchored frame by frame to
COLMAP's measured geometry: 60–150× faster per frame than classical dense stereo on our GPU (VGGT-Ω 0.22 s, VGGT-1B 0.09 s, classical 13.6 s per frame).
It is then placed on the map from the drone's GPS, with no ground control points
required (Stages 4–5). Stage 3 classifies every surface into three zones:
- **Zone 1:** well observed; the accuracy backbone.
- **Zone 2:** thinly observed; filled from monocular depth only where it agrees with Zone 1.
- **Zone 3:** never observed; reported as gaps in a GeoJSON file, never faked.

That confidence travels into every export format, so a user always knows which parts of the
model to trust.

---

## 3. Architecture (as built)

```
 video + telemetry
        │
 [0] Input check ── refuses unusable input in seconds, explains why, measures GPS-video sync
        │
 [1] Ingest ─────── streaming decode, telemetry from 12+ formats, overlap-targeted frame selection
        │
 [2] Conditioning ─ blur gate, compression-artifact suppression, exposure/shadow handling,
        │           dynamic-object masks (YOLOv8-seg), GPS outlier filtering + smoothing
        │
 [4] Reconstruction
        ├─ Track A (classical): COLMAP SfM on GPU → cameras + sparse points
        │   └─ GPS-prior refinement pass, kept only if it improves (regression gate)
        ├─ Track B (neural): VGGT-Ω depth per frame, anchored to Track A's points
        │   └─ fallback chain: VGGT-Ω → VGGT-1B → classical dense stereo (never crashes)
        └─ OpenMVS mesh + texture
        │
 [5a] Georeferencing ─ RANSAC similarity to GPS → UTM + EGM96 heights (metres)
        │
 [3] Occluded surfaces ─ three-zone voxel classification, anchored Zone 2 fill, gap map
        │
 [5b] Export ─ OBJ, PLY, LAS, GeoTIFF (DSM + ortho), GLB, FBX + zone/confidence layers
        │
 [6] Web viewer + QA report  ⚪ planned
```

Stage numbers follow the spec's section headings. Stage 3 runs after georeferencing because it
works in metres (DEVLOG SPEC-1).

**Cross-cutting engineering (✅ built):**
- **Resumable:** every stage writes to disk under a run manifest. A crash in Stage 5 does not
  re-run Stage 2.
- **Time-budgeted:** each stage gets a time allotment; on overrun it *degrades* (lower
  resolution, fewer frames) and logs it, rather than blocking.
- **GPU first, CPU fallback, never a crash:** every GPU path logs a downgrade and continues on CPU.
- **No hard-coded parameters:** every tunable lives in `configs/default.yaml`, with `fast` and
  `accurate` presets.
- **One process per run folder:** a lock prevents two runs corrupting each other (found on the box).
- **Stage Lab UI** (Streamlit): each built stage has a panel with parameters, a scorecard
  (0–100 from KPIs) and diagnostic charts.
- **444 automated tests.**

---

## 4. What we implemented, stage by stage (with the technology used)

### Stage 0: Input check ✅
- **What:** before spending the time budget, checks format, resolution, fps, bitrate, sharpness,
  exposure, texture, sky share and burned-in overlays. Also checks GPS presence, rate, gaps,
  speed and altitude datum, plus camera and feasibility (parallax, ground sample distance, budget).
  Verdict PASS / WARN / BLOCK per check, each with a fix.
- **Sync:** measures the telemetry-to-video time offset **from the picture** (image motion vs GPS
  speed): Esri −1.3/−1.5 s, DJI_0047 −0.6 s, DJI_0001 −10.5 s. The last was corroborated by the
  file's own clock.
- **Authenticity:** physical-consistency check (share of matches fitting one rigid 3-D scene,
  motion vs GPS, metadata vs telemetry). We say plainly that this is *consistency*, not proof
  that footage is not AI-generated.
- **Evidence:** 13 sample clips triaged in 4–35 s each.
- **Tech:** OpenCV, FFmpeg/PyAV probing, own telemetry sniffers.

### Stage 1: Ingest ✅
- **Formats are detected by content, not extension.** One MISB sample set used `.ts`, `.mp4`,
  `.mpeg4` and `.H264` for the same container.
- **Telemetry formats:**
  - **MISB ST 0601 / STANAG 4609 KLV**, the defence/ISR video metadata standard. Own parser,
    validated on all 8 public MISB sample clips.
  - DJI SRT (3 firmware dialects).
  - DJI binary flight records. Own decoder: the public library decoded 0 records.
  - CSV/TXT, EXIF, GPX, KML, GeoJSON, PX4 ULog, ArduPilot `.bin`/`.tlog`, GoPro GPMF.
- **Frame selection:** adaptive and overlap-targeted (spec: 70–80%), not fixed-interval, with a
  verified overlap estimator (forward-backward optical flow, phase correlation fallback).
- **Hardware decode:** NVDEC, with gating (it can segfault on MPEG-2 TS, S1-15) and an OpenCV
  fallback.
- **Tech:** OpenCV, PyAV, PyNvVideoCodec (NVDEC), pandas/parquet.
- 🟡 **Known gaps:** §4.3 speed target (< 60 s for 10-min 4K) not met on CPU decode (S1-4).
  Overlap is unmeasurable on forward-oblique footage (S1-8).

### Stage 2: Conditioning (noise robustness) ✅ / 🟡
Design rule from the spec: **exclude what is destroyed, correct what is merely degraded.**

| Module | What it does |
|---|---|
| **Motion blur (ii)** | Variance-of-Laplacian + FFT directional-blur detection. Rolling baseline so textureless scenes are not rejected: a real clip went from 71% false rejects to 0% (S1-7) |
| **Compression artifacts (ii)** | 8×8/16×16 blockiness detection. Edge-preserving bilateral + guided filtering (blocking reduced by ratio 0.233 on the test scene). Veto of keypoints on the block grid |
| **Illumination (iii)** | Exposure chain across frames, CLAHE on L channel, shadow mask (low luminance + blue ratio), low-light gamma + denoise |
| **Dynamic objects (iv)** | YOLOv8-seg masks (person/car/truck/bus/bike/animal, dilated). Features on moving objects are excluded from SfM. Geometric-consistency check as a second net |
| **GPS noise (v)** | Physics envelope outlier rejection, median filter, RTS Kalman smoother, baro/GPS altitude fusion, RTK auto-weighting |

- **Tech:** OpenCV, Ultralytics YOLOv8-seg (GPU with CPU retry), NumPy/SciPy.
- 🟡 **Known gaps:**
  - S2-9: the exposure chain drifts on long ISR clips (bounded, not solved).
  - S2-3: the shadow detector over-detects on synthetic footage.
  - S4-9: dynamic masks reach SfM but not yet dense fusion.

### Stage 4: Reconstruction (Track A classical + Track B neural) ✅
- **Track A:**
  - **COLMAP** (pycolmap, CUDA) SfM, with the focal length seeded from telemetry or a
    known-camera table and held fixed. A free focal drifted −17% in height on Esri and −30% on
    DJI_0047, folding the model.
  - Split flights (several SfM pieces) are merged through GPS.
  - **OpenMVS** Delaunay mesh + texture.
- **Refinement bridge (spec §7.3):** a second mapping pass with GPS position priors, **kept only
  if** GPS error drops, reprojection error does not worsen and no frames are lost. Esri:
  camera-vs-GPS 7.5 → 3.2 m.
- **Track B (hybrid, §7.4):** **VGGT-Ω** (Meta, 2026) predicts depth for every frame in about
  10 s. Each frame's depth is anchored to COLMAP's measured points; frames whose depth disagrees
  are rejected. Multi-view consistency fusion follows.
  - **Why not use VGGT's own camera poses?** Measured: one VGGT pass over the 800 m Esri flight
    put cameras **187.6 m** off GPS (COLMAP: 7.1 m). So we use VGGT for dense depth and COLMAP
    for poses: the best of each, chosen by measurement.
- **Measured on Esri (box):**
  - VGGT-Ω depth median error **0.86% (0.89 m)** vs COLMAP's 1920 px stereo, at **24 cm/px**
    ground resolution.
  - 9.7 s for 45 frames, 5.9 GB GPU.
  - Stage 4 score 96.9/100.
- **Fallback chain:** VGGT-Ω → VGGT-1B → classical dense stereo. Each step is logged; the run
  never crashes.
- **Tech:** pycolmap 4.2 (CUDA 12), OpenMVS 2.4, PyTorch, VGGT-Ω / VGGT-1B, trimesh.
- 🟡 **Known gap:** speed (S4-8), see §8.

### Stage 5: Georeferencing and export ✅
- **Georeferencing:**
  - RANSAC similarity (scale + rotation + translation) from camera centres to GPS.
  - A **straight-path roll constraint**: on a straight flight line the roll about the line is
    otherwise undetermined.
  - UTM zone chosen automatically; heights converted from GPS ellipsoidal to **EGM96
    orthometric**.
  - Compound CRS written into the LAS and GeoTIFF headers (e.g. `EPSG:32617+5773`).
  - **No GCPs required.**
- **All six formats written and read back on the box:**

  | Format | Content |
  |---|---|
  | OBJ | textured mesh |
  | PLY | point cloud with confidence, zone and source per point |
  | LAS | point cloud with the CRS in the header, tiled for large scenes |
  | GeoTIFF | DSM + orthophoto |
  | GLB | web-ready mesh, plus confidence and zone versions |
  | FBX | headless Blender 4.2.3 |

- **Stage 3 layers:** zone per point, `model_zones.glb`, `gaps.geojson`, `zone_map.tif`.
- **Mesh check:** a mesh with absurd or non-finite coordinates is refused before the writers
  (point formats still go out).
- **Tech:** pyproj (+ EGM96 geoid), rasterio/GDAL, laspy, trimesh, pygltflib, Blender headless.
- ✅ **Closed 26 Sept (laptop-verified; box re-run pending):**
  - **Altitude datum measured, not assumed:** the log's takeoff altitude is compared with a terrain
    model plus the geoid. On DJI_0047 it found the altitude was ellipsoidal (to 0.6 m) *and* that the
    log's "altitude" column was really height above takeoff. Before this fix, DJI cameras sat about
    19 m too high.
  - **Orthophoto rendered from the textured mesh** at the texture's own resolution (Esri 0.14 m,
    against 0.5 m from the point cloud).
  - **Ground control points (optional):** triangulated from the images and weighted by accuracy.
    *Check points* are kept out of the fit and give an independent accuracy number.
  - **LAS classified:** ground / above ground / low noise (ASPRS classes 2 / 1 / 7).

### Stage 3: Occluded surfaces, the core innovation ✅
Spec §6: "the single highest-value component". It answers challenges (i) and (vii) directly and
targets Completeness (20%) and Innovation (15%).

**Three-zone model:**

| Zone | Rule (config) | Treatment |
|---|---|---|
| **1 Well observed** | ≥ 4 confirming views **and** triangulation angle ≥ 5° | Measured stereo depth. Weight 1.0. The accuracy backbone |
| **2 Thinly observed** | 1–3 views, or many at a poor angle | Monocular (VGGT-Ω) depth **anchored to Zone 1**. Weight ≤ 0.45 |
| **3 Never observed** | Seen by the cameras, no surface measured | **Flagged as a gap** in `gaps.geojson` (WGS84, with areas); never faked |

**How:**
- Sparse voxel grid (voxel ≈ 2× ground sample distance).
- Per voxel: confirming views, widest triangulation angle, z-buffer visibility per camera,
  photometric consistency.
- A ground coverage map from the camera footprints gives the gap regions.

**Zone 2 anchoring (spec §6.3):**
1. Find the boundary band where the thin region touches Zone 1.
2. Fit depth = s·mono + t there with RANSAC. Tolerances grow with range, because monocular
   error does.
3. **Refuse** the frame if inliers are too few, the residual is too high, or the scale disagrees
   with the georeferencing.
4. Feather the seam.
5. Fuse with weight strictly below Zone 1.

**Critical rule:** monocular depth never enters a Zone 1 voxel. It fills, it never corrects.
The evaluator re-checks this on every run: 0 violations so far.

**Honest accuracy:** each fit is tested on a *held-out* ring of Zone 1 pixels it never saw.

**Measured (box):**

| | Esri (MISB clip, 46 cameras) | DJI_0047 (4K, 136 cameras) |
|---|---|---|
| Stage 3 score | 86.4 | 90.9 |
| Zone 1 / 2 / 3 of visible ground | 47.1 / 21.1 / 31.8% | 76.7 / 11.1 / 12.2% |
| Coverage, measured → with fill | 65.4 → **68.2%** | 87.7 → 87.8% |
| Fill frames anchored | 33 of 35 | 1 (of 3 planned; the time budget stopped it, fixed since, box re-check pending) |
| **Held-out Zone 2 error** | **0.48% of depth (0.51 m at ~105 m)** | 0.82% (0.85 m) |
| Monocular points inside Zone 1 | 0 | 0 |
| Gaps reported | 218 regions, 59,304 m² | 184 regions, 24,072 m² |

Esri's uncovered ground is mostly a river: water gives no consistent stereo depth.

**Refusal in practice:** on Esri, frame 216 fitted a scale of 267 on a thin band (true ≈ 1). It
would have contributed a third of all fill points. It is now refused by the scale check.

**Stated deviations from the spec (S3-3):**
- Zone 2 uses a confidence-weighted voxel fusion instead of TSDF.
- Mesh faces with no measured support are *flagged* as inferred rather than trimmed.
- Plane completion (§6.5, a stretch goal) was not built.

### Stage 6: Web viewer + QA report ⚪ planned (from spec §8.4–8.5)
- **Viewer** (three.js / Potree, single page), in the spec's priority order:
  1. loads the `.glb` and orbits smoothly;
  2. **confidence overlay toggle** (recolour by zone/confidence);
  3. gap highlighting (Zone 3 regions);
  4. **point-to-point measurement tool**;
  5. stats panel: time, coverage %, accuracy estimate.
- **QA report:**
  - **Synthetic degradation harness:** blur, recompression, low light, shadows, GPS noise and
    dynamic objects, reconstructed with and without conditioning, giving a table of error deltas.
  - **Single-pass simulation:** reconstruct a multi-strip survey with all strips, then with one;
    the difference is an honest single-pass accuracy number.
  - Per-zone accuracy (Zone 1 vs Zone 2) against a reference.
- **Inputs already exist:** exports + `model_zones.glb` + `gaps.geojson` + per-point confidence.
  The viewer only has to display them.

---

## 5. Technology stack (for the "tech used" slide)

| Layer | Technology | Role | Licence |
|---|---|---|---|
| Language / infra | Python 3.10–3.13, NumPy, SciPy, pandas, Click, PyYAML | Pipeline, config, CLI | Open source |
| Video | OpenCV, PyAV, FFmpeg, PyNvVideoCodec (NVDEC) | Decode, frame analysis | Open source / NVIDIA |
| Telemetry | Own parsers (KLV ST 0601, DJI SRT/flight record, ULog, …) | GPS/attitude/metadata | Ours |
| Conditioning | OpenCV, Ultralytics YOLOv8-seg | Blur, artifacts, light, dynamic masks | AGPL (YOLO) |
| Structure-from-motion | COLMAP via pycolmap (CUDA 12) | Cameras, sparse points, GPS-prior BA | BSD |
| Neural depth | **VGGT-Ω** (Meta FAIR), VGGT-1B fallback, PyTorch | Fast dense depth | Non-commercial research (see §9) |
| Meshing / texture | OpenMVS 2.4 | Delaunay mesh, texture | AGPL |
| Geospatial | pyproj (EGM96 geoid), rasterio/GDAL, laspy | CRS, datum, GeoTIFF, LAS | Open source |
| Export | trimesh, pygltflib, Blender 4.2 headless | OBJ/GLB/FBX | Open source (GPL Blender, as a tool) |
| UI | Streamlit (Stage Lab); three.js/Potree (viewer, planned) | Diagnostics, visualisation | Open source |
| Testing | pytest (444 tests) | Regression safety | — |
| Hardware used | Institute GPU notebook: **NVIDIA H100 MIG 2g.20gb slice (20 GB), 3 CPU cores, 56 GB RAM**, CUDA 12.8; dev laptop without GPU | — | — |

---

## 6. What makes our solution unique (Innovation slide)

1. **Honest three-zone model.** Every surface is labelled measured / anchored-inferred /
   never-seen, and the label survives into PLY, LAS and GLB. Most pipelines silently close holes
   (Poisson surfaces); we report gaps as a GeoJSON a GIS analyst can open.
2. **Confidence-gated hybrid by measurement, not by fashion.**
   - The neural model gives depth 60–150× faster per frame than classical stereo on our GPU
     (VGGT-Ω 0.22 s, VGGT-1B 0.09 s, classical 13.6 s).
   - Classical SfM gives the poses, because we measured VGGT's own poses at 187.6 m off GPS on a
     long flight.
   - Each frame's neural depth is checked against measured geometry before use.
3. **Anchored monocular fill with refusal gates.**
   - Depth-relative RANSAC, a held-out accuracy check and a georeferencing-scale sanity check.
   - A bad anchor becomes a gap, not wrong geometry. The spec's rule, "a bad anchor is worse than
     a gap", is enforced in code and re-verified by the evaluator.
4. **Metric accuracy with zero GCPs:** GPS-anchored similarity, straight-path roll constraint,
   GPS-prior refinement with a regression gate, EGM96 orthometric heights. The GPS altitude datum
   is *measured* against terrain rather than assumed. Optional GCPs add surveyed accuracy, and
   check points give an independent accuracy figure.
5. **Defence-grade input handling:**
   - Native **MISB ST 0601 / STANAG 4609 KLV** parsing, the metadata standard of ISR drone video.
   - DJI and a dozen other telemetry formats, detected by content.
   - GPS-to-video sync measured from the picture itself.
6. **Refuse bad input in seconds.** The input check explains *why* a clip cannot give a 3-D
   model before 15 minutes of GPU time are spent.
7. **Built to degrade, not crash:**
   - GPU → CPU fallback everywhere; fallback chain VGGT-Ω → VGGT-1B → classical.
   - Per-stage time budgets that trade resolution for time.
   - Every stage resumable; every degradation logged and shown on the scorecard.
8. **Validated against independent ground truth.** On Esri we compared against **USGS 3DEP
   lidar** (and NAIP imagery), not just against the drone's own GPS (see §7).

---

## 7. Headline results (Results slide)

Clips (both public, Florida, US):
- **Esri** (`Esri_multiplexer_1.mp4`): a MISB KLV drone video, about 1.7 min.
- **DJI_0047**: a DJI Phantom 3, 4K, with the DJI flight log.

Both are from the QGIS FMV sample set. The competition dataset arrives at the event.

| Metric | Esri | DJI_0047 | Source |
|---|---|---|---|
| Frames reconstructed | 46–49 of 50 | 136 of 136 | box runs |
| Camera centres vs GPS (RMS) | 3.2–3.4 m | **1.35 m** | S4-1, S4-10 |
| Height above ground vs telemetry | −1.4 to −2.6% | n/a | Stage 4 |
| Stage 4 / 3 / 5 scores (0–100) | 96.9 / 86.4 / 90.0 | 93.3 / 90.9 / 93.3 | box, 2026-09-23 |
| Coverage of visible ground | 68.2% | 87.8% | Stage 3 |
| Zone 2 held-out error | 0.48% of depth (0.51 m) | 0.82% (0.85 m) | Stage 3 |
| VGGT-Ω depth vs classical stereo | 0.86% (0.89 m) median | n/a | Track B probe |
| Output formats | all 6 ✅ | all 6 ✅ | Stage 5 |
| Full pipeline wall time | ~11 min (657 s) | ~38 min (2,255 s) | box, 20 GB slice, 3 CPU cores |

**Accuracy against independent ground truth (Esri vs USGS 3DEP lidar, DEVLOG §6 2026-09-22):**
- The model's **shape** is good: after one 7-parameter fit it is **1.3 m horizontal / 1.7 m
  vertical** from the lidar. Along the flight it matches GPS to **1.1 m over any ~90 m stretch**.
- Its **placement** is limited by the telemetry: the Esri KLV GPS track itself is **7–13 m off
  and 2.3–2.9% short** against lidar and NAIP. As delivered, the model sits 2–14 m from the lidar
  horizontally.
- **Conclusion:** relative geometry is close to the 1 m target; absolute placement is bounded by
  the GPS we are given. RTK/PPK input (supported, auto-weighted) is the path to sub-metre absolute
  accuracy, which is also the spec's stated position (§12.5).

Say it this way. Do not claim "≤ 1 m achieved". Claim "≈1 m relative accuracy measured against
lidar; absolute accuracy bounded by consumer GPS, and here is the evidence".

---

## 8. How we meet each criterion (Feasibility / criteria slide)

| Criterion (weight) | What we deliver | Status | Evidence |
|---|---|---|---|
| **Accuracy (30%)** | GPS-anchored metric model, fixed focal, GPS-prior refinement with regression gate, per-zone confidence, lidar-validated | ✅ / 🟡 | 1.35 m (DJI) and 3.2 m (Esri) vs GPS; 1.3/1.7 m vs lidar after fit; per-zone accuracy vs reference is Stage 6 (S3-4) |
| **Completeness (20%)** | Three-zone engine, anchored Zone 2 fill, gaps reported with areas | ✅ | Coverage 68.2% (Esri, river-limited), 87.8% (DJI); 0 fill points in Zone 1 |
| **Speed (20%)** | Time budget per stage with automatic degradation; VGGT depth instead of slow stereo | 🟡 **our biggest gap** | Esri 1.7 min video → 11 min end to end; DJI 4K 136 frames → 38 min. Target is 15 min for 10 min of video. See below |
| **Innovation (15%)** | Honest zones, gated hybrid, refusal gates, KLV-native, input check | ✅ | §6 above |
| **Scalability (10%)** | Streaming decode, chunked neural inference sized to GPU memory, tiled LAS export, voxel caps, resumable stages | ✅ / 🟡 | 6.1 M-point DJI cloud processed; Stage 3 classification 2.5× faster on a DJI-sized benchmark (laptop; box confirmation pending, S3-8) |
| **UI (5%)** | Stage Lab (Streamlit) now; web viewer with confidence toggle and measurement tool | ✅ / ⚪ | Viewer = Stage 6 |

**Speed, stated honestly (S4-8):**
- Measured on a **20 GB MIG slice with only 3 CPU cores**:
  - The GPU parts are fast: VGGT depth is about 10 s per clip.
  - The CPU-bound parts dominate: decode + conditioning ~100 s, OpenMVS mesh + texture ~126 s on
    Esri. On DJI_0047's 4K, 136-frame input, first-pass SfM mapping alone took 1,147 s.
- **Planned levers:**
  - NVDEC decode for all containers.
  - GPU OpenMVS densify.
  - Smaller mesh targets for the fast preset.
  - SfM at reduced image size on 4K.
  - Running on the spec's reference setup (a full 24 GB GPU with more CPU cores).
- Do **not** present the speed levers as done. Present the budget mechanism as done (it is), and
  the numbers as measured.

**Feasibility, in short:**
- It **already runs end to end** on real public drone footage, on a modest 20 GB GPU slice, and
  writes all six required formats with a valid CRS.
- Everything is open-source except the neural model, which has a working licence-clean
  fallback: classical Track A.
- No special hardware, no GCP survey, no RTK required (RTK used when present).

---

## 9. Licensing position (be ready for this; NTRO is the problem owner)

- VGGT-Ω / VGGT-1B carry non-commercial research licences with an acceptable-use clause that
  excludes military / espionage use (DEVLOG LIC-1).
- **Team decision (2026-09-22):**
  - We use VGGT for the competition prototype, because the PS centres disaster management and
    lists military use as one optional application.
  - A selected project would move to an in-house model built with government support.
  - **Track A (COLMAP BSD, OpenMVS AGPL) is a complete licence-clean pipeline on its own:** the
    system works without VGGT, only slower.
- Say this proactively if asked; it shows we read the licences.

---

## 10. Honest limitations (spec §12 + what we measured). Put these on a slide.

1. **≤ 1 m is achievable in good conditions, not guaranteed in all.** We report accuracy per
   condition and per zone.
2. **Never-observed surfaces cannot be reconstructed, only flagged.** This is a correctness
   decision.
3. **Absolute accuracy is bounded by the GPS.** Esri's own GPS track is 7–13 m off lidar; our
   relative geometry is better than that.
4. **Speed target not yet met on long clips** on our 3-core GPU slice (S4-8).
5. **Water, sky and textureless areas yield no stereo depth.** They are reported as gaps
   (Esri's river).
6. **Dynamic objects are removed, not reconstructed.** That is correct for mapping.
7. **Feed-forward models degrade on large scenes.** We use them for depth only, anchored per frame.

---

## 11. Suggested slide outline

If SIH provides an official template, follow its section order. Its usual sections are title,
idea/solution, technical approach, feasibility and viability, impact and benefits, research and
references. Map the content below into them.

1. **Title:** PS ID SIH26158, title, NTRO, team name, member names *(fill in)*.
2. **Problem:** the single-pass challenge in one picture: one flight line, a building seen from
   one side, occluded facades. The 8 challenges as icons.
3. **Our solution:** the one-sentence pitch (§2) + the three-zone graphic (green / amber / red).
4. **Architecture:** the pipeline diagram from §3 (redraw it cleanly); mark Stage 6 as "in
   progress".
5. **Technical approach I, robustness:** Stage 0–2 table (challenge → our method).
6. **Technical approach II, reconstruction:** the Track A + Track B hybrid, why VGGT for depth and
   COLMAP for poses (187.6 m vs 7.1 m), the fallback chain.
7. **Technical approach III, occluded surfaces:** the zone table, the anchoring steps, "a bad
   anchor is worse than a gap", the frame 216 refusal story.
8. **Results:** the §7 table + one screenshot of the textured model + one of `model_zones.glb` +
   the lidar comparison.
9. **Criteria fit:** the §8 table (keep the statuses).
10. **Uniqueness:** the §6 list, four to five points maximum on the slide.
11. **Feasibility and viability:** hardware used, runtime, open-source stack, licensing position,
    what remains (Stage 6, speed levers) with a short plan.
12. **Impact and applications:** disaster damage assessment first, then infrastructure, urban
    planning, border mapping, digital twins.
13. **Limitations:** §10, stated confidently.
14. **References:** COLMAP, OpenMVS, VGGT/VGGT-Ω (Meta FAIR), MISB ST 0601, USGS 3DEP, EGM96,
    YOLOv8.

> ⚠️ **Do not use any current textured-model image in the deck.** On 26 Sept we found every
> textured mesh so far is 30–60% black: an OpenMVS colour-correction setting (DEVLOG S4-11). It's
> fixed in code, but the box must re-run Track A to produce correct textures. Use point-cloud,
> orthophoto and zone images until then.

**Visuals you can produce today:**
- Stage Lab screenshots: `.venv\Scripts\python -m streamlit run ui/app.py`. It shows scorecards
  and charts for Stages 0–5.
- Exported data from the Esri run in `data/box/runs/esri_full2/export/`: LAS and GeoTIFF open in
  QGIS. The GLB/OBJ there have the black-texture fault; wait for the box re-run.
- The newest Stage 3 outputs (`model_zones.glb`, `gaps.geojson`, `zone_map.tif`) for Esri and
  DJI_0047 are still on the GPU box under `data/interim/{esri_s3,dji47_s3}/`. Bring them back with
  `scripts/box_collect.sh` (`CLOUD_GPU_GUIDE.md`) before making the zones slide.
- `gaps.geojson` opens directly in QGIS or geojson.io over a satellite basemap. It makes a strong
  visual for "we tell you exactly what we did not see".

---

## 12. Likely judge questions, and our answers

- **"Do you meet 1 m accuracy?"**
  - Relative geometry: ≈1 m, measured against USGS lidar.
  - Camera track vs GPS: 1.35 m on DJI.
  - Absolute placement is bounded by the GPS supplied: Esri's own GPS is 7–13 m off lidar.
  - With RTK/PPK, which the system detects and weights automatically, absolute accuracy follows.
- **"Why not just use COLMAP / ODM / Pix4D?"**
  - Classical dense stereo was 13.6 s per frame on our GPU vs 0.22 s for VGGT-Ω (≈60×) and
    0.09 s for VGGT-1B (≈150×).
  - Classical tools also silently close holes.
  - ODM needs Docker (absent on our box) and is CPU-bound there; we use the same class of
    components (COLMAP, OpenMVS) directly.
- **"Why not just use the neural model?"** Measured: VGGT's own poses were 187.6 m off GPS on an
  800 m flight. Neural depth is excellent (0.86% error); neural poses at that scale are not.
- **"What happens to the back of a building the drone never saw?"** It is Zone 3: reported as a
  gap with its area in `gaps.geojson`, and never presented as measured.
- **"How do you know the filled (Zone 2) parts are right?"** Every fill is tested on held-out
  measured pixels: 0.48% of depth error on Esri. Fills that fail the checks are refused (example:
  frame 216). Zone 2 never overrides Zone 1.
- **"Will it run on our data?"**
  - Formats are detected by content; 12+ telemetry types are supported, including STANAG 4609.
  - The input check tells you in seconds whether a clip can work and why not.
  - Missing optional inputs have logged fallbacks.
- **"Is it real-time?"**
  - Not yet at the PS target for long clips on our 3-core slice.
  - Budget enforcement and degradation are built.
  - Speed levers are listed in §8.
- **"Licence of VGGT for NTRO?"** See §9. The classical track is a licence-clean complete
  fallback.
- **"How do you handle cars and people?"** YOLOv8-seg masks keep them out of feature matching;
  they are removed, not reconstructed. That is correct for a static map (spec §5.5). Masks in
  dense fusion are an open item (S4-9).

---

## 13. Do / don't when making slides

- **Do:**
  - Quote measured numbers with their clip name.
  - Show statuses.
  - Show the gap map.
  - Show a refusal as a strength.
- **Don't:**
  - Claim "< 15 min" or "≤ 1 m everywhere".
  - Cite VGGT-Ω's own published benchmark gains. The authors flagged possible benchmark
    contamination (spec §3.3); cite our own measurements instead.
  - Present Stage 6 (viewer, degradation harness, single-pass simulation) as built.
  - Present the stage scores (0–100) as accuracy. They are our internal scorecards.

## 14. Glossary (for speaker notes)

- **SfM (structure from motion):** recovers camera positions and sparse 3-D points from
  overlapping images.
- **MVS / dense stereo:** per-pixel depth from several calibrated views.
- **Bundle adjustment (BA):** joint refinement of cameras and points to minimise reprojection
  error.
- **Feed-forward model (VGGT-Ω):** a neural network that predicts depth/cameras in one pass,
  without iterative optimisation.
- **GSD (ground sample distance):** ground size of one pixel.
- **Triangulation angle:** angle between two viewing rays to a point. Small angle means poor
  depth.
- **Zone 1/2/3:** well observed / thinly observed (anchored fill) / never observed (gap).
- **Georeferencing:** placing the model in real-world coordinates (here UTM + EGM96 heights).
- **KLV / MISB ST 0601 / STANAG 4609:** the standard for metadata embedded in ISR drone video.
- **RANSAC:** robust fitting that ignores outliers.
- **Orthometric height (EGM96):** height above mean sea level (geoid), vs the GPS ellipsoid.

**Deeper sources in the repo:** `SIH26158_PS17_BUILD_SPEC.md` (requirements and plan),
`DEVLOG.md` (status board §2, dead ends §4, open issues §5, every measurement in §6),
`CLOUD_GPU_GUIDE.md` (GPU box setup), `configs/default.yaml` (every parameter, with the reason).
