# Guide: building the SIH 2026 national-screening PPT

**For:** the team, filling `NMIET_SIH_2026_PPT_Template.pdf`'s seven slides (title + 6 content slides).
**Source of every number below:** the deployed prototype at
**https://ritesh14g.github.io/Single-Pass-3D-Model-Generation/** and `DEVLOG.md`. Open the site's
**Quality report** panel on either flight to see any figure below traced back to its own KPI.

## Rules from the template (page 7) — do not break these
1. **Maximum 6 slides, including the title slide.** That is only **5 content slides** for everything
   below. Combine where the guide suggests; do not add slides.
2. Points/diagrams/infographics, not paragraphs.
3. Keep the template's slide titles and bullet prompts exactly as given — fill them in, do not reword
   the headings.
4. Save as **PDF** for upload. No PPT/DOCX.
5. Delete the "Important Instructions" slide before you upload.

Below, each of the template's slides is mapped to what we actually built, with exact figures to quote
and which flight (Esri or DJI_0047) each figure comes from. Where two content slides can share one
(e.g. Technical Approach + Feasibility, or Impact + Research), that is noted — use it only if you are
short on space, since the template gives them as separate slides.

---

## Slide 1 — Title page

| Field | Fill with |
|---|---|
| Problem Statement ID | **SIH26158** |
| Problem Statement Title | **Single-Pass Drone Video to Accurate 3D Model Generation System** |
| Theme | Software · Drone / Robotics |
| PS Category | Software |
| Team ID | *(your registered team ID)* |
| Team Name | *(your registered team name)* |

---

## Slide 2 — Idea Title (Proposed Solution)

Template asks for three things. Use three short blocks, not paragraphs.

### 1. Detailed explanation of the proposed solution
A one-line description, then a 7-step pipeline diagram (numbers = the stage IDs judges can ask about):

> **One drone video + its GPS log in → a georeferenced, textured 3D model out, in six industry-standard
> formats, with a written quality report — from a single flight pass, no extra flyovers.**

Pipeline (put this as a horizontal flow diagram, one icon per stage):

```
0 Input check → 1 Ingest → 2 Conditioning → 3 Reconstruction → 4 Occlusion handling → 5 Georeferencing & export → 6 Viewer & QA
```

One line under each stage (these are the actual stage names our own code and quality report use —
keep them, since the live demo uses the same names):
- **0 Input check** — reads the file and the GPS log before spending any processing time; blocks bad
  input with a stated reason and fix.
- **1 Ingest** — decodes the video, selects overlapping frames, rejects blurred ones.
- **2 Conditioning** — fixes compression blocking, exposure drift, shadows, moving objects, noisy GPS.
- **3 Reconstruction** — solves camera positions (COLMAP), adds a learned depth model (VGGT-Ω) for
  detail, builds and textures the mesh.
- **4 Occlusion handling** — labels every surface by how well it was seen (3 zones), fills thin areas
  from anchored depth, draws honest gap outlines for anything never seen.
- **5 Georeferencing & export** — fits the model to GPS/UTM, writes **all 6 required formats**
  (OBJ, PLY, LAS, GeoTIFF, GLB, FBX).
- **6 Viewer & QA** — a web viewer (photo/confidence/zone views, measurement tool) and an automated
  quality report scored against the problem statement's own targets.

### 2. How it addresses the problem
Map straight onto the PS's **8 named challenges** (§1 of the spec) — a table like this reads well on
a slide:

| PS challenge | How we solve it |
|---|---|
| i. Limited viewing angles (single pass) | Stage 4 classifies every surface into 3 observation zones and fills thin (Zone 2) areas from a learned depth model anchored to measured geometry — never guessing where it can't check itself |
| ii. Motion blur / compression artefacts | Stage 2: adaptive blur gate + rejection; compression-block suppression measured before/after |
| iii. Variable illumination / shadows | Stage 2: exposure normalisation across the whole flight, shadow detection and down-weighting |
| iv. Dynamic objects | Stage 2: moving-object masking, excluded from both camera-solving and the dense surface |
| v. GPS inaccuracies / sensor noise | Stage 2 GPS filtering + Stage 3 fitting; the quality report states the exact residual (e.g. **2.05 m RMS** on DJI_0047) rather than assuming GPS is perfect |
| vi. Real-time / near-real-time processing | Stage 6 quality report **forecasts time for a full 10-minute video** on the same hardware — see Slide 4 |
| vii. Reconstruction of occluded surfaces | Stage 4's whole purpose: 3-zone classification + anchored fill + honest gaps (never invented) |
| viii. Metric accuracy without extensive GCPs | Stage 5 fits to the drone's own GPS track; ground control points are supported when available but not required |

### 3. Innovation and uniqueness
Pick 3–4 of these (do not list all — keep it to bullets that fit):
- **Every unseen surface is drawn as a gap, never invented.** Most single-pass systems either leave
  holes or silently interpolate; we classify observation quality per-voxel and only fill where a
  learned depth estimate can be checked against measured geometry first.
- **A live, working demo**, not slides claiming a pipeline — https://ritesh14g.github.io/Single-Pass-3D-Model-Generation/
  runs the real input check, shows the real per-stage timings, and lets a judge open and rotate the
  actual OBJ/PLY/LAS/GeoTIFF/GLB/FBX files two real flights produced.
- **A quality report that grades itself against the problem statement**, broken down by the ~19
  individual processes inside the pipeline (not just per stage), each one stating what its score does
  and does not measure — so accuracy claims are falsifiable, not asserted.
- **A geometry gate that refuses to award a good score to a broken reconstruction.** We found our own
  camera-solver could occasionally fold on a single straight flight line; rather than hide it, we
  added an automatic check (cameras must agree with GPS within 25 m) that caps the score if it happens
  — this is on the live site (S4-12 in the project log).
- Works from **whatever telemetry the footage carries** — DJI `.SRT`, flight-log CSV, GPX/KML, PX4/
  ArduPilot logs, or MISB KLV embedded in the video stream itself (no separate file needed) — since the
  official dataset is only revealed at competition time.

---

## Slide 3 — Technical Approach

### Technologies used
- **Languages/runtime:** Python, Click CLI, Streamlit (internal dev UI)
- **3D reconstruction:** pycolmap (COLMAP) for structure-from-motion; **VGGT-Ω** (Meta, feed-forward
  transformer) for monocular depth, anchored/validated against the measured geometry before use
- **Dense surface + texturing:** OpenMVS
- **Geospatial:** PROJ/pyproj (UTM + EGM96 orthometric heights), laspy (LAS classification), rasterio
  (GeoTIFF)
- **Export:** trimesh, pygltflib, headless Blender 4.2 (FBX)
- **Web viewer/demo:** three.js (WebGL), a hand-written LAS point-cloud reader for the browser
- **Hardware used for the measured runs:** NVIDIA H100 80 GB MIG slice (2g.20gb — 19.6 GB, 3 CPU
  cores); also validated on free-tier cloud GPUs (Kaggle T4) as a fallback path when a dedicated GPU
  isn't available

### Methodology / process
Reuse the same 7-stage diagram from Slide 2 (do not redraw it) — or, if space allows, expand it with
the confidence-zone diagram: a coloured cross-section showing **Zone 1 (measured, green) / Zone 2
(anchored fill, amber) / Zone 3 (gap, red)**, which is exactly what the live viewer shows under its
"Zones" toggle.

**Working prototype line (put this prominently — it's the strongest thing on this slide):**
> Live, interactive prototype: **https://ritesh14g.github.io/Single-Pass-3D-Model-Generation/**
> Two real flights processed end-to-end — input check → pipeline → all 6 output formats, each viewable
> and downloadable — plus the full quality report.

*(If the template's "Feasibility and Viability" slide is tight on space, you may fold "Methodology"
into this slide and use Slide 4 purely for challenges/risks — the template allows combining content as
long as slide count and headings stay within its limits.)*

---

## Slide 4 — Feasibility and Viability

### Feasibility — state plainly, with the number
> **Feasibility is demonstrated, not projected: the full pipeline has run end-to-end on two real,
> independent drone flights**, producing georeferenced models in every required format and a measured
> accuracy figure.

| | Esri FMV sample (surveillance aircraft, ~110 m) | DJI_0047 (quadcopter, 58 m) |
|---|---|---|
| Video length | 1 min 41 s, 4K | 1 min 44 s, 4K |
| Processed in | 7 min 6 s | 44 min 14 s |
| Ground reconstructed (coverage) | 69.3% | 88.0% |
| Camera positions vs GPS | 3.35 m RMS | 2.05 m RMS |
| Shape accuracy vs independent lidar (Zone 1, 100 m tiles) | **2.87 m RMS** | *(no lidar available for this site — see below)* |
| Output formats produced & re-verified | 6 / 6 | 6 / 6 |
| Quality scores (5 process groups, 0–100) | 83 · 86 · 86 · **97** · 91 | 78 · 100 · 96 · **97** · 94 |

Put the **direct link** on the slide so judges can check it live: the "Quality report" button on the
site opens all of the above, broken into ~19 individually-scored processes, each stating what it does
and does not measure.

### Potential challenges and risks (be specific, this is credibility, not weakness)
| Risk | What we found | Status |
|---|---|---|
| **Processing speed vs the PS's 15-minute target** | Measured 7 min for a 1.7-min flight on a *shared, 3-core* GPU slice — the live quality report **projects this to ~42 min for a full 10-minute video** on the same hardware. The reference machine in the spec has 8+ dedicated cores. | Open; route identified (keyframe-only decoding, overlapped stages, dedicated hardware) — quoted directly on the site's Speed tab, not hidden |
| **Single-flight-line reconstruction can occasionally fold** | Found via our own automated GPS-agreement check on one DJI_0047 run; root-caused to solver non-repeatability on a straight line with no loop closure, not to any of our processing stages | Contained: a geometry gate now caps the score automatically if it recurs (see S4-12); fix (retry-on-failure) scoped for next iteration |
| **Absolute placement is bounded by the drone's own GPS** | 2–3.5 m camera-GPS agreement on both flights; this is a property of consumer GPS, not of our reconstruction | Mitigated: RTK/PPK and ground-control-point support already built and accepted as optional input |
| **No universal lidar reference for every site** | DJI_0047's site has no public lidar; the report states this explicitly rather than fabricating an accuracy number | By design: the QA report says "no independent reference given" instead of hiding the gap |

### Strategies for overcoming these
- Speed: keyframe-based frame selection (avoid decoding every frame), pipeline stages to overlap
  instead of running sequentially, target the spec's reference-class GPU (8+ cores) for the national
  round demo hardware.
- Reliability: automatic retry with a different initial frame pair whenever the GPS-agreement gate
  fails, instead of a single deterministic attempt.
- Accuracy without lidar: ground-control-point ingestion is already implemented; a light on-site
  GCP capture procedure is the low-cost path to verified sub-metre accuracy for any deployment site.

---

## Slide 5 — Impact and Benefits

### Potential impact on the target audience
The PS is filed by **NTRO**; state the primary and adjacent audiences:
- **Disaster response / reconnaissance teams** — a georeferenced 3D model from a single pass means a
  usable map minutes after a flight, without a multi-strip survey mission.
- **Infrastructure & terrain mapping** (roads, facades, vegetation) — one flight, six standard formats,
  straight into existing GIS/CAD/game-engine tooling (QGIS, ArcGIS, Blender, Unreal) with no
  conversion step.
- **Any team without RTK-grade GPS or extensive ground control** — the system states its own accuracy
  limits (e.g. "3.35 m from GPS") instead of asserting survey-grade truth it cannot deliver.

### Benefits
| Type | Benefit |
|---|---|
| **Operational** | Single flight pass — no repeated survey flyovers; usable output in minutes, with the tool forecasting exactly how many |
| **Economic** | Removes the need for RTK/PPK hardware or dense ground-control-point networks to get a usable model; runs on a single consumer/cloud GPU |
| **Transparency** | Every model ships with a machine-generated quality report — coverage %, per-zone accuracy, and which surfaces are measured vs inferred — so downstream users know exactly how much to trust each part of the model |
| **Safety / disaster response** | Faster situational 3D maps of affected areas without additional flight risk beyond the single pass already flown |
| **Environmental** | Fewer flights per site = lower drone battery/fuel use and less repeated overflight of sensitive areas (wildlife, disaster zones) |

---

## Slide 6 — Research and References

- **Problem statement:** SIH26158 (PS-17), National Technical Research Organisation (NTRO)
- **Camera pose estimation / structure-from-motion:** COLMAP — Schönberger & Frahm, *Structure-from-
  Motion Revisited*, CVPR 2016
- **Feed-forward depth/geometry model:** VGGT (Meta AI Research / FAIR), used here as VGGT-Ω for
  per-frame depth, anchored and validated against COLMAP-measured geometry before any point is fused
- **Dense multi-view stereo & texturing:** OpenMVS (open-source MVS pipeline)
- **Reference ground truth used for accuracy validation:** USGS 3DEP lidar (public, used to
  independently verify the Esri flight's reconstructed surface: 2.87 m RMS within 100 m tiles)
- **Our own working prototype (primary evidence):**
  https://ritesh14g.github.io/Single-Pass-3D-Model-Generation/
- **Source code:** https://github.com/ritesh14g/Single-Pass-3D-Model-Generation

---

## Quick figure lookup (so nobody misquotes a number)

Copy numbers **exactly** as shown here or from the live Quality report — do not round differently
across slides.

| Figure | Esri flight | DJI_0047 flight |
|---|---|---|
| Video length | 1 min 41 s | 1 min 44 s |
| Total processing time (measured) | 7 min 6 s | 44 min 14 s |
| Coverage (ground reconstructed) | 69.3% | 88.0% |
| Cameras vs GPS (RMS) | 3.35 m | 2.05 m |
| Shape accuracy vs lidar, Zone 1 (RMS / NMAD, 100 m tiles) | 2.87 m / 2.40 m | not available (no local lidar) |
| Output formats produced & re-verified | 6 / 6 | 6 / 6 |
| Recon quality score | 97.2 / 100 | 97.1 / 100 |
| Projected time for a 10-min video (§9 target: 15 min) | ~42 min | ~256 min (highest-detail flight; see speed notes) |

**A note on the two projected-time figures:** DJI_0047 was flown much lower and slower than Esri, so it
keeps ~3× more frames for the same coverage and is correspondingly more detailed *and* slower — say
this on the slide if you quote the 256-minute figure, so it reads as "a harder, more detailed case,"
not as a contradiction of the 42-minute figure above it.

---

## Before you submit
- [ ] Exactly 6 slides (title + 5), matching the template's headings verbatim
- [ ] Every number on the slides matches this guide or the live site's Quality report
- [ ] The live link (https://ritesh14g.github.io/Single-Pass-3D-Model-Generation/) appears at least
      once, ideally on Slides 2 and 4
- [ ] "Important Instructions" slide deleted
- [ ] Exported as **PDF**
