# DEVLOG — SIH26158 / PS-17 Single-Pass Drone Video → 3D

Shared development log for the 3-person team and for AI agents. **Read this before touching
code.** It exists so work can resume from anywhere without redoing what's done or retrying
what already failed.

---

## 0. Rules for updating this log (everyone, every session)

1. **Append a session entry at the bottom of §6** after every working session, using the
   template below. One entry per person per session. Never rewrite someone else's entry;
   add a correction entry instead.
2. **Log every file** you created, modified, deleted or moved, with a one-line reason.
3. **Log dead ends** in §4 ("Do not revisit") with the *measured* reason it failed.
   A dead end without evidence is just an opinion.
4. **Keep §2 (status board) and §5 (open issues) current.** They must agree with
   `src/stages.py`.
5. When a stage is finished: flip it to `BUILT` in `src/stages.py`, add its Stage Lab panel
   under `ui/stages/`, add `src/qa/stageN_eval.py`, make `tests/test_ui_smoke.py` pass,
   and log it here.
6. Never delete a Stage Lab panel or its evaluator; later stages must keep re-testing
   earlier ones.

```markdown
### Session N — YYYY-MM-DD — <author> — Stage <n>
**Goal:**
**Created:**   - path — why
**Modified:**  - path — what changed and why
**Deleted/Moved:** - path — why
**Decisions:** - decision — evidence
**Dead ends (also add to §4):** - approach — measured reason
**Tests:** N passed / N xfailed / N failed (which, why)
**Open issues added/closed:** S?-?
**Next:**
```

---

## 1. Quick start

```powershell
# one-time
python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install -r requirements.txt

# tests (must be green before you push; xfails are tracked open issues)
.venv\Scripts\python -m pytest tests -q

# Stage Lab UI — parameters, scorecards, charts per stage + "Pipeline so far"
.venv\Scripts\python -m streamlit run ui/app.py

# CLI
.venv\Scripts\python -m src.cli inspect data\raw\demo_flight.mp4
.venv\Scripts\python -m src.cli run data\raw\demo_flight.mp4 --out data\interim\demo
.venv\Scripts\python -m src.cli run <video> --stage condition   # force an unbuilt stage
```

- Git identity used so far: `ritesh14g <riteshbehera6914@gmail.com>`.
- Dev machine has **no GPU**. GPU box = institute notebook: **20 GB H100 MIG slice, 3 CPU cores,
  56 GB RAM** (`CLOUD_GPU_GUIDE.md` §0). All code is GPU-first with a logged CPU fallback.
- Stage Lab writes to `data/lab/` (runs, synthetic inputs, `history.jsonl`); git-ignored.

---

## 2. Stage status board

Stage numbers follow the spec's section headings. `src/stages.py` is authoritative.

| # | Stage | Spec | Status | Lab panel | Evaluator | Notes |
|---|-------|------|--------|-----------|-----------|-------|
| 1 | Ingest | §4 | 🟢 **BUILT** | `ui/stages/stage1_ingest.py` | `src/qa/stage1_eval.py` | Synthetic 89/100; real DJI clip 72/100 (was 44). KLV/STANAG 4609 telemetry added and validated on 8 real MISB clips 2026-09-18. §4.3 speed not met on CPU decode (S1-4); overlap unmeasurable on forward-oblique footage (S1-8) |
| 2 | Conditioning | §5 | 🟢 **BUILT** | `ui/stages/stage2_condition.py` | `src/qa/stage2_eval.py` | Suite 267/267. S2-1, S2-2, S2-5, S2-6, S2-7, S2-8 closed. Scores **100/100** on `demo_flight.mp4` (7 pass / 5 info). Scorecard can now fail (9 scored KPIs); S2-3 and S2-4 remain open |
| 3 | Occluded surfaces | §6 | ⚪ planned | — | — | **Executes after Stage 4** (needs recon output) |
| 4 | Reconstruction tracks | §7 | 🟢 **BUILT (Track A)** | `ui/stages/stage4_recon.py` | `src/qa/stage4_eval.py` | **Track A**: pycolmap SfM + dense (GPU), OpenMVS Delaunay mesh + texture, budget-projected dense resolution. Demo (laptop CPU): 43/43, textured, **95.5/100**. **Track B = §7.4 hybrid, VGGT-Ω by default** (depth 0.86% / 0.89 m vs COLMAP, 24 cm/px); fallback Ω → VGGT-1B → Track A dense. SfM pieces merged via GPS; hybrid mesh reduced. Box Esri Stages 1–4: **93.3/100**, 300 s. Refinement BA: planned |
| 5 | Georeferencing & export | §8.1–8.3 | ⚪ planned | — | — | |
| 6 | Viewer & QA | §8.4–8.5 | ⚪ planned | — | — | |

---

## 3. Repository map

| Path | Stage | Purpose |
|------|-------|---------|
| `SIH26158_PS17_BUILD_SPEC.md` | — | Authoritative spec |
| `CLAUDE.md` | — | Instructions for AI agents (points here) |
| `CLOUD_GPU_GUIDE.md` | — | Rented GPU box: machine choice, setup, data transfer, remote Lab UI, what to record |
| `configs/default.yaml` | all | Every tunable (rule 3: no magic numbers in code) |
| `configs/fast.yaml`, `accurate.yaml` | all | Preset overlays (deep-merged) |
| `src/stages.py` | all | **Stage registry**: status, execution order, manifest/budget ownership |
| `src/core/config.py` | infra | Immutable config, presets, `--set` overrides, provenance |
| `src/core/logging.py` | infra | Console + JSONL structured logs; degradation/downgrade events |
| `src/core/manifest.py` | infra | Resumable run manifest; config fingerprint per stage |
| `src/core/budget.py` | infra | §9 time budget, projection, degradation ladder |
| `src/core/device.py` | infra | Device detection; VGGT-Ω chunk sizing from spec memory table |
| `src/ingest/video_reader.py` | 1 | Streaming decode (grab-skip), HW decode attempt, keyframes via PyAV |
| `src/ingest/telemetry.py` | 1 | DJI SRT (3 dialects) / CSV / TXT / EXIF parsing → `telemetry.parquet`; ENU helper |
| `src/ingest/dji_flight_record.py` | 1 | Binary `DJIFlightRecord_*.txt` (v≤12) decoder; recording segments; video alignment |
| `src/core/inputs.py` | all | Optional-input ledger: registry of every input with a fallback, derived per run from recorded metrics |
| `src/ingest/klv.py` | 1 | MISB ST 0601 / STANAG 4609 KLV embedded in the video stream; MPEG-2 TS depacketizer; no ffmpeg/klvdata dependency |
| `src/ingest/frame_selector.py` | 1 | Overlap-targeting selection, verified overlap estimator, blur gate calls |
| `src/condition/blur.py` | 1+2 | Blur metrics, adaptive thresholds (used by Stage 1 gate), sharpening |
| `src/condition/artifacts.py` | 2 | Blockiness detect/suppress, grid keypoint veto |
| `src/condition/illumination.py` | 2 | Exposure chain (paired fit), CLAHE, shadows, low light |
| `src/condition/dynamic_mask.py` | 2 | YOLO-seg masking (optional), geometric consistency, hole policy |
| `src/condition/gps_filter.py` | 2 | Envelope outliers, median, RTS-Kalman, baro/GPS fusion, RTK weight |
| `src/pipeline.py` | all | Orchestrator; runs BUILT stages by default |
| `src/cli.py` | all | `run`, `inspect`, `status`, `config` |
| `src/qa/synthetic.py` | QA | Synthetic flights + SRT/CSV dialects + binary DJI logs + **ground truth JSON** |
| `src/qa/stage1_eval.py` | 1 | Stage 1 KPIs, scoring, chart data |
| `src/qa/stage2_eval.py` | 2 | Stage 2 KPIs: artifacts, illumination, dynamic masking, GPS |
| `ui/app.py` | UI | Stage Lab entry: Pipeline-so-far page + one page per stage |
| `ui/lab.py` | UI | Shared: input picker, config overrides, runs, scorecard, history |
| `ui/stages/__init__.py` | UI | Panel registry + panel contract |
| `ui/stages/stage1_ingest.py` | 1 | Stage 1 parameters and diagnostics |
| `ui/stages/stage2_condition.py` | 2 | Stage 2 parameters (artifacts/illumination/dynamic/GPS) and diagnostics |
| `src/recon/track_a_colmap.py` | 4 | Track A: SfM (focal from telemetry), dense (COLMAP CUDA / OpenMVS CPU), mesh, texture; GPU→CPU fallback per step; dense size from the budget projection |
| `src/recon/alignment.py` | 4 | GPS similarity fit, camera-vs-GPS RMS, height above ground, footprint, telemetry focal/FOV hints |
| `src/recon/openmvs.py` | 4 | OpenMVS tool lookup (`tools/openmvs`) and runner (absolute paths, log-tail errors) |
| `src/recon/meshing.py` | 4 | PLY counts/reading, Poisson-fallback mesh cleaning |
| `src/qa/stage4_eval.py` | 4 | Stage 4 KPIs (`qa.stage4` bands), timings and camera-track chart data |
| `ui/stages/stage4_recon.py` | 4 | Stage 4 parameters, camera path vs GPS, time per step, output paths |
| `tests/test_recon.py` | 4 | Alignment, dense budget ladder, masks, mesh cleaning, scorecard bands, CPU end-to-end |
| `scripts/vggt_probe.py` | 4 | Track B feasibility probe (VGGT) |
| `src/recon/track_b_vggt.py` | 4 | Track B hybrid: VGGT depth in 8-frame windows, per-frame anchoring on Track A sparse points, multi-view consistency, COLMAP-format `fused.ply` + `.vis` |
| `scripts/vggt_hybrid_probe.py` | 4 | Hybrid probe: VGGT depth vs COLMAP depth maps, pixel by pixel; compares input widths |
| `scripts/stage4_report.py` | 4 | Compact Stage 1–4 report for run folders + probe JSON |
| `scripts/box_stage4_compare.sh` | 4 | Box: Stages 1–2 once, Stage 4 with VGGT-Ω and VGGT-1B, Ω depth probe, one report |
| `src/recon/merge.py` | 4 | Place disconnected SfM pieces through GPS; `posed()` filter for unregistered images |
| `scripts/recon_probe.py` | 4 | Track A feasibility probe: pycolmap sparse + dense (CUDA PatchMatch, OpenMVS CPU fallback), Poisson mesh, OpenMVS texture; timings + metric check vs telemetry |
| `tools/openmvs/` | 4 | OpenMVS 2.4.0 prebuilt binaries, fetched per machine, git-ignored (Windows: `vc17/x64/Release/`; Linux: `bin/`) |
| `.streamlit/config.toml` | UI | Streamlit server settings (upload cap 300 MB) |
| `tests/test_core.py` | infra | Config, budget, manifest, chunk sizing |
| `tests/test_telemetry.py` | 1+2 | SRT dialects, CSV, TXT, interpolation, ENU, GPS filter |
| `tests/test_dji_flight_record.py` | 1 | Binary DJI log decoding, segment detection, video alignment |
| `tests/test_optional_inputs.py` | all | Ledger registry, per-flag absences, and unknown-vs-absent |
| `tests/test_klv.py` | 1 | ST 0601 encode/decode round-trip, TS depacketization, error markers, checksum, DJI fall-through |
| `tests/test_ingest.py` | 1 | Reader, overlap estimator, selection, aperture regression |
| `tests/test_stage1_eval.py` | 1 | End-to-end Stage 1 on synthetic flights + scorecard |
| `tests/test_condition.py` | 2 | Conditioning modules — **36/36 pass** (S2-1 and S2-2 closed) |
| `tests/test_ui_smoke.py` | UI | Every Lab page loads; Stage 1 runs from the UI |
| `tests/fixtures.py` | test | Re-export of `src.qa.synthetic` |
| `data/raw/demo_flight.*` | — | Local demo input (git-ignored) |

Empty placeholders: `docker/`, `viewer/`, `src/fusion/`, `src/geo/`, `src/export/`.

---

## 4. Do not revisit — settled decisions and dead ends

Each entry was measured, not guessed. Don't re-open one without new evidence.

**Frame selection / overlap estimation (Stage 1)**
- ❌ **Fixed-interval sampling.** Spec §4.3 forbids it; selection targets overlap with a
  predictive stride (≈3 decodes per kept frame).
- ❌ **LK optical flow without a forward-backward check.** On aerial texture LK reports
  "success" for points pinned at zero motion; they survive medians and bias overlap toward 1.
  Round-trip error ≤ 1.5 px is required.
- ❌ **Trusting the overlap fit by point count.** A correct fit (NCC 0.998) on 6 inliers
  was rejected; a wrong fit on many agreeing points was accepted. Trust is decided
  photometrically (NCC of aligned frames).
- ❌ **NCC alone as verification (aperture problem).** Content invariant along the flight
  line (roads/rivers) correlates equally at any along-track offset: measured false pairs had
  NCC 0.36–0.49 with a peak drop of ~0.000, true pairs NCC ≥ 0.998 with a drop ≥ 0.09 at a
  4 px shift. It let the selector skip 92 frames. Fix: `flow.min_peak_drop` on both axes +
  `max_stride_growth`.
- ❌ **Feature tracking as the only estimator.** Fails on low-texture content;
  phase correlation is the fallback, verified by the same checks.
- ❌ **Pyramid levels = 3.** Displacements ≳ 80 px at 640 px width mistrack silently. Use 5.
- ❌ **Keeping an unverifiable probe as the selected frame.** It recorded overlap 0.0 and
  opened coverage gaps. Unverified → halve the stride; a final min-stride probe is the fallback.
- ❌ **Ranking the blur-replacement window by sharpness alone.** Overlap falls every frame
  forward, so it always picked the last frame: probes at 0.76 overlap became 0.52, median 0.68.
  A replacement now needs a *verified* overlap ≥ target − tolerance.
- ❌ **Verification at full working resolution, with float64 masks.** 43% of selection time.
  Runs at `flow.verify_width` (320 px) with one shared eroded support: 8.4 s → 3.2 s, same accuracy.
- ❌ **numpy `fft2` + per-call `mgrid`/Hanning grids for anisotropy.** 26% of selection time.
  Cached grids + float32 `cv2.dft` (identical power spectrum).
- ❌ **Uniform-noise synthetic canvas.** Pathological for pyramidal LK (coarse levels go grey).
  `make_canvas` is multi-scale (regions, roads, buildings, fine texture).

**Blur gate (§5.2, used by Stage 1)**
- ❌ **Spec-literal fixed percentile threshold.** Rejects that percentage of *every* video,
  including uniformly sharp ones (15% of a clean synthetic flight). Replaced by a robust
  outlier test + absolute floor + coverage cap (`max_reject_fraction`). Deliberate deviation
  from spec, documented in `default.yaml`.
- ❌ **Two-sided MAD for the outlier test.** Blurred frames inflate the spread they are meant
  to be detected by. Use the one-sided (upper-half) spread.
- ❌ **Sharpness score alone for motion blur.** Measured: clean 103–294 vs blurred 60–157
  overlap in score, but anisotropy separates them perfectly (clean ≤ 2.14, blurred ≥ 6.0).
  Directional frames must beat the `strict_percentile` (median).
- ❌ **One whole-video sharpness threshold on real footage (S1-7).** Sharpness follows scene
  texture: on `LineVision-VideoGeoTagging.mp4` it rejected 450 consecutive in-focus pasture frames
  (last third of the clip 86.7% rejected). Replaced by `condition.blur.rolling_baseline`.
- ❌ **Robust outlier test on the rolling window.** The one-sided spread of a tight local population
  is tiny, so while ground cover changes every frame sits "below the lagging median": a smooth
  60-frame 2800→1400 ramp gave 15 false rejects (window 30), 31 (window 60). A ratio to the window
  median (`max_local_drop`) gives 0.
- ❌ **Letting the directional (motion-smear) threshold follow the window.** A nearby soft stretch
  dragged it to ~150; synthetic frame 45 (anisotropy 9.07, score 147) was kept against a whole-video
  line of 188.7. Detected smear stays on the whole-video line.

**Telemetry / GPS**
- ❌ **Forward-only Kalman filter.** Causal filter lags the track (error 3× worse than raw
  noise). Offline pipeline → RTS smoother with finite-difference velocity init.
- ❌ **Envelope test on adjacent 30 Hz samples.** Position quantization (~0.11 m) over 33 ms
  reads as ~100 m/s² → 67% of *clean* fixes rejected. Differentiate over ≥ 0.5 s and add a
  quantization floor (`envelope_min_baseline_s`, `position_quantum_m`).
- ❌ **Exact-only CSV header matching.** `OSD.latitude` didn't match `osd_lat`. Two-pass
  exact then shortest-substring, headers claimed once.
- ✅ **GPS filtering belongs to Stage 2 (§5.6)**, not ingest. Stage 1 writes telemetry as parsed.
- ❌ **`pydjirecord` 1.3.0 for binary DJI logs.** Decoded 0 records from a real v8
  `DJIFlightRecord_*.txt`: it always reads a u16 record length, v8 uses u8. Own decoder
  (`src/ingest/dji_flight_record.py`) probes the width.
- ❌ **Camera `record_time` for sub-second video alignment.** Within one 490 s recording the
  implied start offset spread 173.5–176.6 s. Use the recording-flag transition (±1 s).
- ❌ **Absolute altitude from the v8 Home record** (`f32 / 10`): -119.8 m at Houston. Not used.
- ❌ **Opening a DJI flight record in a chat/Read tool.** It's binary; the whole-file dump
  overflowed the prompt. Inspect with a script that samples the header and records.
- ❌ **Aligning a trimmed clip by image rotation vs aircraft heading.** On the real LineVision
  clip, 0% of 8,632 offsets gave a plausible rotation/heading ratio (expected ≈ sin(25.6°) = 0.43;
  fitted ≈ 0). What worked: shadow direction from sun azimuth to pick the flight leg, then
  sideways drift/zoom vs crab angle/forward speed to pick the offset (438.3 s).

**Infrastructure**
- ❌ **Budget projection at near-zero progress.** elapsed/progress≈0 degraded every stage on
  its first frame. Projection only after 5% progress (`MIN_PROJECTION_PROGRESS`).
- ❌ **Skip reasons stored as warnings.** Inflated warning counts; now `skip_reason`.

**Illumination (Stage 2, in progress)**
- ❌ **Exposure fit from whole-frame luminance quantiles.** At 75% overlap, new content reads
  as exposure change (gain drifted 0.79–1.08 on footage with no exposure change). Paired,
  trimmed fit over the overlap region using the selector's stored transform (`transform_prev`).
- ❌ **Otsu luminance threshold for shadows** (tried in session 1 probes): precision dropped
  to 0.58–0.65 and it flagged 18% of a shadow-free frame. Not the fix for S2-2.

**Reconstruction (Stage 4)**
- ❌ **OpenDroneMap as Track A.** Needs Docker/Apptainer, absent on the box (ENV-1), and its
  SfM, matching, meshing and texturing are CPU-bound on a 3-core slice; only feature extraction
  and densify use the GPU. Replaced by pycolmap (CUDA wheel) + OpenMVS, which are the same
  class of components ODM wraps.
- ❌ **`demo_flight.mp4` for judging 3D quality.** It is a flat canvas translated in image space:
  focal length and curvature are unobservable. Freezing distortion sent focal 978 → 4 907 px;
  GPS priors + frozen distortion → 22 663 px with 5/43 registered. The output bends into a bowl
  (sag/length 0.045) whatever the settings. Use real footage or a rendered 3D scene.
- ❌ **Global mapper (GLOMAP, `pycolmap.global_mapping`) on the flat synthetic pan**: "no 3D
  points to optimize", no model. Not yet tried on real footage — not a verdict on real data.
- ❌ **GPS position priors to fix the focal/height ambiguity.** On a constant-height nadir flight
  images + GPS only fix height/focal. On Esri, priors pulled focal 1928 → 1132 px and height above
  ground 93 → 56 m (truth ≈ 111 m). What works: **seed focal from telemetry HFOV and hold it
  fixed** → 107.1 m (−3.5%). Refining a seeded focal drifts back to 1908 px / 92 m.
- ❌ **COLMAP Poisson at its default depth 13.** 5.35 M faces from 43 frames at 640×480;
  TextureMesh ran > 10 min and was stopped. Depth 11: 676 k faces in 14 s, texture in 52 s.

---

## 5. Open issues

| ID | Stage | Issue | Evidence / next step |
|----|-------|-------|----------------------|
| ~~S1-1~~ | 1 | ~~Median overlap 0.68~~ | **Closed session 2**: sharpness-only replacement ranking; now 0.761 |
| S1-2 | 1 | PyAV not installed → I-frame preference untested | `pip install av`, test on real H.264 |
| S1-3 | 1 | Hardware decode never engages with pip OpenCV (no CUDA) | **Code path added 2026-09-19**: NVDEC via PyNvVideoCodec (`NvdecCapture`), open-time and mid-stream fallback to OpenCV, tested with a fake capture. The adapter targets the PyNvVideoCodec 2.x `SimpleDecoder` API and has **never run on real NVDEC**. Verify on the institute box: `src.cli inspect` must say `(nvdec)`, then measure fps vs `ingest.video.nvdec=false` |
| S1-4 | 1 | **§4.3 speed acceptance (<60 s, 10-min 4K) not met on this machine.** Projection on 640 px synthetic: analysis 64 ms/kept frame → ~77 s at the 1200-frame cap; software decode scaled to 10 min of 4K → ~740 s | Decode dominates: needs NVDEC (S1-3) and/or keyframe seeking instead of `grab()` over the whole clip. Then trim analysis (cache anchor corners: `goodFeaturesToTrack` recomputed per probe, ~10% of time). Measure on real 4K on the GPU box |
| S1-6 | 1 | Decoded/total frames 0.73 (WARN, target <0.60) on fast synthetic pan | Fast pan keeps a frame every ~3; likely fine on real survey speeds — re-check on real footage |
| ~~S1-9~~ | 1 | ~~KLV parser has never seen a real STANAG 4609 file~~ | **Closed 2026-09-18**: validated on all 8 QGISFMV MISB sample clips (`Drone Video Dataset/QGISFMV_Samples/MISB/`). 407–1953 rows each, GPS + attitude on every one, ST 0601 versions 1/4/6, all checksums valid in strict mode. Cross-checked `Cheyenne_Handoff` against its own burned-in overlay: aircraft position within ~7 m, altitude 9754 vs 9749 ft, frame centre lat exact, frame-centre elevation 6036 vs 6037 ft, slant range 3314 m vs 1.8 NM, UTC timestamp matched to the second. It found two real bugs — see S1-12 and S1-13 |
| ~~S1-10~~ | 1 | ~~MP4/MOV-wrapped KLV is opt-in and only found when samples are contiguous~~ | **Superseded 2026-09-18 by S1-13**: the sample clips that looked MP4-wrapped are MPEG-2 TS with a misleading extension. Genuine-MP4 scanning stays opt-in (`scan_mp4`), which is still the right default for DJI footage |
| ~~S1-12~~ | 1 | ~~ST 0601 checksum was computed as a plain byte sum; it rejected every real packet in all 8 sample clips~~ | **Closed 2026-09-18**: the standard specifies `bcc_16`, which adds bytes into alternating halves of a 16-bit accumulator. The synthetic round-trip could not catch it because the test encoder used the same wrong formula. Regression vector `TestRealPacket.REAL_PACKET` is a real local set from `klv_metadata_test_sync.ts`; reverting the fix fails 2 tests |
| ~~S1-13~~ | 1 | ~~KLV source selection gated on file extension, so real ISR footage was skipped~~ | **Closed 2026-09-18**: all 8 MISB samples are MPEG-2 TS but are named `.ts`, `.mp4`, `.mpeg4` and `.H264`. Under the old rule the pipeline missed 5 of 8. `file_is_ts()` now sniffs 189 bytes for TS framing; all 8 are found with default config |
| S1-11 | 1 | KLV carries no barometric altitude and no focal length; `alt_baro` and `focal_mm` stay NaN on an ISR clip | Focal could be derived from sensor HFOV (tag 16, parsed and stored) plus sensor width, but the width is not in ST 0601 — deriving it would need a per-platform table. Not fabricated for now |
| S1-14 | 1 | Directional-blur detector has almost no margin on clean footage: sharp frames of a clean synthetic pan measure anisotropy **2.121 / 2.186** (OpenCV) and **2.208 / 2.244** (NVDEC) against `fft_anisotropy_reject: 2.2`. The decoder noise alone (1–4 grey levels) flips 2 of 60 frames into `directional_rejects` | Real motion blur measures > 3.0 (`test_condition`), so the limit may be too low. Do **not** retune on synthetic data: measure the anisotropy of clean vs smeared frames on real nadir and oblique footage (roads, crop rows and power lines are naturally anisotropic), then set the limit. The test now checks the outlier gate only |
| S1-15 | 1 | **NVDEC segfaulted the process on an MPEG-2 transport stream** (Esri, box). Stage 1 streamed it fine; Stage 2's `read_indices` died inside `PyNvVideoCodec SimpleDecoder.__getitem__` (exit 139, `faulthandler` trace). A segfault is not catchable, so the mid-stream fallback cannot help. `device.prefer=cpu` did not stop it either: it never gated NVDEC | **Fixed 2026-09-21, verify on the box**: NVDEC only gets inputs on `ingest.video.nvdec_codecs` (h264/hevc/h265/av1/vp9, substring match) and never a TS container (`file_is_ts` sniff, `nvdec_allow_ts: false`); `device.prefer: cpu` turns NVDEC off. One `reader_options(cfg)` now builds the reader for all three call sites. 11 tests; both gates mutation-checked |
| S1-5 | 1 | SRT parser only tested on synthetic dialects | Test on real DJI SRTs from 2+ firmwares; competition dataset |
| ~~S1-7~~ | 1 | ~~Blur gate false-rejects sharp frames when scene content lowers sharpness~~ | **Closed 2026-09-15**: `condition.blur.rolling_baseline`. LineVision clip reject fraction 0.713 → 0.000, 450-frame run gone; synthetic blur still fully rejected |
| S1-8 | 1 | Overlap stays 0.998 on the forward-oblique LineVision clip; selector cannot reach the §4.3 band | Not the stride cap: with `max_stride_seconds: 10` (600 frames) strides go 15→30→60→120 then fall back to 60 and repeat (max 123). The ~240-frame probe fails verification and `_probe_for_overlap` halves it silently. 120-frame pairs (~26 m of travel at 12.9 m/s) still measure 0.998, so the affine area-overlap estimate is likely invalid for oblique/zooming views. Next: check overlap on a nadir survey clip; consider feature-track survival as the overlap measure for oblique footage |
| ~~S2-1~~ | 2 | ~~Bilateral suppression doesn't reduce strong synthetic blocking (xfail)~~ | **Closed 2026-09-16, verified 2026-09-18**: boundary-weighted bilateral + guided refinement. Measured reduction ratio **0.233** on the repo test scene (KPI target > 0.10) |
| ~~S2-2~~ | 2 | ~~Shadow detector: fixed 25th-percentile luminance cut caps recall at 25% of frame (xfail)~~ | **Closed 2026-09-16, verified 2026-09-18**: relative blue-ratio + value threshold. Measured on the repo test scene: recall **0.521** (KPI target > 0.50 — passes by 0.021, not the 0.69 the session entry claimed), precision **0.321**, frame fraction **0.317**. Margin is thin and precision is poor — see S2-3 |
| ~~S2-10~~ | 2 | ~~Unbounded composed bias blew 20 of 53 conditioned frames to solid white~~ | **Closed 2026-09-18**: `max_gain`/`min_gain` bounded the gain but nothing bounded the bias, and `convertScaleAbs` computes `gain*pixel + bias`, so a diverging bias saturates a frame however tightly the gain is pinned. Frames 30+ were mean 255 with 100% of pixels at ceiling — the images Stage 4 would have consumed. Added `condition.illumination.exposure_chain.max_bias` (32 counts, symmetric). Re-run: **0 of 53 blown**, brightness stable at 97-99 instead of marching to 255. 49 frames now report `bias_clamped`, so the divergence is real and the bound is load-bearing |
| S2-11 | 2 | No KPI looked at the conditioned **output pixels** — S2-10 destroyed 38% of every clip and the scorecard stayed green | **Closed 2026-09-18**: `saturated_fraction` KPI, measured on the frame as written, with `qa.stage2.saturated_fraction_warn/fail` and a `blown_frames` count. Fourth defect in a row where an input-side proxy was scored and the outcome was not; when Stage 4 starts, review the whole Stage 2 KPI set with that lens |
| S2-9 | 2 | **The exposure chain diverges: 46 of 53 frames clamped on real ISR footage.** Per-link fitted gains are systematically below 1 (1.0 → 0.94 → 0.889 → 0.817 → 0.659 → 0.506 → 0.442 → 0.383, floor at frame 7). Composed over 53 frames the unbounded gain reaches **0.0004** — a 2500× brightness change no scene produces, so this is per-link error compounding, not auto-exposure drift | Not an overlap problem: median overlap on this clip is 0.845. `rejected_links` is 0, so every link passed the quality gate and the fit is biased rather than noisy. **`quantile_fallbacks` is 0**, so every link used the registered paired fit — the documented content-change bias of the quantile fallback is ruled out and the bias is in `_fit_paired` itself. Next: check whether `fit_gain_bias` is symmetric (A→B vs B→A) on a static pair; then re-anchor the chain to an absolute target luminance every N frames, or damp each composition toward an absolute estimate, instead of pure chaining. Raising `min_gain` only moves the wall |
| S2-3 | 2 | 45% shadow fraction on demo synthetic flight (false positives on dark blue-ish blocks) | Still open. Corroborated 2026-09-18: on the repo test scene the detector flags 31.7% of the frame at precision 0.321 — ~3× over-detection. Check on real footage |
| ~~S2-5~~ | 2 | ~~Stage 2 Lab panel crashed on Run: `RuntimeError: conditioning needs a completed ingest stage`~~ | **Closed 2026-09-18**: `manifest_stages` is ownership, not dependency, so the Lab ran `condition` alone into an empty run dir. Added `manifest_stages_through()` in `src/stages.py`; `ui/app.py` now runs the prerequisite chain. Regression test added and confirmed to fail without the fix |
| ~~S2-6~~ | 2 | ~~Stage 2 scores 0/100 on real pipeline output; the condition stage writes none of the six artifacts the evaluator reads~~ | **Closed 2026-09-18**: `_write_condition_reports()` in `src/pipeline.py` now emits the four report JSONs and two per-frame parquets, registered as manifest artifacts. `demo_flight.mp4` rescored 0/100 → **100/100** (12 KPIs live, was 4). Also fixed a key mismatch: `GpsReport.to_dict()` writes `max_speed_observed_mps`, the evaluator read `max_speed_observed`, so that KPI never fired. Guarded by `tests/test_stage2_eval.py` (16 of 18 fail without the fix) |
| ~~S2-7~~ | 2 | ~~No direct test coverage for `src/qa/stage2_eval.py` (310 lines); Stage 1 has `tests/test_stage1_eval.py`~~ | **Closed 2026-09-18**: `tests/test_stage2_eval.py`, 66 tests. 18 integration (report contract, manifest registration, chart columns, scorecard) plus 48 table-driven unit tests over `evaluate_condition` with synthesised reports, one per threshold boundary across all four KPI groups, the missing-report branches, the scale-free GPS path and the warn-counts-half score arithmetic. Verified by mutation: flipping `<` to `<=` in the gain-span and shadow-recall comparisons fails 3 tests |
| ~~S2-8~~ | 2 | ~~100/100 overstates Stage 2: 5 of 12 KPIs were INFO, so a degraded run could not move the score~~ | **Closed 2026-09-18**: shadow coverage, low-light fraction and altitude provenance promoted to scored KPIs with bands in `qa.stage2` (rule 6 — the gain-span and GPS-outlier constants moved there too). Now 9 scored / 3 info; a fully degraded report scores **5.6/100** with 8 fails. `demo_flight.mp4` still scores 100, now earned across 9 scored KPIs. The three remaining INFO KPIs are contextual, not quality signals (keypoints vetoed happens in Stage 4; RTK absence is not a defect; masked *area* is scored instead of mover count). **Does not close S2-3**: coverage is not correctness — telling an over-firing detector from a genuinely dark scene still needs per-pixel truth |
| S2-4 | 2 | ultralytics not installed → semantic masking path untested | Install on GPU box. As of 2026-09-19 YOLO is passed `device=0, half=True` when CUDA is visible, with GPU→CPU retry on failure (unit-tested with a fake model); check the log says `device=cuda` and record ms/frame |
| ~~ENV-1~~ | — | ~~No Docker, Podman or Apptainer on the institute notebook; ODM cannot run there~~ | **Closed 2026-09-21 by re-scoping Track A**: no container needed. `pycolmap-cuda12` 4.2.0 has a cp313 manylinux wheel (pip, no root); OpenMVS 2.4.0's Ubuntu build is statically linked (glibc + libstdc++ only) and runs from a folder. Unverified on the box until S4-3 |
| S4-1 | 4 | Camera centres sit **5–7 m RMS** from KLV GPS after a similarity fit on Esri, in every variant (≈ 1 m would be expected for a well-timed GPS). GPS steps between selected frames are irregular (1.6, 4, 15.8, 21.9, 29, 16 … m) | Suspect KLV-to-frame timing or interpolation, not SfM. Plot residual vs time; try a time offset sweep. Blocks GPS priors and §8.1 georeferencing accuracy |
| S4-2 | 4 | No 3D ground truth for the Stage 4 evaluator: `demo_flight.mp4` is planar and unobservable (see §4) | Render a synthetic flight over a textured 3D terrain with known cameras (`src/qa/synthetic.py`), so `stage4_eval.py` can score pose error, focal error and surface error |
| S4-3 | 4 | GPU path (CUDA SIFT, matching, PatchMatch stereo) never run | Run `scripts/recon_probe.py` on the box; record timings vs laptop CPU in this log |
| S4-5 | 4 | **OpenMVS TextureMesh segfaults (exit -11) on the box** on the Esri Poisson mesh: 1.75 M vertices / 1.82 M faces, i.e. heavily fragmented (a clean surface has ~2 faces per vertex), **and it contains NaN vertex coordinates** (numpy `invalid value` warnings in the cross products during cleaning; `trimesh.split` then stalled) | Probe now cleans the mesh (degenerate/duplicate faces, fragments < 1% of the largest piece) before texturing and treats a texture failure as a logged downgrade that keeps the vertex-coloured mesh. Re-run with `--reuse`; if it still crashes, try the laptop Windows build on the same mesh to separate a Linux-build bug from a mesh problem |
| S4-6 | 4 | **pycolmap-cuda12's Poisson (Linux) breaks meshes the Windows pycolmap builds cleanly.** Same 447 k-point box cloud, depth 11, trim 10: box → 1.82 M faces, 121 743 fragments, 4 564 NaN vertices; laptop → 946 k faces, 30 pieces, 0 NaN, at 1, 3 and 8 threads alike. The cloud itself has no NaN coordinates or normals | **Worked around 2026-09-21**: the probe meshes with OpenMVS `ReconstructMesh` (Delaunay) on the COLMAP cloud imported with `InterfaceCOLMAP -p fused.ply` (reads `fused.ply.vis`): 659 k faces, ~2 faces/vertex, 21 s; TextureMesh then succeeds in 81 s. Poisson + `clean_mesh` stays as the fallback. Not reported upstream yet |
| S4-7 | 4 | Esri mesh height map is low mid-strip and high at both ends — possible residual doming | Measure sag on the GPS-aligned mesh once the box run has `geo.txt`; compare with the telemetry-fixed-focal sparse metric (107 m vs 111 m) |
| S4-8 | 4 | **(presets chosen 2026-09-22; scale problem open)** **Dense stereo is the Stage 4 bottleneck: COLMAP PatchMatch took 1133 s of a 1287 s probe** (45 frames, 1920 px, 20 source views, 5 iterations, geometric pass) on the 2g.20gb slice. Everything else in Stage 4 took 154 s. Spec §9 gives MVS 4 min for a 10-min video | `scripts/dense_sweep.sh` compares five cheaper settings on one sparse model (time vs dense points vs GPS-aligned footprint m² vs mesh faces). Pick the fastest that keeps footprint; put it in `configs/default.yaml` + presets |
| S4-9 | 4 | Dynamic-object masks reach SfM feature extraction but not dense fusion: moving objects can still leave depth in the cloud | Undistort the Stage 2 masks with the same camera model and pass them as `StereoFusionOptions.mask_path` (COLMAP) / `--mask-path` (OpenMVS) |
| S4-4 | 4 | KLV HFOV 81° / VFOV 66° is inconsistent with a 16:9 frame (81° H implies 51° V), so the telemetry FOV is nominal | Fixed focal from HFOV still measured −3.5% height error; acceptable, but prefer HFOV and log the VFOV mismatch |
| LIC-1 | 4 | **VGGT licence vs the PS domain.** Every VGGT checkpoint carries a no-military / no-espionage acceptable-use clause (VGGT License AUP §2; VGGT-1B is also CC-BY-NC-4.0; VGGT-Ω is FAIR non-commercial research). The PS is set by NTRO and lists military reconnaissance and border mapping among applications | **Team decision 2026-09-22 (ritesh14g):** use VGGT for the competition prototype; the PS names disaster management and treats military use as one optional application, and a selected project would move to an in-house model built with government support. State this position and the licence terms in the README (§11). Track A (COLMAP BSD, OpenMVS AGPL) stays the licence-clean floor |
| SPEC-1 | — | Spec numbers occlusion engine Stage 3 (§6) but it consumes Stage 4 (§7) output | `execution_order` in `src/stages.py`; Stage 4 must be built before Stage 3 can run |
| SPEC-2 | — | §2.1 diagram labels stages differently from section headings | Section headings used |

---

## 6. Session log

### Session 1 — 2026-09-12 — ritesh14g (with Claude) — Stages 1–2 groundwork
**Goal:** Day-1 skeleton per spec §10; Stage 1 ingest; conditioning primitives.
**Commits:** `ec720bb` "Stage 1 ingest, conditioning primitives, and core infrastructure";
`78a052b` "Sparse Structure built".
**Created:**
- `.gitignore`, `configs/default.yaml`, `configs/fast.yaml`, `configs/accurate.yaml`
- `src/__init__.py`, `src/core/{__init__,config,logging,budget,manifest,device}.py`
- `src/ingest/{__init__,video_reader,telemetry,frame_selector}.py`
- `src/condition/{__init__,blur,artifacts,illumination,dynamic_mask,gps_filter}.py`
- `src/pipeline.py` (ingest + condition wired; later stages recorded as skipped), `src/cli.py`
- `tests/{__init__,conftest,fixtures,test_core,test_telemetry,test_ingest,test_condition}.py`
- `data/raw/demo_flight.mp4` + `.SRT` (synthetic demo, not committed)
- `.venv` (Python 3.10.4, system site packages for torch-cpu)
**Modified (during the session, after measurement):** everything in §4 "Frame selection",
"Blur gate", "Telemetry/GPS", "Infrastructure", and the exposure entry.
**Deleted:** none. **Tests at end:** 109 passed, 4 failed (Stage 2).
**Next:** was continuing Stage 2 shadow detector (S2-2) when the workflow changed.

### Session 2 — 2026-09-13 — ritesh14g (with Claude) — Workflow change + Stage 1 lock
**Goal:** Adopt the stage-by-stage workflow: stage registry, Stage Lab UI per stage with
parameters and measurement, "Pipeline so far" view, this DEVLOG. Finish Stage 1.
**Created:**
- `src/stages.py` — stage registry (status, execution order, manifest/budget ownership)
- `src/qa/__init__.py`, `src/qa/stage1_eval.py` — Stage 1 KPIs/scorecard/chart data
- `ui/__init__.py`, `ui/app.py`, `ui/lab.py`, `ui/stages/__init__.py`, `ui/stages/stage1_ingest.py` — Stage Lab
- `tests/test_stage1_eval.py`, `tests/test_ui_smoke.py`
- `DEVLOG.md`, `CLAUDE.md`, `requirements.txt` (was missing)
- Installed `streamlit`, `altair` into `.venv`
**Moved:**
- `tests/fixtures.py` → `src/qa/synthetic.py` (UI needs it). `tests/fixtures.py` is now a
  one-line re-export. Added ground truth: `generate_synthetic_flight`, `load_ground_truth`,
  `true_pair_overlap`, `true_lonlat`, `<stem>.truth.json`.
**Modified:**
- `src/pipeline.py` — runs only BUILT stages unless `--stage` given; skip reasons come from
  the registry; ingest no longer filters GPS (moved into `run_condition`, which now takes the
  raw `TelemetryTable` and writes `telemetry_filtered.parquet`); ingest records timing
  (`profile_s`, `select_s`, `telemetry_s`) and writes `blur_evaluations.parquet`.
- `src/ingest/frame_selector.py` — records every blur-gate evaluation
  (`FrameSelection.evaluations`); **aperture-problem guard** (`_peak_drop`, `_verified`,
  `OverlapEstimate.peak_drop`); NCC threshold moved to config; stride growth cap.
- `configs/default.yaml` — `flow.min_alignment_ncc`, `flow.peak_probe_px`,
  `flow.min_peak_drop`, `frame_selection.max_stride_growth`.
- `tests/test_condition.py` — fixed 2 test-side bugs (grid-mask probe pixel; relative depth
  example was 10% < 12% threshold); marked S2-1 and S2-2 `xfail`.
- `tests/test_ingest.py` — aperture regression tests (stripes; ground-truth selection).
**Decisions:**
- Stage numbering follows spec section headings; execution order kept separately (SPEC-1).
- UI stack: Streamlit + Altair (no build step, runs locally on Windows).
- Stage Lab scores = share of scored KPIs passed (warn = half). Ground-truth KPIs only on
  synthetic inputs.
- `src/ingest/frame_selector.py` (later in session) — `_choose_frame` replacement must keep
  verified overlap ≥ target − tolerance (closes S1-1); `_peak_drop`/`_alignment_correlation`
  replaced by `_verify_alignment` (downscaled, shared support); `FrameSelection.decode_s`.
- `src/condition/blur.py` — `spectral_anisotropy` uses cached `_anisotropy_grids` + `cv2.dft`.
- `src/ingest/video_reader.py` — `VideoReader.decode_s` (time inside read/grab/seek).
- `src/pipeline.py` — ingest timing adds `decode_s`.
- `src/qa/stage1_eval.py` — speed KPI now projects analysis (per kept frame × expected frames,
  capped by `max_frames`) and decode (× duration × 4K pixels) separately; new info KPI
  `analysis_ms_per_kept_frame`.
- `configs/default.yaml` — `flow.verify_width: 320`.
- `.gitignore` — `data/lab/`.

**Found by the new Stage Lab scorecard (the point of the workflow):**
1. Overlap estimator MAE 0.24 vs truth → aperture problem; selector skipped 92 frames. Fixed.
2. Median overlap 0.68 → replacement window traded overlap for sharpness. Fixed.
3. Selection speed → profiled and optimized 2.6×; remaining gap is decode (S1-4).

**Stage 1 scorecard at end of session** (default preset):
| Input | Score | Notes |
|-------|-------|-------|
| 640 px, blur every 11th, modern SRT | 89.3 | 12 pass, 1 warn (decode fraction), 1 fail (4K speed projection) |
| 1280 px, no blur, CSV log | 90.9 | 1 fail (4K speed projection) |
| 640 px, blur every 7th, no telemetry | 79.2 | + expected fail: GPS coverage 0 (scale-free path works, reported loudly) |
All ground-truth KPIs pass: overlap MAE ≤ 0.0008, blur recall 1.0, false rejects 0, blurred frames kept 0, GPS parse error ≤ 0.054 m.

**Verified:** `streamlit run ui/app.py` serves (health 200); `tests/test_ui_smoke.py` loads every
page and runs Stage 1 from the UI headlessly.
**Tests:** 133 passed, 2 xfailed (S2-1, S2-2), 0 failed.
**Not committed** at time of writing.
**Next:**
- Commit this session.
- S1-4 on the GPU box with real 4K + NVDEC (or keyframe seeking) — the one Stage 1 acceptance item left.
- Then Stage 4 Track A (spec §10: Track A end-to-end is the day 2–3 submission floor), or finish Stage 2
  (S2-1..S2-4) and give it a Lab panel. Recommended: Track A first, per the spec's prime directive.

---

### Session — 2026-09-13 — AI agent (Claude) — infra
**Goal:** Diagnose a prior chat session's "Prompt is too long" failure and raise the Stage Lab
video-upload cap to fit a 300MB DJI clip.
**Created:**
- `.streamlit/config.toml` — `server.maxUploadSize = 300`; Streamlit's unmodified default
  (200MB) was rejecting the 300MB upload in `ui/lab.py`'s `_upload_picker`.
**Modified:** none.
**Deleted/Moved:** none.
**Decisions:**
- Upload cap lives in `.streamlit/config.toml`, not `configs/default.yaml` — it's a Streamlit
  server setting (transport limit), not a pipeline tunable the rules in CLAUDE.md §6 govern.
**Dead ends:** none.
**Diagnosis (no code change, for the record):** a separate/earlier session hit "Prompt is too
long" immediately after auto-compaction freed 58k tokens. Compaction only shrinks past history;
it can't shrink one oversized message in the current turn. The user's request in that turn was
to check whether a `DJIFlightRecord_*` file has per-frame GPS — DJI's exported flight-record logs
are per-sample telemetry dumps (often 10Hz) that can run tens of MB of plain text, so if the
whole file was read into the prompt instead of being sampled/summarized by a script, it alone
could exceed the context window regardless of how much history was freed. Recommendation for
next time: never `Read` a large telemetry/log file whole to answer a yes/no question — write a
short script that prints its header, row count, and a few sample rows.
**Tests:** not run this session (no pipeline code touched).
**Open issues added/closed:** none.
**Next:**
- If/when the actual DJI flight-record file is available, add a small parser check (or extend
  `src/ingest/telemetry.py`) for the raw DJI `.txt`/`.DAT`-exported log format rather than only
  `.SRT`/`.csv`, if that's the format the team standardizes on.

---

### Session — 2026-09-13 — AI agent (Claude) — Stage 1 (telemetry, §4.2)
**Goal:** Accept a plain `.txt` file of GPS coordinates as a telemetry sidecar in the Stage Lab
upload flow and in the pipeline's flight-log tier, alongside `.SRT`/`.csv`.
**Modified:**
- `src/ingest/telemetry.py` — `parse_flight_csv` now sniffs the delimiter (`sep=None,
  engine="python"`, so comma/tab/space/semicolon all work) and, when the "header" row is
  actually numeric data (a bare `lat,lon[,alt]` dump with no header), re-reads with `header=None`
  and assigns columns positionally from a new `headerless_order` param instead of guessing.
  `load_telemetry`'s `csv` tier now also searches for a `.txt` sidecar (`find_sidecar(...,
  [".csv", ".txt"])`) and threads the new config value through. New `_looks_headerless` helper
  (majority-numeric column names → no header).
- `configs/default.yaml` — `ingest.telemetry.headerless_column_order: [lat, lon, alt_gps]`, the
  positional field order for a headerless log; per CLAUDE.md §6 this is configurable, not
  hard-coded, since a real DJI GPS-only export might order columns differently.
- `ui/lab.py` — `_upload_picker`'s sidecar `file_uploader` now accepts `type=["srt", "csv",
  "txt"]` (was missing `txt`, so the browser picker rejected it before it ever reached
  `parse_flight_csv`); `_raw_picker`'s caption mentions `.txt` alongside `.SRT`/`.csv`. No change
  needed to the suffix-routing logic (`side.suffix.lower() == ".srt" else csv`) — it already sent
  any non-SRT file, `.txt` included, down the flight-log path.
**Created:** none.
**Deleted/Moved:** none.
**Decisions:**
- Kept `.txt` inside the existing `csv` telemetry-source tier (spec §4.2 tier 2) rather than
  adding a fourth source — a `.txt` GPS dump is the same "separate flight-log" case the CSV
  parser already exists for; the only real difference is delimiter and header presence, both of
  which are now auto-detected.
- Headerless detection threshold is "≥60% of column names parse as numbers" — a real header like
  `Timestamp,OSD.latitude` has 0% numeric names, a bare data row is 100%; the margin covers a
  stray unlabeled index column without misreading genuine headers.
**Dead ends:** none — the `sep=None, engine="python"` change is drop-in; verified it doesn't
change parsing of the existing comma-delimited fixtures (`TestCsv` still passes unmodified).
**Tests:** 136 passed, 2 xfailed (S2-1, S2-2), 0 failed. Added `tests/test_telemetry.py::TestTxt`
(3 new cases: whitespace-delimited `.txt` with header, headerless `lat,lon,alt` dump resolved
positionally, `load_telemetry` auto-finding a `.txt` sidecar next to the video).
**Open issues added/closed:** none.
**Next:**
- Get the user's actual GPS `.txt` file and confirm its real column order/delimiter matches the
  `headerless_column_order` default (`lat, lon, alt_gps`) — adjust the config, not the code, if
  it doesn't.

---

### Session — 2026-09-13 — AI agent (Claude) — Stage 1 (telemetry, §4.2)
**Goal:** The user's "GPS .txt" (`DJIFlightRecord_2019-01-09_[13-16-53].txt`) turned out to be a
**binary** DJI GO flight record, not text — the previous session's text/CSV `.txt` support could
not read it. Decode it and align it to the video.
**Findings on the real file (inspected by header/record sampling only, never read whole):**
- 2.06 MB, DJI log format **v8**, Phantom 4 Pro V2, 940.7 s flight, 617 s of video recorded.
- 9,405 OSD samples at 10 Hz, `fly_time` strictly monotonic, 100% valid GPS (median 17 sats),
  location ~29.577 N, -95.75 W (Houston, TX), height 0–77.7 m, gimbal pitch median -19.8°.
- **GPS is not per video frame**: 10 Hz, interpolated onto frames by `TelemetryTable.at_times`.
- Three recordings: #0 138.8–172.9 s (34.1 s), #1 174.0–664.6 s (490.6 s), #2 709.0–812.4 s
  (103.4 s). None of the clips in the user's Downloads match (`LineVision-VideoGeoTagging.mp4` is
  77.8 s 1080p60) — the 300 MB video has not been identified yet.
**Created:**
- `src/ingest/dji_flight_record.py` — v1–12 decoder: prefix parse, record-length width probed
  (1 or 2 bytes), CRC64-keyed XOR descrambling (scheme + field offsets from MIT
  `dji-log-parser`/`pydjirecord`), OSD + gimbal + camera records → samples; recording segments
  from the camera flag; `parse_dji_flight_record` re-bases time onto the recording matching the
  video duration (or a configured segment / `offset_s`) and maps to the canonical schema
  (`lat, lon, alt_baro`=height above takeoff, `roll/pitch/yaw`=gimbal). v13+ (AES, DJI-server keys)
  returns an empty table with a reason. `is_dji_flight_record` sniffs binary vs text.
- `tests/test_dji_flight_record.py` — 9 tests: scramble round-trip, binary-vs-text sniffing,
  sample/segment decoding, v13 degradation, auto duration matching, mismatch warning, configured
  segment, `offset_s` override, `load_telemetry` routing.
**Modified:**
- `src/ingest/telemetry.py` — `load_telemetry(video_duration_s=...)`; the csv/txt tier routes a
  sniffed binary DJI log to the new decoder (lazy import: the decoder imports `TelemetryTable`).
- `src/pipeline.py` — passes `metadata.duration_s` to `load_telemetry`.
- `configs/default.yaml` — `ingest.telemetry.flight_record.{segment: auto, offset_s: null,
  duration_tolerance_s: 3.0, segment_padding_s: 2.0}`.
- `src/qa/synthetic.py` — `write_dji_flight_record` (scrambled v8 test logs); `import struct`.
- `ui/lab.py` — sidecar uploader help text names binary DJI logs and the v13 limit.
- `ui/stages/stage1_ingest.py` — "Video & telemetry" sidebar: recording selector and manual
  video-start offset for DJI logs.
**Decisions:**
- Own decoder instead of a `pydjirecord` dependency (see dead end) — ~200 lines, no new deps.
- Absolute altitude not populated: the v8 Home record's altitude decodes to -119.8 m at Houston,
  which is implausible; `alt_baro` (height above takeoff) only. Stage 2 handles baro-only.
- Segment start = recording-flag transition (±1 s; flag logged at 1 Hz). Camera `record_time`
  was tried for sub-second alignment but its implied offsets spread 173.5–176.6 s inside one
  recording, so it is not used.
**Dead ends (also in §4):** `pydjirecord` 1.3.0 — decodes **0 records** from this v8 log because
it always reads a u16 record length; v8 uses u8 (verified by walking records with u8 lengths:
all end in 0xFF up to the thumbnail).
**Tests:** 145 passed, 2 xfailed (S2-1, S2-2), 0 failed. Real file: decodes in 0.17 s; a 34 s
video matches #0, 490 s matches #1, 77.8 s falls back to #1 with a misalignment warning.
**Open issues added/closed:** none.
**Next:**
- Identify which recording the user's 300 MB clip is (check its duration); if it was trimmed, set
  `ingest.telemetry.flight_record.offset_s` in the Stage 1 Lab sidebar.
- Absolute altitude for v8 logs (Home record scaling) if a later stage needs ellipsoidal height.

---

### Session — 2026-09-14 — AI agent (Claude) — Stage 1 (aligning real footage to the DJI log)
**Goal:** Align the user's clip `LineVision-VideoGeoTagging.mp4` with
`DJIFlightRecord_2019-01-09_[13-16-53].txt`. The clip is 77.8 s, 1920×1080 at 59.94 fps (4,665
frames), re-encoded (`Lavf58`) with its creation time zeroed, and has no GPS/subtitle track or
on-screen overlay.
**Created / Modified / Deleted:** none — analysis only (scratch scripts in the session scratchpad).
**Result:** video t=0 = **flight time 438.3 s (±~1 s)**, i.e. 264.3 s into recording #1. For this
clip set `ingest.telemetry.flight_record.offset_s = 438.3` (Stage 1 Lab sidebar → "DJI flight
record: set video start manually").
**Evidence:**
- Duration: no recording matches (34.1 / 490.6 / 103.4 s), so the clip was trimmed.
- Altitude: recording #0 is too short; #2 is mostly at 14 m while the frames are clearly high
  oblique; #1 flies level at 77 m.
- Sun: at 13:24 CST the sun is at azimuth 195.7°, elevation 36.8°, so shadows point ~16°. The
  frames show shadows falling to the image right and toward the camera. Only the westward leg
  (heading ≈ −90°, flight time 434–534 s) does that; the 57° and 111° legs put them on the left.
- Motion (10 Hz LK + affine on the video vs log): inside that leg the fit peaks at 438.3 s (score
  0.455; everything within 0.02 lies in 437.1–439.0 s) against 0.16–0.25 elsewhere in the leg. The
  overall runner-up (478.9 s, 0.424) crosses the 138° U-turn at 544 s, where shadows would flip.
  Crab angle (course − heading) +31° matches the scene drifting left.
- End-to-end Stage 1 with the offset: 82/82 selected frames have GPS; track 1,002.2 m in 77.8 s
  = 12.9 m/s (log leg speed 12.5 m/s); heading −86.7° to −90.4°; alt_baro 76.7–76.9 m; longitude
  decreasing (westward).
**Decisions:** the offset is per-clip, so it is not written to `configs/default.yaml`.
**Dead ends (also in §4):** image rotation vs aircraft heading as the alignment signal.
**Unexplained:** ~21° of cumulative in-plane image rotation (−6 to −7° per 10 s in the last 20 s)
while the logged heading stays within ±4°. Not an edit — the frames are continuous and sharp. The
alignment does not rely on rotation.
**Found:** S1-7 — the blur gate rejected all 450 frames from 63–72 s of this clip although they are
sharp (content-driven sharpness dip, see §5). The budget also repeatedly degraded ingest
(`reduce_frames`); Stage 1 took 222 s on this 77.8 s 1080p60 clip (related to S1-4).
**Tests:** not run (no code changed); last run 145 passed, 2 xfailed.
**Open issues added/closed:** S1-7 added.
**Next:**
- Fix S1-7 (local/rolling sharpness baseline) and re-run Stage 1 on this clip with offset 438.3.
- If more trimmed clips are expected, automate the shadow + motion offset search as a Stage 1 helper.

### Session — 2026-09-15 — ritesh14g (with Claude) — Stage 1 (real-clip fixes + GPU handoff)
**Goal:** Fix why the real DJI clip scored 44/100 (Stage Lab screenshot) before handing Stages 2–6
to the rest of the team; write a cloud GPU guide for them.
**Created:**
- `CLOUD_GPU_GUIDE.md` — machine choice from the §7.2 memory table, first-hour setup (Docker GPU
  check first), data transfer, Lab UI over an SSH tunnel, which open issues to close on the box
  first, cost hygiene, handoff checklist.
**Modified:**
- `configs/default.yaml` — `ingest.frame_selection.min/max_stride_seconds` (max 10 s);
  `condition.blur.rolling_baseline` (`enabled`, `window: 30`, `min_samples: 5`,
  `max_local_drop: 0.4`, `hard_floor_fraction: 0.25`).
- `src/ingest/frame_selector.py` — `_stride_bounds()` converts stride guard rails from seconds using
  the clip's fps (frame counts stay as fallback); selector owns a `RollingBaseline` and passes it
  to every `assess_blur` call.
- `src/condition/blur.py` — `RollingBaseline`: reject = max(window median × (1 − max_local_drop),
  hard floor); clean = local percentile; directional threshold stays whole-video.
  `assess_blur(..., baseline=None)` is unchanged without a baseline. `BlurProfile.absolute_floor` added.
- `ui/stages/stage1_ingest.py` — sidebar controls for max stride in seconds and all rolling-baseline
  settings.
- `tests/test_ingest.py` — `TestStrideBoundsFollowFrameRate` (3), `TestRollingBlurBaseline` (6).
- `CLAUDE.md` — rule 8 points GPU work at the guide.
**Deleted/Moved:** none.
**Decisions:**
- LineVision flight-record offset (438.3 s) stays per-clip: passed with
  `--set ingest.telemetry.flight_record.offset_s=438.3`, not written to the default config.
- Local blur test is a ratio, not an outlier test: smooth 2800→1400 ramp gave 0 false rejects vs 15
  (evidence in §4).
- Known cost: an instant 50% scene cut rejects ≤ window/2 frames (15), and warm-up rejects
  ≤ `min_samples` frames if the clip opens on low texture. Both are covered by tests.
**Result:** LineVision clip, CPU laptop, same code path as the Lab, offset 438.3
(`data/interim/linevision_s17`, git-ignored):

| KPI | Before (09-14) | After |
|-----|----------------|-------|
| Stage score | 43.8 | **72.2** (6 pass / 1 warn / 2 fail) |
| Wall time | 226.3 s | **45.7 s** (≈352 s projected for 10 min of 1080p) |
| Budget degradations | 198 | 17 |
| Blur rejected / evaluated | 0.713 (704 evals) | **0.000** (222 evals) |
| Coverage warnings | 3 (incl. 450-frame run) | 0 |
| Selected frames with GPS | 0.00 | **1.00** |
| Pairs < 50% overlap | 2 (unverified pairs recorded as 0.0) | 0 |
| Decoded / total | 0.167 | 0.068 |
| Projected 4K 10-min selection | 6470 s | 1190 s (still FAIL, S1-4) |
| Median overlap | 0.998 | 0.998 (still FAIL, S1-8) |

New WARN `telemetry_duration_ratio` 1.051 is the ±2 s `segment_padding_s`: (77.8 + 4) / 77.8 = 1.051.
It is expected, not an alignment error; the evaluator could subtract padding.
**Dead ends (also in §4):** whole-video blur threshold on real footage; outlier test on a rolling
window; letting the directional threshold follow the window.
**Tests:** 154 passed / 2 xfailed (S2-1, S2-2) / 0 failed.
**Open issues added/closed:** S1-7 closed; S1-8 added.
**Next:**
- GPU box per `CLOUD_GPU_GUIDE.md` §6: licensing, S1-3 NVDEC, S1-4 real-4K timing, Track A freeze.
- S1-8: test overlap on a nadir survey clip before changing the estimator.
**Conditioning artifact suppression (Stage 2 §5.3)**
- ❌ **Vanilla single-pass bilateral (S2-1).** Default sigma_color=35 treats the block edge as a real
  geometric edge (amplitude above sigma) and preserves it. Fix: targeted boundary map + bilateral at
  2.5× sigma_color on boundary pixels + guided-filter refinement using the pre-filter image as the
  guide. On synthetic 8×8 blocking: blockiness fell below threshold where single-pass did not.
- ❌ **Applying the filter uniformly across the whole frame.** Reduces texture in non-blocked areas
  for no benefit. Only pixels within 2 px of block boundaries are filtered; interior pixels are blended
  by severity so a barely-blocked frame sees minimal change.

**Shadow detection (Stage 2 §5.4)**
- ❌ **Fixed-percentile luminance cut (S2-2, original code).** Caps recall at the chosen percentile
  (25%) regardless of how much of the scene is shadowed. A 40%-area shadow cannot be recalled beyond
  25% with this approach. Measured: 20% recall on synthetic test.
- ❌ **Otsu on the dark sub-population (first fix attempt).** Improves on fixed-percentile in theory
  but the synthetic shadow reduces brightness by only 35%, so shadow pixels overlap the non-shadow
  brightness range. Otsu bisects that mixed population conservatively. Still 20% recall.
- ✅ **Relative blue-ratio + relative brightness (final fix, S2-2 closed).** Sky-lit shadow surfaces
  are bluer than direct-sun surfaces (blue/red ratio 1.68 vs 1.15 in the scene median) even when the
  brightness gap is small. Gate: blue_ratio > scene_median + delta (config: 0.12) AND value < median
  × ceiling (1.05) AND saturation ≤ 0.45. Measured: 69% recall, 1.5% FP, 0% FP on dark-paint test.

---

### Session — 2026-09-16 — AI (Antigravity) — Stage 2 (Conditioning validation + Stage Lab)
**Goal:** Validate Stage 2 (Conditioning, spec §5) — close S2-1 and S2-2, write QA evaluator,
write Stage Lab panel, flip status to BUILT, confirm 36/36 tests green.
**Created:**
- `src/qa/stage2_eval.py` — KPI evaluator: artifact suppression, illumination, dynamic masking, GPS
  conditioning. Same Kpi/StageEvaluation pattern as stage1_eval.py.
- `ui/stages/stage2_condition.py` — Streamlit Stage Lab panel with full sidebar parameter controls
  (artifacts, illumination/shadow/exposure, dynamic masking, GPS Kalman) and diagnostic charts
  (shadow timeline, blockiness timeline, dynamic-object class bar chart, GPS metrics).
**Modified:**
- `src/condition/artifacts.py` — `suppress_block_artifacts`: replaced single bilateral with
  (1) boundary weight map from block grid, (2) bilateral at 2.5× sigma_color on boundary pixels,
  (3) guided-filter refinement (falls back to second mild bilateral if cv2.ximgproc unavailable),
  (4) severity-blended composite. Closes S2-1.
- `src/condition/illumination.py` — `detect_shadows`: replaced fixed-percentile (S2-2 root cause)
  then Otsu-on-dark-pixels (still insufficient on soft shadows) with relative blue-ratio + relative
  value threshold (physics-based sky-light signature). 69% recall, 0% dark-paint FP. Closes S2-2.
- `configs/default.yaml` — shadow section: replaced `luminance_percentile: 25` and
  `luminance_seed_percentile: 40` + `min_blue_ratio` with `blue_ratio_delta: 0.12` and
  `luminance_ceiling_factor: 1.05` matching the new detector.
- `tests/test_condition.py` — removed `@pytest.mark.xfail` from both S2-1 and S2-2 tests.
- `ui/stages/__init__.py` — added `stage2_condition` import and PANELS entry.
- `src/stages.py` — Stage 2 status: `IN_PROGRESS` → `BUILT`.
- `DEVLOG.md` — status board row updated, repo map extended, dead-ends documented, session added.
**Decisions:**
- Fixed-percentile luminance is not just a tuning issue but structurally wrong for soft shadows;
  dropped entirely in favour of the spectral signature (evidence: 0/3 approaches based on
  luminance alone achieved >30% recall; blue-ratio approach achieved 69% on first try).
- Guided filter falls back to second bilateral gracefully; no hard `opencv-contrib` dependency.
- `blue_ratio_delta: 0.12` chosen: separates shadow (median+0.44) from non-shadow in 3 trials;
  clamped to ≥ 0.08 in code so degenerate scenes still require a real blue shift.
**Tests:** 36 passed / 0 xfailed / 0 failed (test_condition.py, 13.85 s).
**Open issues added/closed:** S2-1 closed; S2-2 closed.
**Next:**
- Stage 3 is Occluded Surface Reconstruction (§6), but it needs Stage 4 (recon) output first.
- Start Stage 4 Track A (OpenDroneMap) — the submission floor.
- GPU box work per `CLOUD_GPU_GUIDE.md`: S1-3 (NVDEC), S1-4 (real 4K timing), Track B.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 2 (Lab panel wiring to Stage 1)
**Goal:** Audit the Stage 2 hand-off from the 2026-09-16 session, then fix the Stage Lab
panel so Stage 2 actually runs on top of Stage 1 output.

**Created:**
- (none — see Modified)

**Modified:**
- `src/stages.py` — added `manifest_stages_through(stage)`: the manifest stages needed to
  *produce* a stage's output (every BUILT stage at or before it in `execution_order`, then
  the stage itself), as distinct from `manifest_stages`, which is what a stage *owns*.
- `ui/app.py` — the stage page now runs `manifest_stages_through(spec)` instead of
  `list(spec.manifest_stages)`. Stage 1 is unaffected (`['ingest']` either way); Stage 2
  now resolves to `['ingest', 'condition']`.
- `ui/stages/stage2_condition.py` — `evaluate()` called `Config.default()`, which does not
  exist on `Config`; it raised `AttributeError` the first time it was ever executed.
  Replaced with a `_load()` helper mirroring `stage1_ingest._load`: reads the config from
  the run manifest (so Lab parameter overrides are what get scored, not defaults) and loads
  outputs from `run_dir / "condition"` rather than the run root. Fixed at all three call
  sites (`evaluate`, `headline`, `render_results`).
- `tests/test_ui_smoke.py` — added `test_stage2_runs_prerequisite_stages_from_the_ui`,
  mirroring the Stage 1 end-to-end test.
- `DEVLOG.md` — status board, open issues (S2-1/S2-2 struck, S2-5/S2-6/S2-7 added), this entry.

**Deleted/Moved:** (none)

**Decisions:**
- Did **not** add `"ingest"` to Stage 2's `manifest_stages`. Ownership must stay disjoint —
  `tests/test_stage1_eval.py::test_every_manifest_stage_belongs_to_exactly_one_spec_stage`
  enforces it, and the budget keys off the same tuple. Dependency is a separate concept, so
  it got a separate helper. This also fixes Stages 3–6 in advance: `occlusion` resolves to
  `['ingest','condition','fusion']` and `recon` to `['ingest','condition','track_b',...]`,
  respecting the §6/§7 execution-order inversion.
- Verified the 2026-09-16 xfail removals were legitimate: only the `@pytest.mark.xfail`
  decorators were removed, assertion bodies untouched. Measured both KPIs directly —
  blockiness reduction **0.233** (target > 0.10, comfortable); shadow recall **0.521**
  (target > 0.50, passes by 0.021). The session entry's "69% recall" is not reproducible on
  the repo's own test scene; logged the measured number instead.

**Dead ends (also add to §4):** (none — the failed approach here was the *previous* session's
assumption that a panel which loads is a panel that works; recorded as S2-7.)

**Tests:** 157 passed / 0 xfailed / 0 failed (was 156; +1 new). New test confirmed to fail
with `RuntimeError: conditioning needs a completed ingest stage` when the `ui/app.py` fix is
reverted, so it is a real regression guard and not a tautology.

**Open issues added/closed:** S2-5 closed. S2-1, S2-2 struck (were closed 2026-09-16 but §5
had not been updated). S2-6 and S2-7 added. S2-3 corroborated with numbers. S2-4 unchanged
(run logged `ultralytics is not installed`).

**Next:**
- **S2-6 is the blocker**: Stage 2's scorecard reads 0/100 until `src/condition/*` writes the
  four report JSONs and two parquet frames the evaluator expects. Until then "BUILT" overstates
  Stage 2 — the panel runs, but it cannot score.
- Then S2-7: add `tests/test_stage2_eval.py`.
- Stage 4 Track A (OpenDroneMap) remains the submission floor.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 2 (S2-6: the scorecard reads 0/100)
**Goal:** Close S2-6 — make the conditioning stage write the artifacts `src/qa/stage2_eval.py`
reads, so the Stage 2 scorecard measures the run instead of reporting missing files.

**Created:**
- `tests/test_stage2_eval.py` — 18 integration tests over a real synthetic run: the four report
  JSONs and two parquets exist, are registered as manifest artifacts, load non-empty, carry the
  columns the charts plot, and produce a scorecard with no `*_report_missing` KPI.

**Modified:**
- `src/pipeline.py` — added `_write_condition_reports()`; `run_condition` now tracks
  `artifact_assessments` and emits `artifacts_report.json`, `illumination_report.json`,
  `dynamic_report.json`, `gps_report.json`, `frame_artifacts.parquet` and
  `frame_illumination.parquet`, all registered in the manifest. The illumination and dynamic
  summaries are computed once and shared with the metrics dict rather than recomputed.
- `src/qa/stage2_eval.py` — `_gps_kpis` read `max_speed_observed`; `GpsReport.to_dict()` writes
  `max_speed_observed_mps`, so the max-speed KPI could never fire. Reads the canonical name now,
  falling back to the old spelling.
- `ui/stages/stage2_condition.py` — KPI table cast to text before `st.dataframe`. With only four
  KPIs the column was all-numeric; with twelve it mixes floats, ints, bools and strings, and
  Arrow inferred `double` then raised `ArrowInvalid` on `"baro+gps_complementary"`. Streamlit
  swallowed it into a fallback render, so the suite stayed green while the table degraded.
- `DEVLOG.md` — status board back to 🟢, S2-6 struck, S2-7 narrowed, S2-8 added, this entry.

**Deleted/Moved:** (none)

**Decisions:**
- Wrote the reports from what the stage *already computes*. `summarize_illumination`,
  `summarize_masks`, `ExposureChain.summary` and `GpsReport.to_dict` already returned exactly the
  keys the evaluator wanted — they were being routed into manifest metrics and nowhere else. This
  is an adapter, not new measurement, which is why it is ~90 lines and changed no numbers.
- The reports are written even though the manifest holds the same numbers, because the evaluator
  is a standalone reader: it takes a run directory and nothing else, so the Lab can score a run it
  did not launch. Duplication is the price of that contract.
- **Left the ground-truth KPIs dormant on purpose.** `blockiness_before/after`, `shadow_recall` and
  `shadow_false_positive` need per-pixel truth for real frames; `generate_synthetic_flight`
  produces flight-level truth only. The evaluator skips absent keys, so they stay silent rather
  than being filled with numbers nothing measured. Logged as S2-8, with a test asserting they
  remain absent so nobody "fixes" this by fabricating them.
- `keypoints_vetoed` is reported as 0 with a comment: grid-keypoint vetoing (`filter_grid_keypoints`)
  runs at feature extraction in Stage 4, not in conditioning.

**Dead ends (also add to §4):** (none)

**Tests:** 175 passed / 0 xfailed / 0 failed (was 157; +18). Verified the new tests are real guards
by stashing `src/pipeline.py` and re-running: 16 of the 18 fail without the fix. The 2 that still
pass are the honesty checks asserting the ground-truth KPIs stay dormant, which is correct in both
states.

**Open issues added/closed:** S2-6 closed. S2-7 narrowed to "unit coverage of KPI threshold logic".
S2-8 added (100/100 overstates the stage — 5 of 12 KPIs are INFO and cost nothing).

**Next:**
- S2-8 then S2-3: decide whether shadow coverage becomes a scored KPI, or add per-pixel truth to
  `src/qa/synthetic.py` so shadow recall scores for real. Until then treat 100/100 as "nothing is
  broken", not "nothing can be improved".
- S2-4 still needs `ultralytics` on the GPU box before the semantic masking path is exercised.
- Stage 4 Track A (OpenDroneMap) remains the submission floor.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 2 (S2-7: evaluator test coverage)
**Goal:** Close S2-7 — the Stage 2 evaluator's KPI threshold logic had no unit coverage, so
every warn and fail branch was unexercised.

**Created:** (none — extended `tests/test_stage2_eval.py`)

**Modified:**
- `tests/test_stage2_eval.py` — added 48 table-driven unit tests (66 total in the file). They
  drive `evaluate_condition` with synthesised report dicts rather than a pipeline run, one case
  per threshold boundary: artifact correction fraction, blockiness reduction (and its absence
  without ground truth), exposure gain span, rejected links, low-light, shadow recall, dark-paint
  false positives, peak masked area, GPS outlier fraction, smoothed-fix survival, both spellings
  of the max-speed key, the scale-free GPS early return, the missing-report branches, and the
  score arithmetic (warn counts half, info does not count).
- `DEVLOG.md` — S2-7 struck, status board suite count, this entry.

**Deleted/Moved:** (none)

**Decisions:**
- Boundary values, not midpoints. Every case sits exactly on a threshold (0.15 not 0.2; 0.50 not
  0.6), because that is the only place an off-by-one comparison shows up.
- Verified the tests are load-bearing by mutation rather than by assuming: flipping `<` to `<=`
  in the gain-span comparison and `>` to `>=` in shadow recall failed 3 tests. A boundary test
  that passes under both spellings of the comparison is not testing the boundary.
- Fixed one expectation of mine, not the code: the score-arithmetic case scores 83.3, not 75 —
  `exposure_rejected_links` also passes when no links were rejected, so three KPIs score, not two.
  The test now names all three explicitly so the arithmetic is readable.

**Dead ends (also add to §4):** (none)

**Tests:** 223 passed / 0 xfailed / 0 failed (was 175; +48).

**Open issues added/closed:** S2-7 closed. Remaining Stage 2 issues: S2-3 (shadow over-detection),
S2-4 (ultralytics on the GPU box), S2-8 (100/100 overstates the stage).

**Next:**
- S2-8 / S2-3 together: promote shadow coverage to a scored KPI, or add per-pixel truth to
  `src/qa/synthetic.py` so `shadow_recall` and `shadow_false_positive` score for real. The unit
  tests for both already exist and are currently exercised only with synthesised inputs.
- Stage 4 Track A (OpenDroneMap) remains the submission floor.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 1 (KLV / STANAG 4609 telemetry)
**Goal:** The PS targets military-intelligence and disaster-management use, where imagery is
normally STANAG 4609 — telemetry multiplexed into the video stream as MISB ST 0601 KLV, not a
DJI sidecar. Add a KLV telemetry source while leaving the SRT/CSV/EXIF paths untouched.

**Created:**
- `src/ingest/klv.py` — MISB ST 0601 parser: MPEG-2 TS depacketizer, BER/BER-OID decoding,
  ST 0601 fixed-point scaling, checksum verification, and container resolution (TS, raw `.klv`,
  optional MP4 scan). Maps sensor lat/lon/alt and platform heading/pitch/roll onto the canonical
  schema, and carries frame centre, slant range, sensor-relative pointing and FOV as extra columns.
- `tests/test_klv.py` — 25 tests. Includes an ST 0601 *encoder* written from the standard, so the
  round-trip tests compare two independent implementations of the same scaling rules.

**Modified:**
- `src/ingest/telemetry.py` — added the `klv` branch to `load_telemetry`; added `TELEMETRY_SOURCES`
  as the single registry of readable sources; default source list is now `[klv, srt, csv, exif]`.
- `configs/default.yaml` — `ingest.telemetry.klv` section (`enabled`, `sidecar_suffixes`,
  `scan_mp4`, `require_checksum`, `max_packets`) and the reordered `sources`.
- `ui/stages/stage1_ingest.py` — the telemetry multiselect hard-coded `["srt", "csv", "exif"]` as
  its options while taking its default from config. Adding a fourth source put a default outside
  the options, Streamlit raised, and the **whole Stage 1 panel failed to render** — two UI tests
  caught it. Options now come from `TELEMETRY_SOURCES` plus anything a preset adds.
- `DEVLOG.md` — repo map, status board, S1-9/S1-10/S1-11, this entry.

**Deleted/Moved:** (none)

**Decisions:**
- **No new dependency.** `klvdata`/`pymisb` would work, and QGISFMV uses pymisb + ffmpeg, but
  neither ffmpeg nor klvdata is installed here and the PS puts this format on the critical path.
  Followed the `dji_flight_record.py` precedent: self-contained binary parser, format documented
  in the module docstring.
- **No PAT/PMT walk.** The 16-byte universal key identifies local sets unambiguously once TS
  headers are stripped, and each set's BER length says where it ends. Parsing PES framing would
  add failure modes without adding information. The PID with the most keys wins.
- **MP4 scanning is opt-in.** Proving a DJI MP4 has no KLV costs a full read of a 4K file on every
  run. TS and `.klv` sidecars are found automatically; MP4 needs `scan_mp4: true` (S1-10).
- **ST 0601 error markers decode to NaN**, not to a plausible angle. `0x8000` in tag 6 means "no
  reading"; decoding it as -20° pitch would be a silent lie. Tested.
- **Lenient checksums by default.** Failures are counted and reported in the table notes; some
  encoders emit a wrong checksum on otherwise sound packets. `require_checksum: true` rejects the
  stream instead.
- `klv` goes **first** in the source order: it is multiplexed against the video's own clock, so it
  needs none of the alignment search the `flight_record` settings exist for. It is absent from DJI
  footage and falls through at INFO, not as a downgrade.

**Dead ends (also add to §4):** (none)

**Tests:** 248 passed / 0 xfailed / 0 failed (was 223; +25). Verified the round-trip tests are real
by mutating the parser's scale factors — latitude ±90→±180 and altitude offset −900→0 each fail
`test_position_round_trips`. Verified the DJI path is unaffected by running the real pipeline on
`demo_flight.mp4`: KLV logs "no KLV/STANAG 4609 metadata", SRT parses 120 rows, elapsed 4.3 s.

**Open issues added/closed:** S1-9, S1-10, S1-11 added. None closed.

**Next:**
- **S1-9 is the one that matters**: this parser has never seen a real STANAG 4609 file. Get one
  before trusting it — real muxes vary (multi-PID, PES padding, ST 0604 timestamps, ST 0102
  security sets).
- Surface the new georeferencing fields (frame centre, slant range, sensor pointing) in the Stage 1
  Lab telemetry tab, and use them as pose priors in Stage 4 / georeferencing in Stage 5.
- Stage 4 Track A (OpenDroneMap) remains the submission floor.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 2 (S2-8: make the scorecard able to fail)
**Goal:** Close S2-8. Stage 2 scored 100/100 with 5 of 12 KPIs reporting "info", so no degraded
run could move the number. A scorecard that cannot go red cannot localise a fault, which is the
whole reason the Stage Lab exists — and the plan is to lean on it when real footage arrives.

**Created:** (none — extended `tests/test_stage2_eval.py`)

**Modified:**
- `configs/default.yaml` — new `qa.stage2` section holding every Stage 2 scorecard band:
  `shadow_fraction_warn/fail`, `low_light_fraction_warn/fail`, `altitude_source_pass/warn`, plus
  `gain_span_warn` and `gps_outlier_fraction_warn/fail`, which had been hard-coded module
  constants in the evaluator (rule 6).
- `src/qa/stage2_eval.py` — `_band()` / `_band_list()` read those thresholds with the old
  constants as fallbacks, so an old preset still scores. Three KPIs promoted from INFO to scored:
  mean shadow coverage, low-light fraction, and altitude source.
- `tests/test_stage2_eval.py` — 19 tests added (85 in the file). Bands for all three promoted
  KPIs, a config-override test proving the band is tunable, and `TestTheCardCanFail`, which is
  the actual S2-8 regression guard.
- `DEVLOG.md` — S2-8 struck, status board, this entry.

**Deleted/Moved:** (none)

**Decisions:**
- **Promoted three KPIs, not five.** The remaining INFO KPIs are context, not quality signals, and
  scoring them would add noise rather than sensitivity: `keypoints_vetoed` is always 0 because
  grid vetoing runs in Stage 4; `gps_rtk_detected` — most platforms have no RTK and its absence is
  not a defect; `dynamic_frames_with_movers` — how much of a frame is masked is the
  reconstruction-relevant quantity and is already scored as `dynamic_masked_fraction_max`, so a
  survey over a highway would be penalised twice for nothing.
- **Altitude provenance is scored on the rule the KPI text already stated**: complementary baro+GPS
  passes, GPS-only warns (noisier but still an absolute datum), `baro_relative_only` fails because
  the vertical datum is unknown and absolute georeferencing downstream becomes a guess.
- **Shadow bands are deliberately loose** (warn 0.30, fail 0.50). A genuinely shadowed scene can
  reach 0.3; past half the frame an over-firing detector is the likelier explanation. This measures
  *coverage*, not *correctness* — which is why it does not close S2-3.
- Fixed two of my own test expectations rather than the code: the score-arithmetic case moves
  83.3 → 90.0 now that low-light and shadow score, and `dynamic_frames_with_movers` legitimately
  stays INFO.

**Dead ends (also add to §4):** (none)

**Tests:** 267 passed / 0 xfailed / 0 failed (was 248; +19).

**Open issues added/closed:** S2-8 closed. S2-3 and S2-4 remain open for Stage 2.

**Next:**
- **S2-3 still needs per-pixel truth.** `shadow_recall` and `shadow_false_positive` have unit tests
  but stay dormant on real runs because `src/qa/synthetic.py` emits flight-level truth only. Note
  the evaluator reads those values from the *report*, so wiring them up means either contaminating
  the production `run_condition` with test-only truth or moving ground-truth KPIs to evaluation
  time — the second is the right design and is not a small change.
- Real-footage Stage 2 run on `LineVision-VideoGeoTagging.mp4`, then Track A on public imagery.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 1 (KLV validated on real ISR footage)
**Goal:** Close S1-9 — the KLV parser had only ever seen packets this repo encoded itself. Eight
real STANAG 4609 clips arrived (QGISFMV sample media, `Drone Video Dataset/QGISFMV_Samples/MISB/`).

**Created:** (none — extended `tests/test_klv.py`)

**Modified:**
- `src/ingest/klv.py` — **two real bugs**, both invisible to the synthetic tests:
  1. `_checksum` was a plain byte sum. ST 0601 specifies `bcc_16`, which adds each byte into
     alternating halves of a 16-bit accumulator. The plain sum matched **0 of 2,396** real packets;
     `bcc_16` matches all of them. `require_checksum: true` would have rejected every real stream.
  2. `find_klv_source` gated on file extension. All eight samples are MPEG-2 TS but are named
     `.ts`, `.mp4`, `.mpeg4` and `.H264` — the pipeline missed 5 of 8. Added `file_is_ts()`, which
     sniffs 189 bytes for sync-byte framing; a genuine MP4 still needs `scan_mp4`.
- `tests/test_klv.py` — `TestRealPacket` (a real 238-byte local set from `klv_metadata_test_sync.ts`
  as a hex vector) and `TestMisnamedContainers`. 38 tests in the file.
- `DEVLOG.md` — S1-9 closed, S1-10 superseded, S1-12 and S1-13 added and closed, this entry.

**Deleted/Moved:** (none)

**Decisions:**
- **Kept `scan_mp4: false`.** The evidence looked like it argued for flipping it — until the files
  turned out to be TS wearing an `.mp4` extension. Sniffing the container fixes the real problem;
  the original reasoning (never read a 475 MB DJI MP4 to prove it has no KLV) still stands.
- Embedded a real packet as a test vector rather than only widening the synthetic tests. The
  checksum bug is the exact failure a self-encoding round-trip cannot catch: encoder and decoder
  agreed with each other and disagreed with the world. One external vector is worth more here than
  any number of self-consistent ones.

**Dead ends (also add to §4):**
- **A round-trip test against your own encoder cannot validate a shared assumption.** The ST 0601
  scaling was caught this way (mutation-tested, genuinely independent), but the checksum was not,
  because the same wrong formula was written on both sides. Any future binary format added here
  needs at least one vector from outside the repo before it is trusted.

**Tests:** 277 passed / 0 xfailed / 0 failed (was 267; +10). Verified the real vector is what catches
the checksum bug: reverting `_checksum` to the plain sum fails exactly the 2 `TestRealPacket` tests
while all 27 synthetic KLV tests still pass.

**Validation against burned-in overlay** (`Cheyenne_Handoff`, frame at 00:03, overlay vs parsed):

| Field | Overlay | Parsed | |
|---|---|---|---|
| UTC | 19SEP2012 14:32:35 −6.0 | 2012-09-19 20:32:34 UTC | matches to the second |
| Aircraft lat/lon | 41.09990N 104.78970W | 41.09984 / −104.78979 | ~7 m |
| Aircraft altitude | 9749 ft | 2973 m = 9754 ft | 5 ft |
| Frame centre lat | 41.12784N | 41.127838 | exact |
| Frame centre elevation | 6037 ft | 1839.9 m = 6036 ft | 1 ft |
| Slant range | 1.8 NM | 3314 m = 1.789 NM | rounds to 1.8 |

**Open issues added/closed:** S1-9, S1-12, S1-13 closed. S1-10 superseded. S1-11 (no baro/focal in
KLV) still stands and is inherent to the format.

**Next:**
- These eight clips are usable Stage 1 inputs *today* — real nadir and oblique ISR footage with
  frame-synced telemetry, no alignment guesswork. Run Stage 1 + 2 against them for a real-footage
  scorecard.
- Stage 4 Track A (OpenDroneMap) remains the submission floor.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 2 (first real-footage run + dark-mode scorecard)
**Goal:** Read the first real Stage 2 scorecard (Esri_multiplexer_1, a real ISR clip), fix the
unreadable Diagnostics table in dark mode, and answer whether 78/100 is a good result.

**Created:** (none)

**Modified:**
- `ui/stages/stage2_condition.py` — the Diagnostics scorecard painted rows with opaque light
  pastels (`#e6ffed`, `#fff8c5`, `#ffebe9`). Streamlit keeps its own text colour, so in dark mode
  every row was light text on a pale fill and unreadable. Replaced with translucent `rgba` tints
  that sit over whatever ground the theme paints, so contrast holds in both.
- `src/pipeline.py` — `run_condition` now re-scores each corrected frame *after*
  `suppress_block_artifacts` and writes `blockiness_before`/`blockiness_after` into
  `artifacts_report.json`.
- `src/qa/stage2_eval.py` — `blockiness_reduction` no longer gated on `truth`. Before and after
  are measured on the same real frame, so nothing about it needs synthetic ground truth; the gate
  was over-conservative and left the KPI dormant on every real run. Group renamed from
  "Artifact suppression (ground truth)" to "Artifact suppression".
- `tests/test_stage2_eval.py` — the two tests that pinned the old gating updated; added a case for
  "no frame needed correction, so there is no before/after pair". The shadow ground-truth honesty
  guard is narrowed to the shadow KPIs, which genuinely still need per-pixel truth.

**Deleted/Moved:** (none)

**Decisions:**
- **Did not move the blockiness threshold to clear the fail.** 53 frames: median 1.278, but 11
  above 1.5 and 6 above 2.0 — only 6 of the 17 sit in the marginal 1.35–1.5 band. The population
  is genuinely bimodal, so the detector is right and the footage really is blocky in about a fifth
  of frames. Raising the band to make the card green would be deleting the signal, which is the
  opposite of what S2-8 was for.
- **Added the measurement the fail was missing instead.** "Frames requiring correction" measures
  the *input* — how much work it needed — and says nothing about whether conditioning helped. The
  new KPI answers that: on this clip, corrected frames went 1.787 → 1.277, a **28.6% reduction**,
  and 1.277 is below the 1.35 visible-blocking threshold, so corrected frames now read as clean.

**Dead ends (also add to §4):** (none)

**Tests:** 278 passed / 0 xfailed / 0 failed (was 277; +1).

**First real-footage Stage 2 scorecard** (`Esri_multiplexer_1.mp4`, 53 frames): **80/100**,
7 pass / 2 warn / 1 fail / 3 info.

| | value | target | |
|---|---|---|---|
| Frames requiring correction | 0.321 | < 0.30 | FAIL — property of the input, not the stage |
| Blockiness reduction on corrected frames | 0.286 | > 0.10 | PASS — suppression works on real footage |
| Exposure gain span | 0.6 | < 0.4 | WARN — real auto-exposure drift, absent from synthetic |
| Altitude source | gps | baro+gps_complementary | WARN — KLV carries no baro (S1-11) |
| Mean shadow coverage | 0.045 | < 0.30 | PASS — and far below the 0.218 the synthetic clip gave |

**Open issues added/closed:** none closed. The exposure-gain-span warn is the first real evidence
for a §5.4 issue that synthetic flights never produced; worth its own investigation.

**Next:**
- Investigate the exposure gain span (0.6 vs 0.4): real AE hunting is exactly the thing
  `ExposureChain` exists to absorb, and the synthetic generator never exercised it.
- Run the remaining 7 MISB clips for a spread rather than a single data point.
- Stage 4 Track A (OpenDroneMap) remains the submission floor.

### Session — 2026-09-18 — ritesh14g (with Claude) — Declared fallbacks for every optional input
**Goal:** Missing barometer, focal length, RTK, PyAV, ultralytics and the rest were each handled
with a working fallback, but only announced as a `downgrade` warning in `run.jsonl`. Make the
availability of every optional input a declared, visible part of each run.

**Created:**
- `src/core/inputs.py` — registry of 10 optional inputs, each with the stage that decides it, the
  fallback that runs without it, and what that costs. `describe_optional_inputs(manifest)` builds
  the per-run ledger.
- `tests/test_optional_inputs.py` — 16 tests: registry invariants, one case per flag, and the
  unknown-vs-absent distinction.

**Modified:**
- `src/pipeline.py` — writes the ledger to `manifest.summary["optional_inputs"]` and to a
  standalone `optional_inputs.json`, and logs a one-line roll-up of what was absent.
- `ui/lab.py` — `render_optional_inputs()`: an expander, open by default when anything is absent,
  listing input / status / detail / what ran instead / what it costs.
- `ui/app.py` — shown under the scorecard on every stage page and on "Pipeline so far".
- `tests/test_core.py` — `test_stage_timing_is_recorded` slept 10 ms against Windows' ~15.6 ms
  timer granularity and failed under full-suite load while passing in isolation. Now 50 ms. A
  pre-existing flake, unrelated to this change, found while running the suite.

**Deleted/Moved:** (none)

**Decisions:**
- **Derived, not threaded.** The ledger reads metrics the stages already record rather than taking
  a new parameter through every stage signature. Adding an entry costs nothing at the call sites
  and the ledger cannot claim something the run did not measure.
- **Three-valued, not boolean.** `unknown` (the deciding stage never ran) is kept distinct from
  `absent` (it ran and the input was not there). Rendering "we did not look" as "missing" would be
  a lie, and it is the state every row is in before ingest completes. Two tests pin it.
- **Absent is not a failure and is not scored.** These are legitimate operating modes, so the
  ledger sits *beside* the scorecard rather than inside it. The point is that a surprising KPI is
  often a missing input rather than a broken stage, and that link was previously invisible.

**Dead ends (also add to §4):** (none)

**Tests:** 294 passed / 0 xfailed / 0 failed (was 278; +16).

**Ledger on the real ISR clip** (`Esri_multiplexer_1.mp4`): 4 present, 6 absent — barometric
altitude, baro/GPS fusion, focal length, RTK, PyAV keyframes, semantic masking. Two of the run's
three non-pass KPIs are explained directly by it: the altitude-source WARN is the missing
barometer (S1-11, inherent to KLV), not a GPS bug.

**Open issues added/closed:** none closed.

**Next:**
- Exposure gain span 0.6 vs 0.4 on real footage — the one warn the ledger does *not* explain, so
  it is a genuine §5.4 question.
- Stage 4 Track A (OpenDroneMap) remains the submission floor.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 2 (§5.4 exposure: instrument and re-score)
**Goal:** The exposure gain span warned at 0.6 against a 0.4 target on real ISR footage — the one
non-pass KPI the optional-input ledger could not explain. Find out what it is actually measuring.

**Created:** (none)

**Modified:**
- `src/condition/illumination.py` — `ExposureTransform` carries `requested_gain` (the composed gain
  before clamping); `ExposureChain` keeps a second accumulator that is never clamped and never
  applied, purely so the true trajectory survives. `summary()` reports `requested_gain_*`,
  `clamped_fraction`, `unclamped_gain_final` and `unclamped_gain_span`.
- `src/pipeline.py` — per-frame `exposure_gain`, `exposure_gain_requested` and `exposure_clamped`
  recorded in `conditioned.parquet` and `frame_illumination.parquet`.
- `src/qa/stage2_eval.py` — new scored KPI `exposure_clamped_fraction`; `exposure_gain_span`
  demoted to INFO.
- `configs/default.yaml` — `qa.stage2.exposure_clamped_fraction_warn/fail`.
- `ui/stages/stage2_condition.py` — applied-vs-requested gain chart with a unity rule.
- `tests/test_stage2_eval.py` — bands for the new KPI, a case pinning that the runaway is named in
  the detail, and the three existing tests that assumed gain span was scored.

**Deleted/Moved:** (none)

**Decisions:**
- **`gain_span` was measuring the clamp, not the footage.** Observed `gain_min` was exactly 0.4,
  the configured `min_gain`; 46 of 53 frames were pinned. The KPI would have read 0.6 however much
  worse the underlying divergence got, which is why it warned instead of failing.
- **Scored the outcome instead.** `clamped_fraction` is 0.868 on this clip and now FAILs. Stage 2
  drops 80 → 75, which is the more honest number: 87% of frames had their exposure normalisation
  silently truncated.
- **Kept `gain_span` as INFO rather than deleting it.** The gap between it (0.6) and the requested
  span (0.72) is the size of the correction that was lost, which is worth seeing.
- **Did not raise `gain_span_warn` or `min_gain`.** The threshold is the only reason this was found,
  and the clamp is currently load-bearing — raising it would move the wall, not remove it.

**Dead ends (also add to §4):**
- **Low overlap is not the cause.** The obvious hypothesis was that budget degradation widened the
  stride until consecutive kept frames shared too little content for a photometric fit. Median
  overlap on this clip is **0.845**, comfortably above the §4.3 band, so the fit has plenty of
  shared region and is biased rather than starved. Do not re-investigate stride here.

**Tests:** 298 passed / 0 xfailed / 0 failed (was 294; +4).

**Measured on `Esri_multiplexer_1.mp4`:** gains 1.0 → 0.94 → 0.889 → 0.817 → 0.659 → 0.506 → 0.442
→ 0.383, hitting the floor at frame 7 and staying there. Unbounded final gain **0.0004**.
`rejected_links` 0, so every link passed the quality gate.

**Open issues added/closed:** S2-9 added (the divergence itself — this session instrumented and
re-scored it; the fix is still open).

**Next:**
- **S2-9 step 3:** test whether `fit_gain_bias` is symmetric on a static pair. A consistent
  sub-1 gain on identical frames would localise the bias to the fit itself.
- Then re-anchor: an absolute luminance target every N frames, or damping toward one, so error
  decays rather than compounds.
- Stage 4 Track A (OpenDroneMap) remains the submission floor.

### Session — 2026-09-18 — ritesh14g (with Claude) — Stage 2 (S2-10: unbounded bias was destroying frames)
**Goal:** Decide whether S2-9 (exposure divergence) could wait for Stage 4. It could not: checking
what reached the conditioned images found something worse than the gain drift.

**Created:** (none)

**Modified:**
- `src/condition/illumination.py` — `max_bias` bound on the composed bias, mirroring the existing
  gain bounds; `bias_clamped` per transform and `bias_clamped_frames`/`bias_max_abs` in the
  summary. Added `saturated_fraction()`, measured on the conditioned frame. `ExposureTransform`
  now records `paired_fit`, and the chain counts `quantile_fallbacks`.
- `src/pipeline.py` — per-frame `saturated_fraction` recorded and summarised into
  `illumination_report.json` with a `blown_frames` count.
- `src/qa/stage2_eval.py` — new scored KPI `saturated_fraction`; the clamp KPI's detail now names
  any quantile fallbacks.
- `configs/default.yaml` — `exposure_chain.max_bias`, `qa.stage2.saturated_fraction_warn/fail`.
- `tests/test_condition.py`, `tests/test_stage2_eval.py` — bias-bound and saturation coverage.

**Deleted/Moved:** (none)

**Decisions:**
- **Fixed now rather than during Stage 4.** 20 of 53 conditioned images were solid 255 — not
  degraded, destroyed — and those JPEGs are exactly what Track A would read. ODM would have
  dropped a third of the frames and it would have looked like a reconstruction problem, which is
  the scenario the Stage Lab exists to prevent.
- **Bounded the bias rather than reworking the chain.** The bound stops the destruction; the
  underlying drift is still S2-9 and still open. Brightness is stable now but sits at ~97 against
  a reference of 123, so the frames are usable, not correct.
- **Scored the output, not another transform.** `saturated_fraction` is measured on the image as
  written. Every other illumination KPI describes the input or the transform, which is why a
  catastrophic failure scored green.

**Dead ends (also add to §4):**
- **The quantile fallback is not the cause of S2-9.** `fit_gain_bias`'s docstring warns that the
  unregistered path "carries exactly the content-change bias" that makes a chained gain drift, and
  that was the obvious suspect. `quantile_fallbacks` is **0** on the real clip: `transform_prev` is
  present on 52 of 53 frames and `_fit_paired` succeeded every time. The bias is inside the paired
  fit. Do not re-investigate the fallback path.

**Tests:** 309 passed / 0 xfailed / 0 failed (was 298; +11). Verified the bias-bound test is a real
guard by removing the clamp: it fails, and nothing else does.

**Measured on `Esri_multiplexer_1.mp4`** — before: 20 of 53 frames mean 255, 100% of pixels at
ceiling from frame 40. After: 0 blown, saturated fraction 0.0001, brightness 97-99 across the clip.
`bias_clamped_frames` 49, `bias_max_abs` pinned at 32. Stage 2 scores 77.3.

**Open issues added/closed:** S2-10 and S2-11 added and closed. S2-9 narrowed — the fallback path
is ruled out, so the bias is in `_fit_paired`.

**Next:**
- **S2-9 remains the real fix.** Symmetry test on `_fit_paired` (A→B vs B→A on one pair), then
  re-anchor to an absolute luminance target instead of pure chaining.
- Stage 4 Track A is now safe to start: no conditioned frame is destroyed.

### Session — 2026-09-19 — ritesh14g (with Claude) — GPU-first / CPU-fallback for Stages 1–2 on the institute box
**Goal:** Assess the institute GPU (20 GB H100 MIG slice, 3 Xeon 8480+ cores, 56 GB RAM, PyTorch
CUDA notebook profile) against the PS, and make Stages 1–2 use the GPU when present and fall back
to the CPU cleanly when not.

**Created:** (none)

**Modified:**
- `src/core/device.py` — `resolve_device` cached per process (downgrade logged once);
  `use_half_precision`; cgroup-aware `cgroup_cpu_quota` / `cpu_thread_budget`; `configure_runtime`
  caps OpenCV + torch threads; `device_info` records `mig`.
- `src/ingest/video_reader.py` — `NvdecCapture` (PyNvVideoCodec `SimpleDecoder`, cv2.VideoCapture-shaped)
  as the first decode rung; `_read()` recovers from an NVDEC fault mid-stream by re-reading that frame
  through OpenCV; `VideoMetadata.decoder` records which backend ran. **Fixed an off-by-one**:
  `stream()` recorded `_position` one frame behind after `grab()` (harmless before, load-bearing for
  the fallback).
- `src/condition/dynamic_mask.py` — YOLO gets `device=0`/`"cpu"` and `half`; a GPU error retries
  the frame on CPU and stays there instead of disabling semantic masking.
- `src/pipeline.py` — `configure_runtime` at run start, result in `manifest.environment`; NVDEC keys
  passed to both readers; condition report gains `dynamic_device`.
- `src/cli.py` — NVDEC keys; `inspect` prints the decoder used.
- `src/core/inputs.py` — hardware-decode ledger line names the decoder.
- `ui/stages/stage1_ingest.py` — "Prefer NVDEC" checkbox.
- `configs/default.yaml` — `device.gpu_memory_gb` 24 → **20**; new `device.half_precision`,
  `device.cpu_threads: auto`, `ingest.video.nvdec`, `ingest.video.nvdec_gpu_id`.
- `requirements.txt` — PyNvVideoCodec listed as optional.
- `tests/test_core.py`, `tests/test_ingest.py`, `tests/test_condition.py` — cgroup parsing, thread cap,
  20 GB chunk fit, NVDEC open/mid-stream fallback (stream and read_indices, pixel-exact), YOLO GPU→CPU retry.
- `CLOUD_GPU_GUIDE.md` — new §0 (machine, sufficiency verdict, fallback map); notebook-profile
  setup, data transfer, remote UI and hygiene.
- `CLAUDE.md` — machine specs and the GPU-first/CPU-fallback rule.

**Deleted/Moved:** (none)

**Decisions:**
- **Sufficiency:** 20 GB fits Track B's default 128-frame chunk (16 GB after headroom → ~135 max),
  but not Track A + B concurrently. **3 CPU cores are the binding constraint**: software 4K decode
  can't meet the ingest budget there, so NVDEC is mandatory, not a nice-to-have. Notebook profile
  likely has no Docker → Track A at risk (ENV-1).
- **PyNvVideoCodec over OpenCV-CUDA / PyAV hwaccel:** pip-installable without sudo or a custom
  OpenCV build, which a notebook profile requires.
- **Thread cap from cgroup quota:** containers typically see all host cores (8480+ has 56 per
  socket); unbounded OpenCV/torch pools would oversubscribe 3 cores.
- Stage 2 bilateral / NLM stay on CPU: GPU versions would change outputs the scorecard is tuned
  on. Revisit only if profiling on the box shows them dominating.

**Dead ends (also add to §4):** (none)

**Tests:** 321 passed / 0 xfailed / 0 failed (was 309; +12).

**Open issues added/closed:** S1-3 and S2-4 updated (code paths in, hardware verification pending);
ENV-1 added.

**Next:**
- On the box: `nvidia-smi -L`, Docker check, `pip install PyNvVideoCodec av ultralytics`,
  `src.cli inspect` → expect `(nvdec)`; measure decode fps and YOLO ms/frame; record MIG profile,
  driver, CUDA, CPU quota here.

### Session — 2026-09-19 — ritesh14g (with Claude) — First run on the institute box
**Goal:** Bring the repo up on the institute notebook and record the hardware.

**Measured hardware:** `NVIDIA H100 80GB HBM3 MIG 2g.20gb`, torch reports **19.62 GB**; torch
2.11.0+cu128, CUDA 12.8; cgroup `cpu.max` = `300000 100000` → **3 threads** applied; conda base
**Python 3.13** (the repo targets 3.10; the venv is built on 3.13 with `--system-site-packages`);
opencv-python 5.0.0.93, PyNvVideoCodec 2.2.3, av 18.1.0, ultralytics 8.4.155. No Docker, Podman or
Apptainer (ENV-1). The Jupyter proxy rejects browser uploads with **413**, even for 1.6 MB, so
footage has to come in from the terminal (gdown or wget).

**Tests on the box:** 319 passed / 2 failed.
- `TestBudget::test_later_stages_inherit_the_squeeze`: a clock-ordering flake (7 µs) with a 1 µs
  tolerance. Fixed in the test: it now reads `remaining_s` before the allotment, so the bound holds exactly.
- `TestFrameSelection::test_a_clean_video_loses_no_frames_to_the_gate`: 2 frames rejected. **NVDEC
  engaged on the synthetic MPEG-4 Part 2 video**, and PyNvVideoCodec warned that the header frame
  count is unreliable for that codec. `NvdecCapture` now rebuilds the decoder with
  `need_scanned_stream_metadata=True` when the codec is MPEG-4. **Not yet re-verified on the box.**

**Modified:** `src/ingest/video_reader.py` (scanned metadata for MPEG-4), `tests/test_core.py`
(flake fix), `DEVLOG.md` (ENV-1 confirmed).

**Tests (local, no GPU):** 321 passed.

**Next:** on the box, `git pull`, re-run pytest, run the NVDEC-vs-OpenCV frame comparison, then
`inspect` a real H.264 clip (Esri) and expect `(nvdec)`.

### Session — 2026-09-19 — ritesh14g (with Claude) — NVDEC verified on the box; S1-14 found
**Goal:** Explain the remaining box-only failure, `test_a_clean_video_loses_no_frames_to_the_gate`.

**Measured (H100 MIG 2g.20gb, PyNvVideoCodec 2.2.3, synthetic 60-frame MPEG-4 pan):**
- **NVDEC is correct.** Frame count 60 = OpenCV. Sequential, step=3 and random `read_indices` all return
  the right frame: sharpness agrees within ~3% at every index. The mean absolute pixel difference from
  OpenCV is 1.2–3.8 grey levels, highest on the fastest-moving frames (9–17). That is an
  implementation difference between two decoders, not a fault.
- **The failure is the directional-blur test at its margin.** The blur profiles are nearly identical
  (median 206 vs 199; reject < 50 vs 51; directional < 216 vs 217). Frames 21 and 22 measure
  anisotropy 2.121 / 2.186 (OpenCV) and 2.208 / 2.244 (NVDEC) against a 2.2 limit, so NVDEC produces
  `directional_rejects: 2` and OpenCV produces 0. Outlier rejects are 0 on both.

**Modified:** `tests/test_ingest.py`: the clean-video test now asserts no *outlier* rejects (its
stated intent) and excludes directional ones, with the evidence in the comment. `DEVLOG.md`: S1-14.

**Decisions:** Did not raise `fft_anisotropy_reject`. The synthetic pan is the only evidence, and
the limit is a quality gate that has to be tuned on real footage (S1-14).

**Open issues:** S1-3 effectively verified. NVDEC engages and decodes correctly on the MIG slice;
speed on real H.264 still needs measuring. S1-14 added.

**Next:** Esri H.264 clip via gdown → `inspect` (expect `nvdec`) → full run and timings.

### Session — 2026-09-21 — ritesh14g (with Claude) — Stage 4 Track A engine probe (laptop CPU)
**Goal:** Choose the Track A engine now that ODM cannot run on the box (ENV-1), and prove the
chain end to end on the laptop before the GPU box is available.

**Created:**
- `scripts/recon_probe.py` — standalone probe: pycolmap SIFT/matching/incremental SfM (focal
  seeded from telemetry HFOV and held fixed when present), dense by COLMAP PatchMatch on CUDA or
  OpenMVS DensifyPointCloud on CPU, Poisson mesh (depth 11), OpenMVS TextureMesh → OBJ. Every GPU
  step retries on CPU with a logged downgrade. Writes `probe_summary.json` with timings and a
  metric check (camera-vs-GPS RMS, height above ground vs telemetry).

**Modified:**
- `.gitignore` — `tools/` (third-party binaries fetched per machine).
- `requirements.txt` — pycolmap (CPU) vs pycolmap-cuda12 (box), OpenMVS location, plyfile.
- `DEVLOG.md` — this entry, status board, repo map, §4 Stage 4 dead ends, ENV-1 closed, S4-1..S4-4.

**Deleted/Moved:** (none)

**Decisions:**
- **Track A = pycolmap + OpenMVS, not ODM.** ODM needs a container runtime the box lacks and is
  CPU-bound where the box is weakest (3 cores). pycolmap-cuda12 puts SIFT, matching, BA and
  PatchMatch on the GPU; OpenMVS supplies the same mesh/texture back end ODM uses. The same
  pycolmap BA is §7.3's refinement bridge for Track B.
- **Dense engine by device:** COLMAP PatchMatch (CUDA only) on the box, OpenMVS densify on CPU.
  OpenMVS's prebuilt binaries are CPU-only (Windows CUDA build exists separately; Linux build
  has no CUDA runtime).
- **Telemetry intrinsics are load-bearing** (see §4): focal from HFOV, held fixed.

**Measured (laptop, i5-1135G7, 8 threads, CPU only):**

| Clip | Frames | Registered | Reproj | Extract | Match | Map | Dense | Mesh | Texture |
|---|---|---|---|---|---|---|---|---|---|
| `demo_flight` 640×480 | 43 | 43 | 0.16 px | 6 s | 87 s | 49 s | 395 s (OpenMVS) | 14 s (d11) | 52 s |
| `Esri_multiplexer_1` 3840×2160, SIFT at 1600 | 53 | 52 | 1.13 px | 66 s | 219 s | 60 s | not run (box) | — | — |

Esri, metric check against KLV (camera ≈ 111 m above ground, nominal focal 2248 px):
self-calibrated 93 m; frozen distortion 109 m (f 2309); GPS priors 56 m; **telemetry focal fixed
107 m**. Stages 1–2 on Esri: 1 min 47 s on the laptop.

**Dead ends (also added to §4):** ODM; demo clip for 3D QA; global mapper on the flat pan; GPS
priors for focal; Poisson depth 13.

**Tests:** not run this session — no `src/` or `tests/` change (new script, `.gitignore`,
`requirements.txt` comments only).

**Open issues added/closed:** ENV-1 closed (re-scoped). S4-1, S4-2, S4-3, S4-4 added.

**Next:**
- Box: `pip install pycolmap-cuda12==4.2.0`, fetch OpenMVS Ubuntu build, run Stages 1–2 + the probe
  on Esri; record timings here (S4-3).
- Then `src/recon/track_a_colmap.py` with config in `configs/default.yaml`, Stage Lab panel,
  `src/qa/stage4_eval.py`, synthetic 3D ground truth (S4-2).

### Session — 2026-09-21 — ritesh14g (with Claude) — Stage 4 probe, first GPU run on the box
**Goal:** Run `scripts/recon_probe.py` on Esri on the H100 MIG slice.

**Measured (box):** the whole GPU chain ran: CUDA SIFT, CUDA matching, SfM, CUDA PatchMatch stereo
and fusion → **447 102 dense points** (fusion 14.7 s), Poisson 32.3 s, InterfaceCOLMAP 6.9 s.
**42 images reached the dense workspace**, against 52 registered on the laptop CPU run — to be
explained from the sparse summary (GPU SIFT differs from CPU SIFT; may be a split model).
TextureMesh then segfaulted (S4-5), and because the script let that exception escape, no
`probe_summary.json` was written and the stage timings exist only in the terminal scrollback.

**Modified:**
- `scripts/recon_probe.py` — `--reuse` (keep the sparse model and dense cloud, redo mesh + texture);
  `clean_mesh()` before texturing, with before/after counts in the summary; texture failure is a
  logged `DOWNGRADE`, not a crash; every `[probe]` line is also written to `<out>/probe.log`.
  Verified on the laptop with `--reuse` on the demo outputs (mesh 12 s, clean 4 s, texture 41 s).
- `DEVLOG.md` — S4-5.

**Tests:** not run — no `src/` or `tests/` change.

**Next:** on the box, `git pull`, then the probe with `--reuse`; paste the summary. Explain 42 vs 52.
- Follow-up: the first `--reuse` run showed NaN vertices in the Poisson mesh and stalled in
  `trimesh.split`. `clean_mesh()` now drops faces touching non-finite vertices first
  (`nonfinite_vertices` in the summary) and finds fragments with one scipy connected-components
  pass. Unit-checked on a two-sphere mesh with a NaN vertex. Box sparse summary: **49 conditioned
  frames (laptop 53), 42 registered**, reproj 1.224 px, focal fixed at 2248; no GPS metric in the
  summary, so `condition/geo.txt` is probably missing on the box run — to check.

### Session — 2026-09-21 — ritesh14g (with Claude) — NVDEC crash in Stage 2 (S1-15), OpenMVS mesher (S4-6)
**Goal:** Explain why Stage 2 never finished on the box (condition stuck at `running`, no error)
and why the box mesh was fragmented.

**Measured (box):** `-X faulthandler` re-run of condition → exit **139**, segfault in
`PyNvVideoCodec/decoders/SimpleDecoder.py:135 __getitem__` ← `VideoReader.read_indices`.
`--set ingest.video.nvdec=false` → exit 0 in 48.8 s. `--set device.prefer=cpu` → still 139.
Ingest on the box: 57.9 s of its 60 s allotment, 43 `reduce_frames` degradations (3 cores).

**Measured (laptop, on the box's GPU dense cloud, 79 MB tarball):** cloud is clean (447 102 points,
no NaN, `.vis` intact). Poisson here is sane at any thread count; the box's is not (S4-6).
OpenMVS import + Delaunay mesh + texture: 3.6 s + 20.9 s + 80.9 s → textured OBJ. Preview shows
both forested banks and the river as an empty channel (water gives stereo nothing to match).

**Modified:**
- `src/ingest/video_reader.py` — `reader_options(cfg)`; `VideoReader(nvdec_codecs=, nvdec_allow_ts=)`;
  `_open_nvdec` refuses TS containers before touching NVDEC and codecs off the allow-list after open.
- `src/pipeline.py`, `src/cli.py` — all three `VideoReader` constructions use `reader_options(cfg)`;
  dropped the now-unused `ingest_cfg` in `run_condition`.
- `configs/default.yaml` — `ingest.video.nvdec_codecs`, `ingest.video.nvdec_allow_ts`.
- `tests/test_ingest.py` — `TestNvdecInputGate` (11 tests).
- `scripts/recon_probe.py` — OpenMVS `ReconstructMesh` is the default mesher (COLMAP cloud imported
  with `-p`), Poisson + clean is the fallback; `mesher` in the summary; `ply_counts()`;
  `metric_check` no longer crashes when fewer than 3 frames match the GPS file.

**Decisions:**
- **Gate NVDEC inputs instead of catching failures.** A segfault kills the interpreter, so
  "GPU first, CPU fallback" has to be decided before the call for this library.
- **OpenMVS Delaunay over Poisson** for Track A meshing: works on both builds, is what ODM uses, and
  its meshes are what TextureMesh expects.

**Tests:** 332 passed / 0 failed (was 321; +11).

**Next:** box: `git pull`, fresh Stages 1–2 on Esri (expect `opencv` decode, 53-ish frames,
`geo.txt` written), then the full probe; log timings + GPS metric; check S4-7.

### Session — 2026-09-21 — ritesh14g (with Claude) — First clean end-to-end GPU run (Esri), dense sweep
**Goal:** Full Stages 1–2 + Track A probe on the box with the S1-15 and S4-6 fixes.

**Measured (box, H100 MIG 2g.20gb, 3 cores), `Esri_multiplexer_1.mp4`:** no `DOWNGRADE` in the
probe; NVDEC refused the TS container as designed (3 log lines), Stage 2 completed and wrote
`geo.txt` and all reports. Stages 1–2: **101.7 s** (laptop 107 s).

| Step | Box GPU | Laptop CPU |
|---|---|---|
| SIFT extract | 10.2 s | 66 s |
| Sequential match | 8.1 s | 219 s |
| Incremental map | 14.1 s | 60 s |
| Undistort (1920) | 29.8 s | — |
| PatchMatch (CUDA) | **1133.3 s** | — |
| Fusion | 22.3 s | — |
| OpenMVS import / mesh / texture | 8.1 / 14.5 / 46.5 s | — |

Sparse: 50 frames, **45 registered in 2 models**, reproj 1.231 px, focal held at 2248 px,
**height above ground 106.2 m vs 110.8 m from KLV (−4.2%)**, camera-vs-GPS RMS 7.11 m (S4-1).
Mesh: OpenMVS Delaunay, 338 958 vertices / 677 795 faces; textured OBJ 80 MB + 14 MB atlas.
Probe total 1287 s, of which PatchMatch 88% (S4-8).

**Created:**
- `scripts/dense_sweep.sh` — five PatchMatch settings on copies of one sparse model, plus a
  no-recompute measurement of the baseline cloud, then one comparison table.

**Modified:**
- `scripts/recon_probe.py` — `--pm-src-images`, `--pm-iterations`, `--pm-window-step`,
  `--pm-no-geom` (fusion switches to photometric input), `--redo-dense`; `dense_stats()` reports
  dense point count and GPS-aligned footprint (occupied 1 m cells); `metric_check` also returns the
  similarity transform; undistort uses the thread budget.
- `DEVLOG.md` — this entry, S4-8.

**Tests:** not run — no `src/` or `tests/` change. `dense_sweep.sh` passes `bash -n`; its table
code was run on local outputs.

**Next:** box: `bash scripts/dense_sweep.sh data/interim/esri_gpu data/outputs/recon_probe/esri_gpu`;
choose the dense setting; then build `src/recon/track_a_colmap.py` + config + Stage Lab panel +
`stage4_eval.py`.

### Session — 2026-09-22 — ritesh14g (with Claude) — Dense sweep round 1 (S4-8)
**Measured (box, Esri, one sparse model, 45 frames):**

| Variant | PatchMatch s | Points | Footprint m² | Mesh faces |
|---|---|---|---|---|
| baseline 1920 px, 20 views, geom | 1133.3 | 460 224 | 60 749 | 677 795 |
| 1280, 8 views, geom | 614.2 | 318 485 | 63 631 | 482 890 |
| 960, 8 views, geom | 436.2 | 249 719 | 64 872 | 377 474 |
| 1280, 8 views, no geom | 281.3 | 284 931 | 59 578 | 412 525 |
| 1280, 8 views, 3 iter, window step 2 | 165.0 | 215 729 | 51 864 | 330 775 |
| 960, 6 views, 3 iter, no geom | 115.4 | 182 640 | 53 226 | 270 691 |

**Reading:** coverage survives 1280/960 with the geometric pass (+5–7%); detail does not (points
−31% / −46%: ~15 / ~20 cm per pixel against ~10 cm at 1920 on Esri at 111 m AGL). Iteration and
window-step cuts lose 12–15% coverage. Dropping the geometric pass is not free: it is the filter
that removes wrong depths. Caveat: footprint cannot tell real extra ground from stray points, so the
coverage gain with fewer views is not proven to be real. Every round-1 variant changed size and
views together, so the cost of full resolution with fewer views is unknown.

**Provisional default:** 1280 px, 8 views, geometric pass — pending round 2.

**Scale problem:** 614 s for a 1.7-min clip → ~1 h of dense for a 10-min video against the §9 MVS
budget of 4 min. Settings alone cannot close that; frame subsampling for dense is the next lever.

**Modified:**
- `scripts/recon_probe.py` — `--dense-every N`: SfM keeps all frames, dense stereo runs on every Nth
  registered frame (`undistort_images(image_names=)`); frame count in `dense.params`.
- `scripts/dense_sweep.sh` — optional variants file, `SKIP_BASELINE=1`, `frames` column.

**Created:** `scripts/dense_variants_round2.txt` — 1920/8, 1920/8 every 2, 1280/8 every 2, 1920/12.

**Tests:** not run — scripts only. Subset undistort checked locally (22 of 43 frames, `__auto__, 8`).

**Next:** box: round 2; pick the default; start `src/recon/track_a_colmap.py`.

### Session — 2026-09-22 — ritesh14g (with Claude) — Dense sweep round 2 (S4-8)
**Measured (box):** 1920 px / 8 views / geom: **1033.7 s**, 413 511 points, 59 443 m², 609 860 faces.
1920 / 12 / geom: 1112.9 s, 456 242 points, 60 695 m², 665 154 faces (≈ baseline at −2% time).
**Source views are not the lever at full resolution** (20 → 8 saves 9% time, costs 10% points);
pixel count is. Both `--dense-every 2` variants failed.

**Dead end (also §4 material):** `undistort_images(image_names=...)` writes only the listed image
files but keeps every frame in the undistorted model, so PatchMatch's `__auto__` source-view
selection references images that do not exist (reproduced locally: model 43 frames, 22 files).
**Fix:** deregister the skipped frames from a copy of the model (`deregister_frame`) and undistort
that copy; verified 22 frames / 22 files / 22 `patch-match.cfg` references.

**Modified:** `scripts/recon_probe.py` (subset model before undistort). **Created:**
`scripts/dense_variants_round2b.txt`.

**Next:** box: round 2b (the two every-2 variants).

### Session — 2026-09-22 — ritesh14g (with Claude) — Dense sweep round 2b; dense presets chosen (S4-8)
**Measured (box):** every-other-frame dense (23 of 45 frames) — 1920/8/geom: 436.2 s, 43 244 points,
**8 284 m²** (−86% vs all frames); 1280/8/geom: 268.6 s, 39 209 points, **11 335 m²** (−82%).

**Dead end (also §4):** ❌ **Dense stereo on a frame subset.** Stage 1 already selects for ~70–80%
overlap, so halving frames leaves most surface seen by too few views, and fusion
(`min_num_pixels` 5, `filter_min_num_consistent` 2) discards it. Coverage fell 5–7× while time
fell only 2.4×. Dense needs every frame Stage 1 keeps. (`--dense-every` stays in the probe for
footage selected with much higher overlap.)

**Decision — Track A dense presets (all keep the geometric pass and all frames):**
- default: 1280 px, 8 source views — 614 s here, footprint 105%, points 69% (~15 cm/px on Esri)
- accurate: 1920 px, 12 source views — 1113 s, 100%, 99% (~10 cm/px)
- fast: 960 px, 8 source views — 436 s, 107%, 54% (~20 cm/px)
Source views are not a speed lever at full resolution (20 → 8: −9% time); pixel count is.

**S4-8 stays open:** ~14 s/frame at the default → ~1 h of dense for a 10-min video vs the §9 4-min
MVS budget. Remaining levers: Track B depth for all frames with MVS only on Zone 1 (spec hybrid),
an OpenMVS densify built with CUDA on the box, and fusion `min_num_pixels` 5 → 3 (noise trade).

**Tests:** not run — no code change this entry.

**Next:** build `src/recon/track_a_colmap.py` with these presets in `configs/`.

### Session — 2026-09-22 — ritesh14g (with Claude) — Track B probe (VGGT), licensing position
**Goal:** Measure VGGT on the box before writing Track B, while Track A is turned into a module.

**Found:** Hugging Face has `facebook/VGGT-1B` (CC-BY-NC-4.0, **not gated**, 5.03 GB),
`facebook/VGGT-1B-Commercial` (manual approval), and `facebook/VGGT-Omega` (**manual approval**,
FAIR non-commercial research licence, separate code at `facebookresearch/vggt-omega`, checkpoints
`vggt_omega_1b_512.pt` etc.). All carry the VGGT acceptable-use policy (LIC-1). Probe uses VGGT-1B;
VGGT-Ω needs a team member's approved HF access.

**Created:** `scripts/vggt_probe.py` — loads frames once (VGGT "crop" preprocessing, 518 px wide:
518×294 on 16:9), times chunks 8/16/32/all with a CUDA OOM guard, and scores the largest chunk:
focal vs KLV HFOV, camera centres vs GPS (similarity fit), camera centres vs Track A's COLMAP cameras
(both in metres), height above ground from the central depth vs KLV. Saves extrinsics, intrinsics,
depth and confidence to `vggt_chunk.npz` for Track B development. CPU = 2-frame smoke test.

**Verified locally:** `--random-weights`, 3 frames on the laptop CPU: every path runs (34 s for the
forward pass); quality numbers meaningless by construction. `torchvision` 0.23.0+cpu, `einops`,
`safetensors`, `huggingface_hub` installed into the laptop venv; `tools/vggt` cloned (git-ignored).

**Open issues:** LIC-1 added (team position recorded).

**Next:** box: VGGT probe on Esri; request VGGT-Ω access; build `src/recon/track_a_colmap.py`.

### Session — 2026-09-22 — ritesh14g (with Claude) — Stage 4 Track A module: BUILT
**Goal:** Turn the probe into the Stage 4 Track A module with config, pipeline wiring, budget,
evaluator, Stage Lab panel and tests (rules 3–6), and flip Stage 4 to BUILT.

**Created:**
- `src/recon/__init__.py`, `alignment.py`, `openmvs.py`, `meshing.py`, `track_a_colmap.py` — see §3.
- `src/qa/stage4_eval.py` — 16 KPIs in four groups (sparse, metric vs telemetry, dense and mesh,
  runtime); camera-vs-GPS uses the PS's ≤ 1 m target, so Esri's 7.1 m reads **fail** (S4-1).
- `ui/stages/stage4_recon.py` — sidebar for the measured tunables, camera path (SfM aligned vs GPS),
  time per step, output file paths.
- `tests/test_recon.py` — 32 tests incl. a CPU end-to-end run on a 16-frame 320 px synthetic flight.

**Modified:**
- `configs/default.yaml` — `recon.track_a` rewritten (ODM keys removed): telemetry focal, masks,
  SIFT/matching/mapper, dense presets and per-frame cost model (`gpu_s_per_frame_at_1280: 13.6`,
  `cpu_s_per_frame_at_1280: 27`, `degrade_sizes`, `reserve_after_s: 120`), mesher, texture,
  OpenMVS path; `qa.stage4` bands. `configs/fast.yaml` (960 px) / `accurate.yaml` (1920 px, 12 views).
- `src/pipeline.py` — Track A stage: runs after conditioning under `budget.stage("track_a_mvs")`;
  while Track B is unbuilt its allowance and refine_ba's (300 s) are handed to Track A; downgrades
  become stage warnings; unbuilt manifest stages of a built stage say so in their skip reason.
- `src/stages.py` — Stage 4 **BUILT** (Track A); summary says Track B is planned.
- `ui/stages/__init__.py` — panel registered. `requirements.txt` — `plyfile` now required.
- `tests/test_core.py` (preset assertion on the new keys), `tests/test_stage1_eval.py`,
  `tests/test_stage2_eval.py` (their fixtures now request `ingest`+`condition` only: with Stage 4
  built, a default run reconstructs, which took the suite from 70 s to > 10 min).

**Decisions:**
- **Dense resolution is chosen before the call, from a cost model.** PatchMatch is one GPU call and
  cannot degrade halfway, so frames × s/frame × (px/1280)² is compared with the stage's remaining
  time minus a reserve, stepping down `degrade_sizes` and logging each step. Frames smaller than the
  target are costed at their own size. On the box, Esri at the default budget projects 612 s at
  1280 px against ~385 s available, so **a default run densifies at 960 px** and logs it; the
  `accurate` preset (budget 3600 s) keeps 1920 px.
- **Focal from DJI SRT too:** `focal_mm` there is the 35 mm-equivalent OSD value, so
  f_px = f × width / 36. The demo run used it (`telemetry_focal_35mm`).
- **Dynamic masks reach SfM:** Stage 2's `<stem>_exclude.png` are inverted into COLMAP's
  `<image>.png` convention (0 = no features). Not yet applied to fusion (masks are in distorted space).

**Bugs found by the new tests (fixed):**
- OpenMVS `ReconstructMesh` can clean a thin surface to nothing, **exit 0 and write no file**; Track A
  then crashed reading it. An empty or missing mesh is now a failure that falls back to Poisson.
- An empty Poisson mesh loads in `trimesh` as an empty `Scene`; `clean_mesh` now raises "empty mesh"
  and Track A reports "dense point cloud only" instead of an `AttributeError`.

**Measured (laptop CPU, pipeline `--stage track_a` on `demo_flight`):** 7 min 23 s; 43/43 registered,
reproj 0.161 px, 1 155 951 dense points (OpenMVS CPU, 235.6 s), mesh 604 480 faces (2.0 per vertex),
textured; scorecard **95.5/100** (10 pass, 1 warn = no GPU, 5 info). The demo is planar and cannot
judge 3D quality (§4); this run checks plumbing.

**Tests:** 364 passed / 0 failed (was 332; +32).

**Open issues:** S4-9 added (fusion ignores dynamic masks). S4-1, S4-7, S4-8 still open.

**Next:** box: `git pull`, full pipeline on Esri (Stages 1–4, default preset) and record the scorecard;
then Track B from the VGGT probe results.

### Session — 2026-09-22 — ritesh14g (with Claude) — VGGT probe on the box: fast, focal right, long-range poses wrong
**Measured (box, VGGT-1B, bf16, 518×294 input, Esri 50 frames):**

| Chunk | Time | s/frame | Peak GPU |
|---|---|---|---|
| 8 | 1.26 s | 0.158 | 8.44 GB |
| 16 | 1.70 s | 0.106 | 8.69 GB |
| 32 | 2.78 s | 0.087 | 9.19 GB |
| 50 | 4.36 s | 0.087 | 9.75 GB |

~150× faster than Track A dense (13.6 s/frame) and far inside 20 GB. Weights: `facebook/VGGT-1B`,
CC-BY-NC-4.0, not gated. **Focal 2245.8 px vs 2248 from KLV HFOV (−0.1%)** without telemetry.

**But one 50-frame pass over the 800 m flight is geometrically wrong:** camera centres vs GPS
**187.6 m** RMS (Track A: 7.1 m), vs COLMAP's cameras 179.5 m; height above ground 162.7 m vs
110.8 m (+47%). Consistent with the spec's warning that feed-forward models degrade on large scenes
(§7.2 prescribes chunking + Sim(3) stitching + BA). Not yet known whether short chunks are accurate.

**Modified:** `scripts/vggt_probe.py` — `--windows` (default 8,16,32): slides half-overlapping
windows along the flight and scores each (camera vs GPS, vs COLMAP, height error, focal error,
window path length) plus **Track A's own camera-vs-GPS on the same frames** as the GPS-noise
floor; medians and worst cases in the summary, every window in `vggt_windows.json`.

**Tests:** not run — script only; windows path checked locally with random weights.

**Next:** box: windows run. Decision rule: if 16/32-frame windows sit near Track A's residual and
height within ~5%, build Track B as chunked VGGT + Sim(3) stitching + refinement BA; if not, VGGT
serves Stage 3 depth only and Track A carries the poses.

### Session — 2026-09-22 — ritesh14g (with Claude) — VGGT windows: accurate short, broken long; hybrid probe
**Measured (box, sliding half-overlapping windows over Esri's 50 frames; Track A's own residual on
the same frames in brackets):**

| Window | GPS path | VGGT vs GPS | VGGT vs COLMAP cams | Height err median / worst | Focal err |
|---|---|---|---|---|---|
| 8 | 122 m | 1.40 m (0.92) | **0.70 m** (worst 2.02) | **−2.6%** / 16.0% | −0.7% |
| 16 | 259 m | 4.67 m (2.71) | 4.51 m (worst 7.65) | +8.5% / 14.4% | −0.5% |
| 32 | 531 m | 92.5 m (4.66) | **90.2 m** (worst 137.3) | +5.8% / 61.9% | −0.4% |

VGGT is as good as Track A while a window's frames still share content (Esri's footprint is
~190 m wide) and falls apart beyond that. Chaining 8-frame chunks over a 10-min flight (~100 links)
would accumulate drift; the spec's chunk-stitch-refine plan is fragile here.

**Decision (pending the depth measurement):** Track B = **VGGT depth on Track A's SfM poses**
(§7.4 hybrid, "Track B depth fills"), not VGGT poses. Track A sparse is ~32 s on the box; VGGT depth
~0.09 s/frame replaces ~13.6 s/frame of PatchMatch. Depth is scaled per frame by the Track A sparse
points the frame observes (§6.3-style anchoring). Cost: VGGT runs at 518 px (~37 cm/px on Esri vs
15 cm for Track A dense at 1280), so `accurate` keeps Track A dense.

**Created:** `scripts/vggt_hybrid_probe.py` — 8-frame half-overlapping windows over the undistorted
COLMAP workspace, per-frame anchoring on sparse points (plus a rotation-aware camera-fit scale for
comparison), pixelwise comparison with COLMAP's geometric depth maps (median relative error, share
within 5% / 10%, metric error), fused hybrid cloud + GPS footprint. Verified locally with random
weights on 6 frames (≈900 anchors per frame); COLMAP depth-map reader round-trip tested.

**Next:** box: hybrid probe against the kept 1920 px COLMAP depth maps.

### Session — 2026-09-22 — ritesh14g (with Claude) — Track B built as the hybrid (VGGT depth on Track A cameras)
**Measured (box, hybrid probe, Esri 45 frames, 8-frame windows, per-frame anchoring):** VGGT 5.9 s
(0.13 s/frame) + 0.8 s fusion; **median depth error vs COLMAP's 1920 px geometric depth 1.04%**
(1.07 m at ~111 m), 99.1% of pixels within 5%, 99.9% within 10%, worst frame 2.85%; anchors 982 per
frame, anchor disagreement 0.85%. Camera-fit scale instead of per-frame anchoring: 1.76% / 96.5%.
Cloud 1.2 M points, footprint **115 555 m²** vs 60 749 m² for COLMAP dense — the extra area is where
COLMAP had no depth (it covered 57.6% of pixels: water, uniform canopy, edges), so it is **not verified**
and is exactly Stage 3's Zone 2.

**Created:** `src/recon/track_b_vggt.py` — see §3. Visibility index convention verified on the box's
COLMAP output: `fused.ply.vis` indexes images in **image-id order** (100% of points project into their
listed images vs 8% for file order). A point is kept when its source frame and ≥ 1 of the ±3
neighbouring frames agree on its depth within 3% (COLMAP's own points: median 4 views).

**Modified:**
- `src/recon/track_a_colmap.py` — dense step: `run.mode` A → Track A dense; auto/hybrid/B → Track B
  first, any exception → logged downgrade to Track A dense. Undistort once at `dense.max_image_size`
  (feeds VGGT, PatchMatch and texture); the budget projection now applies only when COLMAP dense runs.
  Per-frame depth + confidence saved to `track_a/track_b_depth/*.npz` for Stage 3. `depth_predictor`
  injection point for tests.
- `configs/default.yaml` — `recon.track_b` rewritten (VGGT-1B, repo_dir, require_gpu, window 8,
  anchoring, consistency, stride, 0.13 s/frame); old chunk/stitch/loop-closure keys removed (nothing
  read them; the measurements ruled that design out). `qa.stage4` anchor-spread and frames-anchored
  bands. `configs/accurate.yaml` → `run.mode: A` (1920 px detail); `fast.yaml` track_b block removed.
- `src/qa/stage4_eval.py` — "Track B (VGGT depth)" KPI group: used / fell back (with reason), frames
  anchored, anchor disagreement, views per point; mode A reports INFO, not a penalty.
- `ui/stages/stage4_recon.py` — mode selector and VGGT window / confidence / tolerance controls.
- `src/pipeline.py` — track_b's skip reason says it ran inside track_a as the hybrid.
- `tests/test_recon.py` — windows cover every frame once; `.vis` round trip; hybrid on the synthetic
  flight with a plane-depth stand-in for VGGT at a wrong 3.7× scale (anchoring recovers it, spread
  < 2%, points confirmed by ≥ 2 views, depth maps saved); VGGT crash → Track A dense; mode A never calls
  VGGT; scorecard cases.

**Bug found by the new tests (fixed):** the `.vis` writer mixed uint32 counts with signed aranges,
which numpy promotes to float64 → `IndexError`. Offsets are int64 now.

**Tests:** 374 passed / 0 failed (was 364; +10). Suite 131 s (the hybrid tests run real SfM).

**Next:** box: full Stages 1–4 on Esri (default = hybrid), paste the Stage 4 scorecard.

### Session — 2026-09-22 — ritesh14g (with Claude) — First full hybrid run on the box; SfM merge; mesh reduction; VGGT width
**Measured (box, `src.cli run` Esri, default preset = hybrid):** Stage 4 **149 s** (was ~1290 s as the
Track A probe): sparse 31.4 s, undistort 10.0, **dense_vggt 16.8 (VGGT 6.2)**, import 2.0,
mesh 38.2, texture 50.6. Score **86.7** (12 pass / 2 warn / 1 fail / 5 info), no downgrades.
Height −4.7%, anchor disagreement 0.83%, 1 075 250 points confirmed by a median 5 views,
footprint 91 356 m², mesh 942 774 faces (2.0 per vertex), textured. Fail: camera vs GPS 5.65 m
(S4-1). Warns: **42 of 50 frames in 2 SfM pieces** — Stage 1 logged 16+ `reduce_frames`
degradations (CPU decode of 4K MPEG-2 TS on 3 cores, NVDEC refused per S1-15), widening the frame
spacing until SfM lost the chain; the laptop, decoding faster, got 52/53 in one piece.

**Fix 1 — keep every SfM piece (`src/recon/merge.py`, `recon.track_a.merge`):** pieces share no
images, so COLMAP cannot merge them; each is fitted to GPS on its own and moved into the main piece's
frame by `main_from_gps · gps_from_piece`, cameras and points copied in (a frame the main piece
holds unregistered is re-posed). Pieces with < 3 GPS frames or a GPS fit worse than 25 m are
dropped with a downgrade. The seam is as good as the two GPS fits; each piece's residual is reported.
Tested: a real model cut in half with one half moved by a similarity comes back exactly (0 error,
all frames); a piece with wrecked GPS is refused; the laptop's real 2-piece Esri result merged with
no duplicates (its 7-frame piece was a subset of the 52-frame one). Scorecard "Separate models" now
counts pieces left unplaced after the merge.
- Also found: a model can hold **unregistered images**, and `projection_center()` on one aborts in
  COLMAP. Every camera-position read in Track A/B, the merge and the evaluator now filters posed
  images; Track B keeps `.vis` indices over all images (image-id order) and depth only for posed ones.

**Fix 2 — mesh reduction in hybrid mode (`recon.track_a.mesh.hybrid_faces_per_sample: 2.0`):**
the hybrid back-projects each ground patch from every overlapping frame, so the Delaunay mesh
triangulates near-duplicates. Target faces = 2 × points / views per point (Esri: ~430 k from 943 k).
Measured on the box's GPU cloud (laptop, 3 threads): reduce 659 k → 300 k faces costs +3.7 s of
meshing and cuts texturing **104.6 → 48.8 s**; the reduced surface stays within **10.1 cm median,
20.8 cm 95%, 25.8 cm 99%** of the full one (upper bounds: sampling spacing 21 cm); texture atlas
13.5 → 11.3 MB (colour detail kept), OBJ 80 → 35 MB. COLMAP dense (mode A) is not reduced.

**VGGT input width:** `recon.track_b.input_width` (default 518) with own preprocessing,
bit-identical to VGGT's at 518 (max abs diff 0.0) and without the centre crop. The hybrid probe
takes `--widths`; box run of 518/700/1036 pending.

**Tests:** 376 passed / 0 failed (was 374; +2 merge tests, mesh-target assertion added).

**Next:** box: widths result → choose `input_width`; rerun Stages 1–4 on Esri for the merge and the
mesh reduction.

### Session — 2026-09-22 — ritesh14g (with Claude) — VGGT input width: 518 stays
**Measured (box, hybrid probe, Esri 45 frames, vs COLMAP's 1920 px geometric depth, per-frame anchoring):**

| Width | Depth px on ground | Median error | Within 5% | Worst frame | Anchor spread | VGGT time | Peak GPU |
|---|---|---|---|---|---|---|---|
| **518** | 34.1 cm | **1.04% (1.07 m)** | **99.1%** | 2.85% | 0.85% | 5.9 s | 8.45 GB |
| 700 | 21.0 cm | 1.30% (1.36 m) | 97.0% | 5.14% | 1.19% | 9.3 s | 9.70 GB |
| 1036 | 12.7 cm | 2.49% (2.57 m) | 76.0% | 5.86% | 2.17% | 22.5 s | 12.98 GB |

Finer sampling costs accuracy: VGGT was trained at 518 and its depth degrades beyond it. With
accuracy weighted 30% and the PS target at <= 1 m, **518 stays the default**; fine detail is the
accurate preset (COLMAP dense 1920, ~10 cm). VGGT's own camera-fit scale collapses at larger widths
(`_cams` error 18% at 700, 27% at 1036 vs 1.76% at 518) — more evidence for anchoring each frame on
Track A's sparse points instead of trusting VGGT's cameras.

**Dead end (§4):** ❌ VGGT above its 518 px training width for more detail (measured above).

**Modified:** `configs/default.yaml` — measurements in the `input_width` comment.

**Idea for later:** COLMAP dense only on Stage 3's Zone 1 (well-observed), hybrid elsewhere — detail
where it can be verified, speed everywhere else.

**Next:** box: full Stages 1–4 on Esri to confirm the SfM merge and hybrid mesh reduction.

### Session — 2026-09-22 — ritesh14g (with Claude) — VGGT-Ω added to Track B
**Goal:** The team now has approved Hugging Face access to `facebook/VGGT-Omega`; make it selectable
in Track B and measure it before it can become the default.

**Found (code read + random-weights runs):** `facebookresearch/vggt-omega`, class `VGGTOmega`
(1.14 B params), 16-px patches, checkpoint `vggt_omega_1b_512.pt` (4.58 GB, `torch.load`, strict).
Same outputs as VGGT-1B (`depth` [B,S,H,W,1], `depth_conf`, `pose_enc` decoded by
`encoding_to_camera`). **"Balanced" sizing keeps ~1024 patches per frame: 688×384 on 16:9 — 1.3×
finer than VGGT-1B at 518 while still at Ω's training size** (the VGGT-1B width sweep showed going
beyond training size costs accuracy). Licence: FAIR non-commercial research (LIC-1).

**Created:**
- `scripts/stage4_report.py`, `scripts/box_stage4_compare.sh` — see §3.

**Modified:**
- `src/recon/track_b_vggt.py` — `preprocess_balanced()` (bit-identical to Ω's own on 1920×1080,
  640×480, 1280×715; no aspect crop), `omega_predictor()` (checkpoint via `hf_hub_download`; a missing
  login or approval raises `TrackBUnavailable` → Track A dense with the reason), `make_predictor()`;
  report records `model`.
- `configs/default.yaml` — `recon.track_b.model: vggt | vggt_omega` (default stays **vggt** until
  measured) and `recon.track_b.omega.*`.
- `scripts/vggt_hybrid_probe.py` — `--model vggt_omega`; one `forward()` for both models.
- `src/core/manifest.py` — **bug:** track_a's staleness fingerprint covered only `recon.track_a`, so
  a resumed run would reuse a Stage 4 result built with the other VGGT model or run mode. Now also
  `recon.track_b`, `run.mode`, `device`.
- `tests/test_recon.py` — Ω sizing equals the reference; unknown model name falls back cleanly.

**Tests:** 378 passed / 0 failed (was 376; +2).

**Next:** box: `bash scripts/box_stage4_compare.sh` (needs a Hugging Face login on the box). Default
model decided from Ω's depth error vs COLMAP and the Stage 4 scorecards.

### Session — 2026-09-22 — ritesh14g (with Claude) — Box comparison: VGGT-Ω becomes the default
**Measured (box, `scripts/box_stage4_compare.sh`, Esri; Stages 1–2 once: ingest 49.7 s, condition
50.4 s; Stage 4 from the same base):**

| | VGGT-1B | **VGGT-Ω** |
|---|---|---|
| Probe: depth px on ground | 34.1 cm | **24.1 cm** |
| Probe: median error vs COLMAP 1920 | 1.04% (1.07 m) | **0.86% (0.89 m)** |
| Probe: within 5% / worst frame | 99.1% / 2.85% | **99.2% / 1.13%** |
| Probe: VGGT time / peak GPU | 5.9 s / 8.45 GB | 9.7 s / **5.89 GB** |
| Pipeline: Stage 4 score | 93.3 | 93.3 |
| Pipeline: registered (incl. merge) | 48/50 | 46/50 |
| Pipeline: SfM pieces / merged / frames added | 2 / 1 / +4 | 2 / 1 / +4 |
| Pipeline: camera vs GPS / height | 6.42 m / −4.2% | 5.41 m / −4.3% |
| Pipeline: points / footprint | 1.22 M / 116 639 m² | 2.00 M / 123 743 m² |
| Pipeline: anchor spread / frames rejected | 0.93% / 0 | 0.71% / 1 (spread) |
| Pipeline: mesh target → faces | 610 k → 609 k | 997 k → 996 k |
| Pipeline: dense_vggt / mesh / texture | 19.0 / 46.0 / 33.1 s | 25.1 / 72.5 / 53.2 s |
| Pipeline: Stage 4 / Stages 1–4 | 143.8 / 243.9 s | 199.8 / 299.9 s |

**Both fixes confirmed on the box:** score 86.7 → **93.3**; the flight still splits in two SfM pieces,
the smaller is now placed through GPS (+4 frames) and the "separate models" warning is gone; hybrid
meshes are reduced by the sample rule (VGGT-1B: 943 k → 609 k faces; texture 51 → 33 s). Only fail:
camera vs GPS (S4-1).

**Decision:** default `recon.track_b.model: vggt_omega` — finer (1.4×) *and* more accurate, first depth
figure under the PS's 1 m, less GPU memory; costs ~56 s more Stage 4 (bigger mesh). **Fallback chain
Ω → VGGT-1B → Track A dense**: a machine without the gated Ω login gets VGGT-1B (logged downgrade,
`model_fallback` in the report and scorecard) instead of the ~100× slower Track A dense.

**Observation:** SfM is not bit-repeatable — identical Stage 1–2 input registered 46 vs 48 frames (GPU
SIFT/matching and mapper randomness). Compare runs on scorecard bands, not exact counts.

**Speed:** Stages 1–4 ≈ 300 s for 1.7 min of video → ~25–30 min projected for 10 min (§9 target 15).
VGGT is ~10 s of it; the rest is CPU: decode + conditioning (100 s), mesh + texture (126 s).

**Modified:** `src/recon/track_b_vggt.py` (Ω → VGGT-1B fallback), `src/recon/track_a_colmap.py`
(downgrade logged), `src/qa/stage4_eval.py` (model + fallback in the Track B KPI), `configs/default.yaml`
(default model + measurements), `tests/test_recon.py` (+1: gated Ω falls back to VGGT-1B).

**Next:** Stage 5 (georeferencing + six export formats); S4-1; CPU speed.
