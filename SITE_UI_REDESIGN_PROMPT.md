# Public site UI redesign — staged implementation prompt

**For the agent or teammate doing the work.** This is a UI-only brief for the public demo site
(GitHub Pages). It is deliberately split into eight stages so each one is a single reviewable commit.
Do the stages in order. Do not start a stage before the previous one's **Accept** checks pass.

- Live site: https://ritesh14g.github.io/Single-Pass-3D-Model-Generation/
- Deploy branch: `gh-pages`. Code branches carry the source only (see Stage 0).
- Read `DEVLOG.md` §6 entry *"2026-09-28 (night) — demo console replaces the landing page"* before
  you touch anything: it lists the bugs already found and fixed in this exact page. Do not reintroduce them.

---

## 0. The one rule that matters

**Nothing outside the site's own files may change.** No edits to `src/`, `configs/`, `tests/`,
`ui/` (that is the Streamlit Stage Lab, a different thing), or any pipeline script. The only
non-site file you may touch is `scripts/build_site.py`, and only its **file-copy** step in Stage 0 —
never its data extraction, its asset export, `PROCESS_INFO`, `PROCESS_ORDER` or `FORMAT_INFO`.

Every number on the site is read from `data.json`. **Never compute, round, re-derive, interpolate
or hard-code a measured value in JavaScript.** If a field is missing, render `—`, not a guess.
If you find yourself wanting a number that is not in `data.json`, the answer is to not show it.

---

## 1. What exists today (scan result — trust this, verify before extending)

### Files that make up the site
| File | Status | Role |
|---|---|---|
| `data/site/index.html` | hand-written, **git-ignored** | 4 `<section class="step">` blocks + QA slide-over |
| `data/site/app.js` | hand-written, **git-ignored** | ~885 lines, ES modules, all rendering + three.js viewers |
| `data/site/styles.css` | hand-written, **git-ignored** | ~285 lines, light default + `prefers-color-scheme` dark |
| `data/site/runs/index.json` | **generated** | run list |
| `data/site/runs/<key>/data.json` | **generated** | everything measured, per run |
| `data/site/runs/<key>/assets/*` | **generated** | `mesh.glb/obj/fbx`, `cloud.ply/las`, `dsm.png`, `ortho.png` |
| `data/site/vendor/**` | **generated** (copied) | three.js r169 ESM + OrbitControls, GLTF/OBJ/PLY/FBX loaders |
| `data/site/<key>/viewer/`, `<key>/report.html` | **generated** | the pipeline's own viewer and QA report |

`.gitignore:40` ignores all of `data/site/`. **The three hand-written files are therefore untracked** —
they exist only on this laptop and on `gh-pages`. That is why Stage 0 exists and why it comes first.

### Current steps
`STEPS = ['Input', 'Input check', 'Pipeline', 'Outputs']`, rendered as `.step[data-step="0..3"]`,
one visible at a time via `goto(i)`, with a pill rail in the sticky header (`#rail`).

### Data contract you may rely on (all already in `data.json`)
```
key, run, video, device, preset, written
input_check { verdict, checks[{id,group,label,detail,fix,status,value}], sync{r_best,lag_s},
              camera{model,hfov_deg,source}, telemetry{source,rows,duration_s,has_gps,has_baro,
              has_attitude,has_focal,has_rtk}, timing_s{total}, note, recommended }
processes[{ process, stage, score, counts{pass,warn,fail,info}, what, covers, lacks,
            kpis[{label,detail,value,unit,target,status,group}] }]
scorecards{<stage_key>:{title,score,counts}}
time { stage_seconds{preflight,ingest,condition,track_a,geo,fusion,export}, total_s, video_s,
       budget_s, stage_budgets_s }
projection { video_s,total_s,budget_s,target_video_s,factor,projected_total_s,stages[],basis }
coverage, zones{coverage_pct,zone1_pct,zone2_pct,zone3_pct,gaps{count,total_m2}}
georeferencing{crs,rms_inliers_m,inliers,...}
accuracy{reference{source},local{points_after_tile_offset{...},doming{span_m,sag_m,tilt_m_per_km}}}
benchmarks{degradation{rows[],severities[]}}
limitations[]
formats{ produced[], info{}, files{<name>:bytes}, tiles,
         assets{mesh{glb,obj,fbx,faces,vertices,centre}, points{ply,las,points_web,points_full,
                classes,zone_counts,confidence_mean}, rasters{dsm,ortho}} }
metadata{crs,coordinate_frames,confidence,coverage,files}
manifest_stages{<key>:{status,duration_s}}   # includes track_b / refine_ba / qa, some "skipped"
viewer{faces,...}
```
Runs present: **`esri`** (MISB KLV telemetry inside the video stream) and **`dji`** (DJI_0047, flight-log CSV).

### Landmines already paid for once — do not regress
1. `[hidden] { display: none !important; }` must stay. An explicit `display` in any later rule beats
   the `hidden` attribute, and that is what left the loading overlay stuck on screen.
2. glTF custom attributes arrive **lower-cased**: read `_confidence` / `_zone`, keep the
   `attrs._confidence || attrs._CONFIDENCE` fallback.
3. Never put `display: grid` on a `<summary>` — it collapses the rows inside. Use flex.
4. Camera framing must fit by **field of view** (`frameObject()`), not by a fixed distance.

---

## 2. Target design

**Dark, technical, colour-coded.** One dark theme only — drop the light palette and the
`prefers-color-scheme` switch. Deep neutral background, luminous blue as the primary accent, and a
**distinct hue per pipeline stage** so a viewer can tell at a glance which step they are on.

Reserve semantics strictly:
- **Status colours are reserved.** Green = met, amber = near target, red = below target, grey = recorded
  for information. No stage accent, chart, chip or decoration may use those three hues for anything else.
- **Zone colours are reserved** and must keep their current meaning everywhere: Zone 1 green `#29a645`,
  Zone 2 amber `#e8a521`, Zone 3 red `#d93838`.
- Stage accents therefore live in the cool/violet/teal range. Suggested, refine as you build:

| Page | Accent | Token |
|---|---|---|
| Landing | `#38bdf8` sky | `--s-home` |
| Input | `#60a5fa` blue | `--s-input` |
| Input check | `#818cf8` indigo | `--s-check` |
| Ingest | `#22d3ee` cyan | `--s-ingest` |
| Conditioning | `#a78bfa` violet | `--s-condition` |
| Reconstruction | `#f472b6` pink | `--s-recon` |
| Occlusion | `#2dd4bf` teal | `--s-fusion` |
| Georeferencing | `#7dd3fc` pale blue | `--s-geo` |
| Outputs | `#c4b5fd` lavender | `--s-export` |

Every accent must hold **≥ 4.5:1** against the page background for text and **≥ 3:1** for a
progress-bar fill or a border. Check it; do not eyeball it.

**Voice on every page:** technical keywords, present tense, no marketing. Each page carries a row of
keyword chips naming the real algorithms (see Stage 5), then **two to four short sentences or a
four-line bullet list** — never a paragraph block. Descriptions say what the step does and what it
measures. The existing honest framing (what a score does *not* cover) stays.

---

## Stage 0 — Make the site source committable *(no visual change)*

**Goal:** the three hand-written files become tracked source, so Stages 1–7 are real commits.

**Do**
1. Create a tracked directory `site/` at the repo root and move the hand-written files into it:
   `site/index.html`, `site/app.js`, `site/styles.css`. Copy them from `data/site/` as they are today —
   byte-for-byte, no edits.
2. In `scripts/build_site.py`, at the end of `main()` beside the existing `vendor` copy, copy every
   file under `site/` into `SITE` (`data/site/`), preserving relative paths, overwriting. Keep it to
   `shutil.copy2` in a loop; log one line with the count. Nothing else in that file changes.
3. Leave `.gitignore:40` (`data/site/`) exactly as it is — the build output stays ignored.
4. Update `DEVLOG.md` §3 repository map with the `site/` row, and append the §6 session entry.

**Don't** move or rename anything under `data/site/runs/`, `data/site/vendor/` or `data/site/<key>/`.

**Accept**
- `git status` shows `site/index.html`, `site/app.js`, `site/styles.css` as new tracked files.
- `.venv\Scripts\python scripts/build_site.py` finishes and `data/site/index.html` is identical to
  `site/index.html`.
- The site still loads and behaves exactly as before — all four steps, both runs, all six formats.
- `.venv\Scripts\python -m pytest tests -q` unchanged (it does not cover this site; confirm it is still green).

**Commit:** `chore(site): track the demo-console source under site/ and copy it in build_site`

> From here on, **edit `site/*` and re-run `build_site.py`** to preview. Never edit `data/site/*` directly.

---

## Stage 1 — Dark theme and the design system *(existing four steps keep working)*

**Goal:** one dark palette, stage accent tokens, and the shared components later stages need.

**Do**
1. Replace the `:root` block in `site/styles.css` with a single dark palette. Delete the
   `@media (prefers-color-scheme: dark)` override and the `[data-theme="light"]` guard — there is one theme.
   Set an explicit `background` on `body`.
2. Add the nine `--s-*` stage accent tokens plus, for each, a `-soft` (≈12% over the panel) and a
   `-line` (≈40%) variant. Add `--accent: var(--s-home)` as the page default.
3. Add a `[data-accent="<page>"]` mechanism: setting that attribute on `<main>` or the active section
   rebinds `--accent` to that page's stage token, so every accent-coloured component recolours itself
   with no per-component code.
4. Update `setBg()` in `app.js` to a single dark scene background (drop the `matchMedia` branch) and
   pick a value that keeps the vertex-coloured mesh readable.
5. New components, in CSS with tiny render helpers in `app.js`:
   - **`.chips` / `.chip`** — keyword chips: monospace, small caps tracking, accent-tinted border,
     transparent fill. A `.chip[title]` may carry a one-line expansion.
   - **`.progress`** — the segmented bar for Stage 4: one `<li>` per step, fill in the step's own accent,
     done / current / upcoming states, current step labelled and wider.
   - **`.statgrid` / `.stat`** — number-first tile: value (tabular numerals, large), label above, one
     line of context below.
   - **`.kpirow`** — compact measured-value row: label + detail, value + unit + target, status tag.
     Reuse for both the input check and the stage pages.
6. Re-tune the existing `.panel`, `.btn`, `.tag`, `.callout`, `.tablewrap`, `.glass`, `.stage3d`,
   `.legend` for the dark palette. Keep every class name — `app.js` depends on them.

**Don't** change any DOM structure, any step, or any string of copy in this stage. This is purely visual.

**Accept**
- Whole site renders dark with no light flashes, including the QA slide-over and the loading overlay.
- No console errors. All six formats still load and all colour modes still switch.
- Contrast measured for every accent, both as text and as a bar fill.
- Zone and status colours unchanged from their current values.

**Commit:** `feat(site): dark theme, stage accent tokens, chips / progress / stat components`

---

## Stage 2 — Landing page: hero with the blue node model

**Goal:** a new first page. Left: a blue node-and-edge 3D form that reads as a mountain ridge with a
geolocation marker. Right: what the system is, in a few lines, and the entry buttons.

**Do**
1. Add `<section class="step" data-step="home">` as the **first** section in `site/index.html`, and make
   it the default view. Two-column grid: canvas left, copy right; stacks to one column under 900 px with
   the canvas first and capped at ~46vh.
2. Right column: product name, one line of provenance (`SIH 2026 · PS-17 · NTRO · working prototype`),
   a headline, **three to four sentences maximum** on what it does, a keyword chip row
   (`single-pass SfM`, `GPS-anchored georeferencing`, `feed-forward monocular depth`,
   `confidence-graded surfaces`, `six export formats`), a primary **"Run a flight through the pipeline →"**
   that enters the wizard at the Input step, and a ghost **"See what it produces"** that jumps to Outputs.
   Below them a thin strip of live figures read from `runs/index.json` — nothing invented.
3. Hero geometry, **generated procedurally in JavaScript** — never a downloaded model, the page must paint
   immediately:
   - A ridge heightfield on roughly a 44 × 32 grid (≈1,400 nodes): two or three summed sine ridges plus
     value noise, one clear dominant peak and a saddle, normalised so the silhouette reads as a mountain.
   - `THREE.Points` for the nodes, 2–4 px, slight size attenuation, additive blending.
   - `THREE.LineSegments` for the grid edges, opacity ≈ 0.22, additive — this is what makes it read as a mesh
     of nodes rather than a dot cloud.
   - Height-ramped blue: deep indigo `#1b3a8f` in the valleys → `#38bdf8` on the slopes → near-white at the
     peak. Set the colours as a vertex-colour attribute; do not tint with a light.
   - A **geolocation marker** on the saddle: a thin vertical beam, a small octahedron head, and one pulsing
     ring on the ground plane. Plus two concentric graticule rings and a faint meridian arc so the
     "on the map" reading is unmistakable.
   - Motion: continuous Y rotation ≈ 0.05 rad/s, pointer parallax within ±6°, the marker ring pulsing on a
     ~2.4 s cycle. A slow one-time settle-in on load is welcome; nothing that loops distractingly.
4. **Resource rules — these are not optional.** The outputs page already owns a WebGL context; a second
   one that never sleeps will fight it.
   - Give the hero its own renderer, created lazily on first view.
   - Pause with `renderer.setAnimationLoop(null)` whenever the landing section is not the active step or
     an `IntersectionObserver` reports it off-screen; resume on return.
   - Cap `setPixelRatio` at 2.
   - Honour `prefers-reduced-motion: reduce` — render one static frame, no rotation, no pulse.
   - If WebGL is unavailable or the context is lost, hide the canvas and show an inline SVG fallback of the
     same ridge silhouette. The page must never show an empty box.
5. Routing: the landing page is **not** a numbered step and gets no progress bar. Clicking the wordmark
   in the header returns to it.

**Don't** load any file from `runs/*/assets/` for the hero. Don't animate on the main thread while the
outputs viewer is active.

**Accept**
- Headless Chrome (SwiftShader): the hero renders, no console errors, first paint is not blocked on it.
- Navigating landing → outputs → landing leaves exactly one live animation loop each time; the WebGL
  context count never grows (check by navigating back and forth ten times).
- Reduced-motion and WebGL-off both render something sensible.
- Readable at 360 px wide with a 16 px gutter and no horizontal scroll.

**Commit:** `feat(site): landing hero — procedural blue node terrain with a geolocation marker`

---

## Stage 3 — Landing page: what it does, and what it produced

**Goal:** below the hero, the case for the project, entirely from measured data.

**Do**, in this order down the page:
1. **The problem, in three lines.** A single pass sees each surface from few angles, so a system either
   leaves holes or invents geometry. Name the approach: measure what was seen, infer only where inference
   is checked against measurement, and report the rest as gaps.
2. **Capability cards** (six or eight, `.statgrid`-style, each with an accent, a two-line description and a
   keyword chip row): input check before any compute · overlap-targeted frame selection · frame conditioning ·
   GPS-anchored SfM + dense MVS · anchored monocular depth for thin coverage · three-zone confidence
   labelling with gap outlines · georeferencing with UTM + EGM96 · six verified export formats + web viewer.
3. **Targets vs measured.** A table from the PS §1.4 targets against what the runs actually did — formats,
   viewer, mesh, coverage, accuracy, speed — each row marked met / partial / not met **with the reason**,
   reading `coverage`, `georeferencing`, `projection` and `formats.produced` from both `data.json` files.
   Keep the existing honesty: a partial stays partial.
4. **The two prepared flights**, side by side: length, processing time, frames placed, coverage,
   cameras vs GPS, and the per-stage scores from `scorecards`. Say in one line why the low flight costs
   more time than the high one. Each card enters the wizard with that run selected.
5. **Applications** — one compact row of the PS-named uses (border mapping, disaster assessment, urban
   planning, infrastructure inspection, digital twins). Text only, no claims of deployment.
6. Footer keeps its current provenance sentence, the hardware line and the source-code link.

**Don't** write a number that is not in `data.json` or `runs/index.json`. Don't add testimonials,
made-up logos, or a "trusted by" strip.

**Accept**
- Every figure traceable to a field in `data.json`; spot-check five against `data/site/runs/*/data.json`.
- With one run's `data.json` missing, the section still renders with `—` and no thrown error.
- No horizontal scroll at 360 px.

**Commit:** `feat(site): landing page — capabilities, targets vs measured, both flights compared`

---

## Stage 4 — One page per pipeline stage, with a segmented progress bar

**Goal:** replace the single "Pipeline" step with one page per stage, and put a barred progress
indicator across the top showing where the user is.

**Do**
1. Restructure `STEPS` into a named table. Nine entries, the landing page outside the count:

   | # | id | Title | Accent | Time source |
   |---|---|---|---|---|
   | 1 | `input` | Input | `--s-input` | — |
   | 2 | `check` | Input check | `--s-check` | `input_check.timing_s.total` |
   | 3 | `ingest` | Ingest | `--s-ingest` | `time.stage_seconds.ingest` |
   | 4 | `condition` | Conditioning | `--s-condition` | `time.stage_seconds.condition` |
   | 5 | `recon` | Reconstruction | `--s-recon` | `time.stage_seconds.track_a` |
   | 6 | `fusion` | Occlusion handling | `--s-fusion` | `time.stage_seconds.fusion` |
   | 7 | `geo` | Georeferencing | `--s-geo` | `time.stage_seconds.geo` |
   | 8 | `export` | Outputs | `--s-export` | `time.stage_seconds.export` |

2. Replace the header pill rail with the **segmented progress bar**: eight segments, each labelled,
   completed ones filled in their own accent, the current one wider and brighter with its label always
   visible, upcoming ones outlined. The bar is sticky under the header and hidden on the landing page.
   Under 720 px keep the segments but show only the current label. Every segment is clickable.
   `aria-current="step"` on the current one; the bar is a `<nav>` with a `<ol>`.
3. Each stage page also gets, in its header: the step eyebrow (`Step 4 of 8 · Conditioning`), the title,
   the measured time for that stage on this run, and a **prev / next** pair at the foot of the page.
4. `goto()` takes an id, not an index. Set `data-accent` on `<main>` so the whole page recolours.
   Scroll to top on change. Keep the existing lazy behaviour: build the outputs viewer only when the
   outputs page is entered, and never leave a running animation loop behind on any other page.
5. **Deep links must keep working.** Accept `?step=<id>` going forward, and keep the old numeric form
   working by mapping `1→input`, `2→check`, `3→recon`, `4→export` (the nearest equivalent of the old four).
   `?run=`, `?format=` and `?qa=` behave exactly as now. `?step=` with an unknown value falls back to the
   landing page rather than a blank screen.
6. The quality report stays a slide-over reachable from every page, with its four tabs unchanged.
   On a stage page, opening it should land on the tab most relevant to that stage.

**Don't** change what any KPI means, and don't drop the animated stage-timing replay — it moves to
Stage 5 as a per-page element.

**Accept**
- All eight pages reachable by bar click, by prev/next, and by `?step=<id>`; the four old numeric deep
  links still land somewhere sensible.
- Both runs selectable from any page; switching run keeps you on the same step.
- No console errors walking all eight pages forwards and backwards in both runs.
- Keyboard: the bar is tabbable, focus visible, Enter/Space activate.

**Commit:** `feat(site): one page per pipeline stage with a segmented progress bar`

---

## Stage 5 — Stage page content: keywords, short descriptions, measured results

**Goal:** fill the six pipeline pages. Each is self-contained and technical without becoming an essay.

**Every stage page has exactly this shape**
1. **Keyword chips** — the real techniques, as chips. Starting set, extend from `DEVLOG.md` §3:
   - *Ingest:* `single-pass demux`, `grab-skip streaming decode`, `NVDEC attempt → CPU fallback`,
     `overlap-targeted selection`, `forward-backward LK check`, `phase-correlation fallback`,
     `NCC peak-drop verification`, `Laplacian blur gate`, `MISB ST 0601 / STANAG 4609 KLV`, `GoPro GPMF`,
     `PX4 ULog`, `ArduPilot DataFlash`
   - *Conditioning:* `DCT blockiness detection`, `grid-keypoint veto`, `paired exposure fit`, `CLAHE`,
     `shadow masking`, `YOLOv8-seg dynamic masks`, `geometric consistency fallback`,
     `GPS envelope outliers`, `RTS-Kalman smoothing`, `baro / GPS fusion`, `RTK weighting`
   - *Reconstruction:* `COLMAP incremental SfM`, `intrinsics prior from telemetry`,
     `bundle adjustment with anisotropic GPS priors`, `GPS-prior refinement`, `PatchMatch MVS (CUDA)`,
     `VGGT-Ω feed-forward depth`, `RANSAC scale / shift anchoring`, `multi-view consistency`,
     `OpenMVS Delaunay meshing`, `MRF texturing`, `geometry gate vs GPS`
   - *Occlusion handling:* `sparse voxel grid`, `confirming-view count`, `triangulation angle`,
     `z-buffer visibility`, `Zone 1 / 2 / 3 labelling`, `depth-relative residual gate`, `C0 seam`,
     `weighted voxel fusion`, `gap polygons → GeoJSON`
   - *Georeferencing:* `RANSAC similarity fit`, `straight-path ground constraint`, `UTM zone selection`,
     `EGM96 orthometric heights`, `altitude-datum resolution`, `GCP check-point accuracy`
   - *Outputs:* `WKT compound CRS`, `ASPRS classification`, `LAS tiling`, `texel-resolution orthophoto`,
     `glTF custom attributes (_CONFIDENCE, _ZONE)`, `read-back verification`
2. **What it does** — two to four sentences, or four bullets. Not a paragraph block. State the input,
   the operation, the output, and the one decision the stage makes.
3. **Measured on this run** — a `.statgrid` of three or four headline numbers, then the `.kpirow` list.
   Build both by filtering `data.json → processes` on `p.stage`; each process keeps its
   `what` / `covers` / `lacks` text in a collapsible. `stage` values in the data are `Ingest`,
   `Conditioning`, `Reconstruction`, `Occlusion handling`, `Georeferencing and export` — the last one feeds
   two pages, so split it by process name (`Georeferencing` → geo page; `Formats (re-opened)`, `Coverage`
   → outputs page).
4. **One diagnostic visual per page**, drawn from data already present — no new data, no new fields:
   - Ingest: kept-frames / overlap summary and the blur-rejection share.
   - Conditioning: before/after style bars for the conditioning KPIs.
   - Reconstruction: frames registered, reprojection, cameras vs GPS, and the per-step time breakdown.
   - Occlusion: the Zone 1/2/3 split as a stacked bar in the reserved zone colours, gap count and area.
   - Georeferencing: horizontal / vertical residual and the CRS written.
   - Outputs: the existing viewer (Stage 6).
   If a needed field is absent for a run, render the panel with `—` and a one-line note. Never hide a panel
   silently.
5. **What this stage does not tell you** — one short callout, taken from the `lacks` text. This framing is
   the site's whole credibility; keep it.
6. **Time** — measured seconds for this stage, its `stage_budgets_s` share, and a slim bar. The animated
   replay from the old pipeline page becomes this bar's fill animation, played once on entry.

**Don't** invent an explanation for a KPI. If `PROCESS_INFO` in `build_site.py` does not describe it, show
the KPI plainly rather than writing new prose about it — and do not edit `PROCESS_INFO`.

**Accept**
- Every `processes[]` entry appears on exactly one page; none orphaned, none duplicated. Verify by
  counting: 19 processes in, 19 rendered.
- Both runs render all six pages with no `undefined`, no `NaN`, no empty panel.
- `dji`'s skipped `track_b` / `refine_ba` and `esri`'s absent lidar reference both read as explicit
  "not run" / "no reference" states, not as zeros.

**Commit:** `feat(site): per-stage pages — keywords, short descriptions, measured KPIs, diagnostics`

---

## Stage 6 — Outputs page: same behaviour, better looking

**Goal:** the format viewer keeps every capability it has and looks like the rest of the redesign.

**Do**
1. **Change no viewer logic.** `loadMesh`, `loadPoints`, `showRaster`, `meshModes`, `pointModes`,
   `frameObject`, `ramp`, `drawLegend` and especially **`parseLAS`** keep their behaviour. This is a
   presentation pass.
2. Format switcher: from a pill row to a labelled set showing, per format, its name, what it is, and
   whether it was re-opened and verified. The active format takes the page accent.
3. The 3-D stage: taller, full-width on wide screens with the fact panel as an overlay rather than a
   side column; keep the side column on narrow screens. Restyle the `.glass` overlays — mode bar,
   legend, HUD — as one consistent family. Keep the current corners and keep them clear of the canvas centre.
4. Loading and failure states: the spinner and its message get the new palette, and a load failure states
   which reader failed and on which file — as it does now, just legible.
5. The fact panel keeps every row it has (`points_full`, file sizes, classification counts, layers carried,
   CRS, resolution, bounds, tiles) but as `.kpirow`s. Keep the "Take a copy" downloads, the "open the viewer
   the pipeline wrote" link, and the "why this format is in the set" note.
6. GeoTIFF previews: give the DSM and orthophoto images a proper frame with their CRS, resolution and
   bounds beside them, and keep the height legend.

**Don't** touch the colour-mode maths (`confColour`, `zoneColour`, the height ramp, `CLASS_COLOUR`) —
those encodings are documented in the report and the legends.

**Accept**
- All six formats load in both runs; every colour mode switches; the LAS reader still decodes.
- Legends still match the encodings exactly.
- One WebGL context, and it is released or paused when leaving the page.
- Screenshot both runs × all six formats and compare against the current site: nothing lost.

**Commit:** `feat(site): outputs page restyled — format switcher, overlays, fact panel`

---

## Stage 7 — Verify, document, deploy

**Do**
1. Headless Chrome (SwiftShader), the same way the last session verified this page: walk landing plus all
   eight steps, both runs, all six formats, every colour mode, the QA slide-over and all four of its tabs.
   **Zero console errors or warnings**, and capture a screenshot of each.
2. Check the deep links used in the deck, old and new form, including `?run=dji&step=export&format=las&qa=speed`.
3. Responsive pass at 360, 768, 1024, 1440 and 1920 px. No horizontal scroll, no clipped label, no
   overlapping overlay.
4. Accessibility pass: tab through the whole site; the progress bar, format switcher, dropzones, run cards
   and slide-over are all reachable and show focus. Escape closes the slide-over. Run an automated contrast
   check over the palette and fix anything under 4.5:1 for text.
5. Confirm `.venv\Scripts\python -m pytest tests -q` is still green and that `git status` shows changes
   **only** under `site/`, plus `scripts/build_site.py` (Stage 0 only) and the two markdown files.
6. `DEVLOG.md`: append one §6 session entry covering all eight stages — Created / Modified / Deleted /
   Decisions / Dead ends / Open issues / Next — and update §3's repository map. Record the verification
   evidence (which browser, which pages, what was checked) and note that the pipeline was not touched.
7. Deploy: rebuild with `build_site.py`, publish `data/site/` to `gh-pages` the way the previous deploy did,
   then re-run the same headless checks **against the live URL**.

**Accept**
- Live site matches local on every page.
- The deck's deep links resolve on the live URL.
- DEVLOG entry present and the status board unchanged (no stage status changed — this was UI only).

**Commit:** `docs(devlog): site UI redesign — dark theme, landing page, per-stage pages` + the deploy push.

---

## 3. Definition of done

- One dark theme, nine accents, status and zone colours untouched in meaning.
- A landing page with the procedural blue node terrain, a geolocation marker, and the project's case made
  from measured data.
- Eight pages, one per pipeline step, each with keyword chips, a short technical description, the measured
  KPIs for that stage, a diagnostic, and an honest "what this does not tell you".
- A segmented progress bar showing the current step on every wizard page.
- The outputs page does everything it did before and looks like it belongs.
- `git diff` touches `site/`, `scripts/build_site.py` (copy step only), `DEVLOG.md` and this file.
  **`src/`, `configs/`, `tests/` and `ui/` are byte-identical.**
- `pytest tests -q` green.
- Every figure on the site traceable to a field in `data.json`.
