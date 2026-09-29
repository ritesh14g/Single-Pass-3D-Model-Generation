# Site v2 — product flow, light theme, and a real GPU backend

**Scope decision (2026-09-29): build S1-S5 only. S6-S9 (backend, real runs, SSH/cloud targets, full verification) are the next iteration.**
Connect GPU is built as a UI panel now: real fields, saved targets, honest not-connected states, and Run refused without a GPU.
It connects to nothing until the backend exists, and says so.

~~**Plan only. Nothing here is built yet.**~~ (S1-S5 are now being built; S6-S9 remain a plan.) Written for whoever implements it (agent or teammate),
and for ritesh14g to approve or correct first.

Supersedes `SITE_UI_REDESIGN_PROMPT.md`, which is complete and shipped (commits `b977462`…`5c9571c`).

---

## 1. What changes, in one paragraph

The site stops being a walkthrough and becomes a tool. Landing page keeps its hero and the two
tested flights and loses the three "showoff" sections. One button — **Create 3D model** — opens an
input window (video, plus telemetry marked optional), and one **Run** button drives it. The eight
wizard pages disappear: progress is a strip that fills as each stage finishes, and the output page
is the destination. Everything measured moves behind two buttons on that page, **Metrics** (top
right) and **Forecast** (bottom right), each a two-thirds scrollable table with a fixed one-third
summary. The quality report goes. The theme flips to light. And a new backend makes **Run** real:
with a GPU attached it processes the user's own video and returns results in the same format as the
two prepared flights; with no GPU it refuses and says why.

---

## 2. Decisions taken (answers given 2026-09-29)

| Question | Answer |
|---|---|
| Background | **Light, `#E3F2FD`.** Full flip from the dark theme. |
| Connect GPU | **Real backend, including remote/cloud over SSH.** |
| Someone's own video | **If a GPU is connected, actually run it** and present results in the prepared-flights format. **If not, warn that CPU would take hours and do not proceed.** |

### 2.1 The one conflict in those answers, and how I propose to resolve it

A light page and "keep the output page's colours, they look good" cannot both hold literally: that
page's look comes from dark glass panels over a dark canvas.

**Proposal: light product chrome, dark 3-D viewport.** The page, tables, cards and panels go light
(`#E3F2FD` ground, white surfaces). The 3-D stage stays a dark inset with its existing glass
overlays and legends untouched. This is what CAD and GIS tools do — QGIS, Blender, Fusion — because
a dark viewport is where a textured model reads best, and it keeps exactly the part that was called
out as good. Say the word if you would rather the viewport went light too; it is a small change to
`setBg()` and the overlay tokens, but the model will look washed out.

### 2.2 Two modes, one page

The backend cannot exist on GitHub Pages. So the page probes for one at startup and behaves
accordingly. Nothing is faked in either mode.

| | **Demo mode** (GitHub Pages, the deck link) | **Connected mode** (backend reachable) |
|---|---|---|
| Prepared flights | yes | yes |
| Connect GPU | opens, saves a target, states no backend is reachable | detects and connects for real |
| Upload + Run | refused, with the CPU warning | runs the pipeline, real streamed progress |
| Metrics / Forecast | from the prepared runs | from the run just finished |

**The deck link stays demo-only.** That is worth knowing before national screening: judges clicking
the public URL get the two flights, not a live run, unless someone runs the backend and shares it.

---

## 3. Flow

```
Landing  ──scroll──▶  the two tested flights
   │                        │
   │ [Create 3D model]      │ [Open this flight]
   ▼                        │
Input window                │   video (required) · telemetry (optional, highlighted)
   │  [Run]                 │   no prepared flights here — they live only on the landing page
   ▼                        │
Progress strip ── stages fill as each completes ──▶
   ▼                        ▼
            Output page (six formats, as today)
                 │                    │
        [Metrics] top-right    [Forecast] bottom-right
```

Back and forth: the wordmark returns to the landing page from anywhere; the input window has a back
arrow to the landing page; the output page has "New model" (back to input) and "Back to overview".
Metrics and Forecast are overlays over the output page — Escape and a close button return, they are
not pages. Deep links keep working: `?run=esri`, `?format=las`, plus new `?panel=metrics|forecast`.

---

## 4. Stages

Nine stages, each a single commit that leaves the site working. Estimates are rough relative
sizes, not hours.

### S1 — Light theme *(large)*
Re-derive every token for light. `--bg #E3F2FD`, surfaces white, ink near-black, the palette's
`#2196F3` as the primary accent and `#0D47A1` for emphasis. The nine stage accents collapse to a
**four-blue ramp from the palette** plus two neutrals, because nine hues on a light ground look like
a toy; stages are then distinguished by position and label, not by nine colours.
- Status colours (green/amber/red) and zone colours (`#29a645`/`#e8a521`/`#d93838`) keep their
  meanings but need darker variants to hold 4.5:1 on white.
- The hero must be reworked, not just recoloured: it uses additive blending, which disappears on a
  light ground. Switch to normal blending, dark-blue nodes, mid-blue edges, keep the geometry.
- The 3-D viewport and its overlays stay dark (§2.1).
- Re-run the contrast script over every pair; it already exists.

### S2 — Flow restructure *(large)*
Delete from the landing page: the eight capability cards, the targets-against-measured table, the
applications row. Keep hero and the two flights. Rename the hero button to **Create 3D model**.
- New input window: video panel, telemetry panel with a strong "optional" treatment, one **Run**.
- Delete the six per-stage pages, the segmented wizard bar and `site/stages.js`'s page rendering;
  its content moves to S3. Delete the quality report (`#qa`, `wireQA`, `buildQA`, the four tabs).
- Remove every "what this stage does not tell you" callout and every limitation line **except**
  inside Forecast.
- Progress strip: one row of stage chips that fill as stages complete. In demo mode it replays the
  prepared run's measured durations and **says so** in a small label; in connected mode it is driven
  by real events (S7).

### S3 — Metrics overlay *(medium)*
Top-right **Metrics** button on the output page. Layout: progress/stage strip pinned at the top,
left two-thirds scrollable, right one-third fixed.
- Left: one compact table per stage. Columns stay as they are — metric, value, target, status. No
  new columns. Each metric keeps its rounded box for the score, with the keyword and a one-line
  explanation beneath it, both already present in `data.json` as `label` and `detail`.
- Right: a fixed one-or-two-paragraph summary of the stage in view, written from the process's own
  `what` / `covers` text.
- Stage paging: horizontal swipe, arrow keys and the top strip all move between stages.

### S4 — Charts *(medium)*
**Load the `dataviz` skill before writing any chart code.** Inline SVG, no new dependency — the site
vendors only three.js today and a chart library would cost 100 KB+ for five charts.

| Chart | Data | Where | Note |
|---|---|---|---|
| Camera residual along the flight | `geo/camera_residuals.csv` | Georeferencing | The only genuinely continuous series; shows the doming bend. **Needs a new export** (below). |
| Per-tile placement offset | `accuracy.local.tile_offsets` (14 tiles) | Georeferencing | East/north/up per tile. |
| Accuracy by zone | `accuracy.local.points_after_tile_offset` | Occlusion | 4 categories, RMS and NMAD. |
| Conditioning on vs off | `benchmarks.degradation.rows` (13) | Conditioning | Esri only. |
| Measured vs 10-min forecast | `projection.stages` | Forecast | Paired bars. |

`scripts/build_site.py` gains one export: a downsampled `residuals.json` per run from
`geo/camera_residuals.csv`. That is the site builder, not pipeline logic.
**Every chart states "not recorded for this run" when its data is absent** — DJI has no degradation
bench and no reference survey, and must not render an empty axis.

### S5 — Forecast overlay *(small)*
Bottom-right **Forecast** button, same two-thirds/one-third layout as Metrics. Per-stage measured
versus the 10-minute projection from `projection`, the paired-bar chart, and this is the **one**
place limitations are stated — they belong with a forecast.

### S6 — Backend skeleton and Connect GPU *(large)*
New top-level `server/`. **Nothing under `src/` changes.**
- FastAPI, bound to `127.0.0.1` only, started by `python -m server`.
- `GET /api/device` reuses `src.core.device.resolve_device` and `describe_device` — real GPU name,
  memory and CUDA version, no new detection code.
- `GET /api/health` is what the page probes to pick its mode.
- **Connect GPU** panel, top-right of the landing page, two sections:
  - **This machine** — shows what `/api/device` found; nothing to configure.
  - **Remote / cloud** — host, port, user, key path, remote working directory, or a Jupyter base URL
    plus token for the institute box. `POST /api/targets` saves; `POST /api/connect` tests.
- **No-GPU state**: a clear banner — not connected, CPU would take hours — and **Run is refused**,
  as asked.
- **Credentials rule:** the page never sends a private key. It sends a *path* the backend reads, or
  uses the local ssh-agent. The token for a Jupyter target is held by the backend for the session
  and never written to disk. Say so on the panel.

### S7 — Real runs *(large)*
- `POST /api/runs` takes the uploaded video and optional telemetry, writes them under `data/raw/`,
  starts `python -m src.cli run <video> --telemetry <tel> --out <dir>` as a subprocess. One run at a
  time, guarded by the existing `src/core/runlock.py`.
- `GET /api/runs/{id}/events` streams Server-Sent Events. **Progress is real, not a timer:** the
  backend watches the run's `manifest.json`, whose `stages.<name>` carries `status`, `started_at`,
  `ended_at` and `duration_s`, and tails the JSONL log for downgrade events.
- On success the backend runs `scripts/build_site.py --run <id>=<dir>`, which produces exactly the
  `data.json` and web assets the prepared flights use — so **the output page needs no special case**
  for a live run. This is the mechanism that satisfies "same format as the two present videos".
- Failures surface the stage that failed and the log tail; the input check refusing a file is a
  first-class result, not an error.

### S8 — Remote and cloud targets *(large)*
- **SSH/SFTP**: upload the clip, run the CLI remotely, stream progress by tailing the remote
  `manifest.json`, pull the run folder back, then build the site locally. Mirrors what
  `scripts/box_upload.py` and `scripts/box_collect.sh` already do.
- **Jupyter/contents API**: the institute box route, reusing the resumable part upload already
  written in `scripts/box_upload.py`.
- Kaggle is out of scope unless asked; `scripts/kaggle_ctl.py` already drives it from the terminal.

### S9 — Verify, document, deploy *(medium)*
- Re-point the existing browser suites (`stage2…stage7.mjs`) at the new flow; they are the
  regression net and most of their checks still apply.
- New checks: mode probe both ways, Run refused without a GPU, Metrics and Forecast layout and
  paging, charts present and absent, light-theme contrast, keyboard and 360–1920 px.
- Backend: `pytest` for the API with the pipeline mocked, plus **one real end-to-end run on the
  synthetic fixtures** from `src/qa/synthetic.py` (a real clip on this CPU would take hours).
- `DEVLOG.md` entries per stage, as before. Deploy the static half to `gh-pages`; the backend is not
  deployed anywhere.

---

## 5. What I cannot verify, and should not claim

- **SSH and Jupyter targets (S8).** No box access from here. I can write them against the same API
  `scripts/box_upload.py` uses and test the failure paths, but "connects to the institute box" will
  be untested until someone runs it there. I will log it that way.
- **A real GPU run (S7).** This laptop has no GPU. I can prove the plumbing end to end on CPU with
  synthetic fixtures, and prove the no-GPU refusal, but not a real CUDA run.
- **Timings.** Anything I measure in headless Chrome is software rendering.

---

## 6. Risks worth naming now

1. **Scope.** S6–S8 are a new backend — comparable in size to everything built so far. If time is
   short before screening, S1–S5 alone are a complete, shippable improvement, and S6 gives an honest
   Connect GPU panel without the run machinery.
2. **The deck link.** It stays demo-only (§2.2). If judges must see a live run, someone runs the
   backend on the box during the demo.
3. **Light theme regression.** The hero and the six-format page were tuned for dark. Expect the hero
   to need real rework, not recolouring.
4. **`src/qa/report.py:272`** still measures Reconstruction against `track_a_mvs` alone, so the
   pipeline's own HTML report disagrees with the site about Esri being over budget. Pipeline-side,
   out of scope here, but it should go on the issue list.

---

## 7. Order I suggest

**S1 → S2 → S3 → S4 → S5**, then stop and look. That is the whole product experience, and it ships
without a backend. Then **S6 → S7 → S8 → S9** if the time is there.

Approve, or tell me what to change, and I will start at S1.
