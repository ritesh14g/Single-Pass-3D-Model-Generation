/* Content of the six pipeline pages: keyword chips, a short "what it does", the measured headline numbers,
   one diagnostic, every measured check grouped by process, and what the stage does not tell you.

   Everything measured is read from the run's data.json and printed as it is. Stat labels and their context
   lines are the KPI's own `label` and `detail`, so no meaning is written here that the pipeline did not state.
   The authored text is the technique keywords and the short descriptions; both are about the method, not about
   any run. Where a run has no value (a skipped step, no reference survey), the page says so instead of showing 0. */
import { $, el, esc, num, clock, val, stat, chips, kpiRow, scoreColour } from './ui.js';

const ZONE = { 1: '#29a645', 2: '#e8a521', 3: '#d93838' };          // reserved: same colours the viewers and legends use

// which page each process belongs to; anything not listed falls back to its stage label, so none is dropped
const STAGE_PAGE = { Ingest: 'ingest', Conditioning: 'condition', Reconstruction: 'recon', 'Occlusion handling': 'fusion', 'Georeferencing and export': 'export' };
const PROCESS_PAGE = { Georeferencing: 'geo' };
const pageFor = (p) => PROCESS_PAGE[p.process] || STAGE_PAGE[p.stage] || 'export';

// ── lookups ─────────────────────────────────────────────────────────────────────
const proc = (d, name) => (d.processes || []).find((p) => p.process === name);
const kpi = (d, name, key) => proc(d, name)?.kpis.find((k) => k.key === key);
const pct = (v) => (v == null ? '—' : `${num(v * 100, 1)}%`);                // a fraction the pipeline reports as 0..1, shown as a percentage
const big = (v) => (v == null ? '—' : Number(v).toLocaleString());                 // counts with thousands separators
const shown = (k, mode) => (!k ? '—' : mode === 'pct' ? pct(k.value) : Number.isInteger(k.value) && Math.abs(k.value) >= 10000 ? `${big(k.value)}${k.unit ? ' ' + k.unit : ''}` : `${val(k.value)}${k.unit ? ' ' + k.unit : ''}`);
// A KPI the run did not record still gets a readable label and says so, rather than a raw key beside a dash.
const human = (key) => key.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
const S = (d, name, key, mode) => { const k = kpi(d, name, key); return stat(k?.label || human(key), shown(k, mode), k ? (k.detail || '') : 'Not recorded for this run.'); };

// ── diagnostics ─────────────────────────────────────────────────────────────────
// A KPI as a labelled meter: the fill is the measured share, the colour is the KPI's own status.
function meter(d, name, key, mode) {
  const k = kpi(d, name, key); if (!k) return '';
  const raw = mode === 'pct' ? (k.value == null ? null : k.value * 100) : (k.unit === '%' ? k.value : null);
  const w = raw == null ? 0 : Math.max(0, Math.min(100, raw));
  return `<div class="meter"><div class="lbl">${esc(k.label)}</div>` +
    `<div class="track"><i class="fill ${esc(k.status)}" style="width:${w}%"></i></div>` +
    `<div class="val">${esc(shown(k, mode))}</div><div class="tgt">target ${esc(k.target || '—')}</div></div>`;
}
const viz = (title, note, body) => `<div class="viz"><div class="viz-h">${esc(title)}</div>${note ? `<p class="hint">${note}</p>` : ''}${body}</div>`;

function zoneBar(d) {
  const z = d.zones || {}, parts = [[1, z.zone1_pct, 'Zone 1 · measured'], [2, z.zone2_pct, 'Zone 2 · thin or anchored'], [3, z.zone3_pct, 'Zone 3 · never observed']];
  if (parts.every(([, v]) => v == null)) return viz('Observation zones', '', '<p class="hint">No zone map for this run.</p>');
  const bar = parts.map(([n, v]) => `<i style="flex:${Math.max(v || 0, 0.001)} 1 0;background:${ZONE[n]}" title="${esc(`Zone ${n}: ${num(v, 1)}%`)}"></i>`).join('');
  const legend = parts.map(([n, v, l]) => `<div class="row"><span class="sw" style="background:${ZONE[n]}"></span>${esc(l)} <b>${num(v, 1)}%</b></div>`).join('');
  const g = z.gaps || {};
  return viz('Observation zones', 'Share of the ground the cameras saw, by how well each part was observed.',
    `<div class="zonebar">${bar}</div><div class="zlegend">${legend}</div>` +
    `<p class="hint" style="margin:10px 0 0">${g.count != null ? `<b>${num(g.count, 0)}</b> gap outlines covering <b>${num(g.total_m2, 0)} m²</b>, the largest <b>${num(g.largest_m2, 0)} m²</b>, written to gaps.geojson and never filled.` : 'Gap outlines are not recorded for this run.'}</p>`);
}

function timeBars(d, currentKey) {
  const s = d.time?.stage_seconds || {}, names = { preflight: 'Input check', ingest: 'Ingest', condition: 'Conditioning', track_a: 'Reconstruction', fusion: 'Occlusion handling', geo: 'Georeferencing', export: 'Outputs' };
  const peak = Math.max(...Object.values(s), 1);
  const rows = Object.entries(s).map(([k, v]) => `<div class="tbar${k === currentKey ? ' cur' : ''}"><div class="lbl">${esc(names[k] || k)}</div>` +
    `<div class="track"><i style="width:${(v / peak) * 100}%"></i></div><div class="val">${clock(v)}</div></div>`).join('');
  return viz('Where the time went', 'Measured seconds per stage on this run; bar lengths are relative to the longest.', rows);
}

function manifestPanel(d) {
  const m = d.manifest_stages || {}, rows = [['track_b', 'Track B depth, as its own step'], ['refine_ba', 'Refinement pass, as its own step'], ['track_a', 'Track A (SfM, dense, mesh, texture)']]
    .filter(([k]) => m[k]).map(([k, label]) => {
      const st = m[k].status, ran = st === 'done';
      return `<tr><td>${esc(label)}</td><td class="num">${ran ? clock(m[k].duration_s) : '—'}</td><td><span class="tag ${ran ? 'pass' : 'info'}">${ran ? 'Done' : 'Not run as its own step'}</span></td></tr>`;
    }).join('');
  if (!rows) return '';
  return viz('Steps in the run manifest', 'The manifest lists Track B depth and the refinement pass as steps of their own and records them as skipped; their results are scored inside Track A, in the processes below.',
    `<div class="tablewrap"><table class="data"><thead><tr><th>Step</th><th>Time</th><th>Recorded as</th></tr></thead><tbody>${rows}</tbody></table></div>`);
}

function residuals(d) {
  const g = d.georeferencing || {}, rows = [['Horizontal', g.horizontal_rms_m], ['Vertical', g.vertical_rms_m], ['All axes', g.rms_all_m]];
  const target = 1;                                                    // PS-17 spatial-accuracy target, metres: the yardstick, not a measurement
  const peak = Math.max(target * 1.25, ...rows.map(([, v]) => v || 0));
  const bars = rows.map(([l, v]) => `<div class="rbar"><div class="lbl">${l}</div><div class="track"><i style="width:${v == null ? 0 : (v / peak) * 100}%"></i><b class="mark" style="left:${(target / peak) * 100}%"></b></div><div class="val">${v == null ? '—' : num(v, 2) + ' m'}</div></div>`).join('');
  const a = d.accuracy || {}, l = a.local?.points_after_tile_offset;
  let ref;
  if (l) {
    const R = [['zone1_measured', 'Zone 1 · measured'], ['zone2_measured', 'Zone 2 · measured, fewer views'], ['zone2_fill', 'Zone 2 · filled from anchored depth'], ['all_points', 'All points']]
      .filter(([k]) => l[k]).map(([k, lab]) => `<tr><td>${lab}</td><td class="num">${num(l[k].rms_m, 2)}</td><td class="num">${num(l[k].nmad_m, 2)}</td><td class="num">${num(l[k].n, 0)}</td></tr>`).join('');
    ref = viz('Shape against an independent lidar survey', `${esc(a.reference?.source || 'A reference surface')}, after removing each ${num(a.local.tile_m, 0)} m tile's own placement offset so that what remains is shape alone.`,
      `<div class="tablewrap"><table class="data"><thead><tr><th>Surface</th><th>RMS (m)</th><th>NMAD (m)</th><th>Points</th></tr></thead><tbody>${R}</tbody></table></div>`);
  } else {
    ref = viz('Shape against an independent lidar survey', '', '<p class="hint" style="margin:0"><b>No reference survey.</b> No independent lidar or surveyed points cover this site, so shape is not compared here. The honest measures for this flight are the residuals above and the share of the scene that was observed.</p>');
  }
  return viz('Camera positions against the GPS track', `Root-mean-square residual after the fit; the marker is the problem statement's 1 m target. Coordinate system written: <b>${esc(g.crs || '—')}</b>.`, bars) + ref;
}

// ── the six pages ───────────────────────────────────────────────────────────────
const does = (a) => `<div class="does">${a.map(([k, t]) => `<div><div class="k">${k}</div><div class="t">${esc(t)}</div></div>`).join('')}</div>`;
const PAGES = {
  ingest: {
    chips: ['single-pass demux', 'grab-skip streaming decode', ['NVDEC attempt → CPU fallback', 'Hardware decode is tried first; a failure is logged and decoding continues on the CPU'], 'overlap-targeted selection',
      'forward-backward LK check', 'phase-correlation fallback', 'NCC peak-drop verification', 'Laplacian blur gate', 'MISB ST 0601 / STANAG 4609 KLV', 'GoPro GPMF', 'PX4 ULog', 'ArduPilot DataFlash'],
    does: [['Input', 'One drone video in any container FFmpeg reads, and the position log that came with it, in 12+ formats detected by content.'],
      ['Operation', 'Streams the decode, skips ahead by a predicted stride, checks each candidate\'s overlap with the last kept frame by optical flow, and gates out frames too blurred to use.'],
      ['Output', 'A short set of sharp frames at a controlled overlap, each with its GPS fix aligned to the video clock.'],
      ['Decision', 'Which frames to keep: chosen to hit a target overlap, never at a fixed interval.']],
    stats: (d) => [S(d, 'Frame selection', 'frames_selected'), S(d, 'Frame selection', 'overlap_median'), S(d, 'Telemetry', 'gps_frame_coverage', 'pct'), S(d, 'Blur gate', 'blur_reject_fraction', 'pct')],
    visual: (d) => viz('Selection, blur and telemetry', 'Each bar is the measured share; colour is the check\'s own status.',
      meter(d, 'Frame selection', 'overlap_median', 'pct') + meter(d, 'Frame selection', 'decode_fraction', 'pct') + meter(d, 'Blur gate', 'blur_reject_fraction', 'pct') + meter(d, 'Telemetry', 'gps_frame_coverage', 'pct')),
    notTell: 'Frame selection',
  },
  condition: {
    chips: ['8×8 block-boundary gradient energy', 'edge-preserving bilateral / guided filter', 'grid-keypoint veto', 'paired exposure fit', 'CLAHE', 'shadow masking',
      'YOLOv8-seg dynamic masks', 'geometric-consistency fallback', 'GPS envelope outliers', 'RTS-Kalman smoothing', 'baro / GPS fusion', 'RTK weighting'],
    does: [['Input', 'The kept frames and the raw GPS track.'],
      ['Operation', 'Detects compression blocking and removes it with an edge-preserving filter, evens exposure across the flight, marks shadows and moving objects, and filters position outliers before smoothing.'],
      ['Output', 'Cleaned frames with masks, and a physically plausible GPS track.'],
      ['Decision', 'Per frame, whether blocking is bad enough to correct, and which keypoints sitting on the block grid to veto.']],
    stats: (d) => [S(d, 'Artifact suppression', 'blockiness_reduction', 'pct'), S(d, 'Illumination', 'shadow_fraction_mean', 'pct'), S(d, 'GPS conditioning', 'gps_outlier_fraction', 'pct'), S(d, 'Dynamic masking', 'dynamic_frames_with_movers', 'pct')],
    visual: (d) => viz('Artifacts, illumination and GPS', 'Each bar is the measured share; colour is the check\'s own status.',
      meter(d, 'Artifact suppression', 'artifact_correct_fraction', 'pct') + meter(d, 'Artifact suppression', 'blockiness_reduction', 'pct') + meter(d, 'Illumination', 'shadow_fraction_mean', 'pct') +
      meter(d, 'Illumination', 'saturated_fraction', 'pct') + meter(d, 'Illumination', 'exposure_clamped_fraction', 'pct') + meter(d, 'GPS conditioning', 'gps_outlier_fraction', 'pct')),
    notTell: 'Illumination',
  },
  recon: {
    chips: ['COLMAP incremental SfM', 'intrinsics prior from telemetry', 'GPS-prior bundle adjustment', 'anisotropic GPS pose priors', 'PatchMatch MVS (CUDA)', 'VGGT-Ω feed-forward depth',
      'RANSAC scale / shift anchoring', 'multi-view consistency', 'OpenMVS Delaunay mesh', 'OpenMVS texturing', 'geometry gate vs GPS'],
    does: [['Input', 'The conditioned frames, their masks, and the filtered GPS track.'],
      ['Operation', 'Solves every camera by structure-from-motion with a focal prior from telemetry, refines against the GPS track, adds anchored learned depth where stereo is thin, then meshes and textures the dense cloud.'],
      ['Output', 'Camera poses, a dense point cloud and a textured triangle mesh.'],
      ['Decision', 'Whether the solution is trusted at all: a geometry gate against GPS caps the stage\'s score when the cameras disagree with the track.']],
    stats: (d) => [S(d, 'Sparse (SfM)', 'registered_fraction', 'pct'), S(d, 'Sparse (SfM)', 'reproj_px'), S(d, 'Metric (vs telemetry)', 'cam_vs_gps_rms_m'), S(d, 'Dense and mesh', 'dense_points')],
    visual: (d) => timeBars(d, 'track_a') + manifestPanel(d),
    notTell: 'Metric (vs telemetry)',
  },
  fusion: {
    chips: ['sparse voxel grid', 'confirming-view count', 'triangulation angle', 'z-buffer visibility', 'Zone 1 / 2 / 3 labelling', 'anchored monocular depth',
      'depth-relative residual gate', 'C0 seam', 'weighted voxel fusion', 'gap polygons → GeoJSON'],
    does: [['Input', 'The cameras and the dense cloud with the views that confirm each point, in metres.'],
      ['Operation', 'Counts the cameras that confirm each voxel and the angle they span, labels three zones, fills Zone 2 from learned depth scaled to the measured surface, and outlines Zone 3.'],
      ['Output', 'A zone map, gap outlines as GeoJSON, and a zone label on every point and face.'],
      ['Decision', 'Whether a surface is measured, inferred or never seen; a Zone 1 voxel is never overwritten.']],
    stats: (d) => [S(d, 'Coverage and gaps (§6.4)', 'coverage_pct'), S(d, 'Zones (§6.1–6.2)', 'zone1_pct'), stat('Gap outlines', big(d.zones?.gaps?.count), 'areas the cameras never observed, reported and not filled'), S(d, 'Zone 2 anchoring (§6.3)', 'holdout_error_pct')],
    visual: (d) => zoneBar(d),
    notTell: 'Coverage and gaps (§6.4)',
  },
  geo: {
    chips: ['RANSAC similarity fit', 'straight-path ground constraint', 'UTM zone selection', 'EGM96 orthometric heights', 'altitude-datum resolution', 'GCP check-point accuracy', 'WKT compound CRS'],
    does: [['Input', 'The reconstructed camera path and the GPS track.'],
      ['Operation', 'Fits a similarity transform between them with RANSAC, converts to the UTM zone of the flight, and measures what the altitudes were referenced to.'],
      ['Output', 'The model in a projected coordinate system with orthometric heights, and a residual for every camera.'],
      ['Decision', 'Which cameras agree with the GPS; outliers are rejected by RANSAC, not averaged in.']],
    stats: (d) => { const g = d.georeferencing || {};
      return [stat('Horizontal residual', g.horizontal_rms_m == null ? '—' : `${num(g.horizontal_rms_m, 2)} m`, 'RMS of camera centres against the GPS track'),
        stat('Vertical residual', g.vertical_rms_m == null ? '—' : `${num(g.vertical_rms_m, 2)} m`, 'RMS of camera heights against the GPS track'),
        stat('Cameras fitted', g.cameras == null ? '—' : `${num(g.inliers, 0)} of ${num(g.cameras, 0)}`, 'inliers of the similarity fit'),
        stat('Coordinate system', g.crs || '—', 'UTM zone of the flight, with EGM96 heights')]; },
    visual: (d) => residuals(d),
    notTell: 'Georeferencing',
  },
  export: {
    chips: ['WKT compound CRS', 'ASPRS classification', 'LAS tiling', 'texel-resolution orthophoto', 'glTF custom attributes (_CONFIDENCE, _ZONE)', 'FBX via headless Blender', 'read-back verification'],
    does: [['Input', 'The georeferenced mesh, the point cloud and the zone labels.'],
      ['Operation', 'Writes OBJ, PLY, LAS, GeoTIFF, GLB and FBX, classifies the point cloud, renders the height model and orthophoto, then re-opens every file.'],
      ['Output', 'Six formats, with confidence and zone layers and the gap outlines travelling alongside.'],
      ['Decision', 'Whether each file is valid: one that cannot be read back is reported, not assumed.']],
    stats: (d) => [stat('Formats verified', d.formats?.produced ? `${d.formats.produced.length} of 6` : '—', 'written, then re-opened and checked'),
      stat('Points exported', big(d.formats?.assets?.points?.points_full), 'in the point-cloud export'),
      stat('Triangles exported', big(d.viewer?.faces), 'in the mesh export'),
      S(d, 'Coverage', 'coverage_pct')],
    visual: () => '',
    notTell: 'Formats (re-opened)',
  },
};

// ── assembly ────────────────────────────────────────────────────────────────────
function procCard(p) {
  const det = el('details', 'proc'), cls = scoreColour(p.score);
  det.innerHTML = `<summary><div class="ring tag ${cls}" style="border-radius:50%">${p.score == null ? '—' : Math.round(p.score)}</div>` +
    `<div><div class="st">${esc(p.stage)}</div><h4>${esc(p.process)}</h4></div>` +
    `<div style="display:flex;gap:4px">${['fail', 'warn', 'info', 'pass'].filter((s) => p.counts?.[s]).map((s) => `<span class="tag ${s}">${p.counts[s]}</span>`).join('')}</div></summary>` +
    `<div class="inner"><p class="blurb">${esc(p.what)}</p><p class="blurb"><b>The score accounts for:</b> ${esc(p.covers)}</p><p class="blurb"><b>It does not measure:</b> ${esc(p.lacks)}</p>` +
    `<div class="tablewrap">${(p.kpis || []).map(kpiRow).join('')}</div></div>`;
  return det;
}

function section(d, cfg) {
  const intro = `<div class="sec-h">Techniques</div>${chips(cfg.chips)}<div class="sec-h">What it does</div>${does(cfg.does)}`;
  const measured = `<div class="sec-h">Measured on this run</div><div class="statgrid">${cfg.stats(d).join('')}</div>` +
    (cfg.visual(d) ? `<div class="viz-wrap">${cfg.visual(d)}</div>` : '');
  const lacks = proc(d, cfg.notTell)?.lacks;
  const tail = `<div class="sec-h">Every measured check, by process</div><div class="proclist"></div>` +
    (lacks ? `<div class="callout plain notell"><b>What this stage does not tell you.</b> ${esc(lacks)}</div>` : '');
  return { intro, measured, tail };
}

// Fills every pipeline page from one run. Re-run on a flight switch: previous content is replaced, never stacked.
export function buildStagePages(d) {
  document.querySelectorAll('[data-gen]').forEach((n) => n.remove());
  const claimed = new Set();
  const byPage = {}; (d.processes || []).forEach((p) => { (byPage[pageFor(p)] ||= []).push(p); claimed.add(p); });
  Object.entries(PAGES).forEach(([id, cfg]) => {
    const sec = document.querySelector(`.step[data-step="${id}"]`); if (!sec) return;
    const { intro, measured, tail } = section(d, cfg);
    const list = (byPage[id] || []);
    const make = (html) => { const w = el('div', 'stage-gen', html); w.dataset.gen = id; return w; };
    if (id === 'export') {                                       // the viewer comes first: it is what this page is for; everything else follows it
      const after = make(intro + measured + tail); sec.querySelector('#fmtbelow').after(after);
      list.forEach((p) => after.querySelector('.proclist').appendChild(procCard(p)));
    } else {
      const host = $(`body-${id}`); host.innerHTML = '';
      const w = make(intro + measured + tail); host.appendChild(w);
      list.forEach((p) => w.querySelector('.proclist').appendChild(procCard(p)));
    }
  });
  return { processes: claimed.size };
}
