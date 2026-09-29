/* Landing page sections that come from data: the problem-statement targets against what the runs measured,
   and the two prepared flights side by side. Every number is read from a run's data.json (or runs/index.json)
   and printed as it is; a missing run or field shows "—", never a guess or an error.

   The only constants here are the problem statement's own targets (PS-17 §1.4), which are the yardstick,
   not a measurement. */
import { $, esc, num, clock, stat, scoreColour, RUN_BLURB, RUN_LABEL } from './ui.js';

const PS = { accuracy_m: 1, formats: 6 };                 // PS-17 §1.4: spatial accuracy <= 1 m; OBJ, PLY, LAS, GeoTIFF, GLB, FBX
const RANK = { fail: 3, warn: 2, pass: 1, info: 0 };
const WORD = { pass: 'Met', warn: 'Partial', fail: 'Not met', info: '—' };
const n = (v) => (v == null ? '—' : Number(v).toLocaleString());

// One line per prepared flight, each prefixed with its short label. `fn(d)` returns { text, status } or null.
function perRun(runs, all, fn) {
  const lines = [], statuses = [];
  runs.forEach((r) => {
    const d = all[r.key], label = esc(RUN_LABEL[r.key] || r.key);
    const out = d ? fn(d, r) : null;
    lines.push(`<div class="pf"><span class="pfk">${label}</span> ${out ? out.text : '—'}</div>`);
    if (out) statuses.push(out.status);
  });
  const status = statuses.length ? statuses.reduce((a, b) => (RANK[b] > RANK[a] ? b : a)) : 'info';
  return { html: lines.join(''), status };
}

function targetRows(runs, all) {
  const lidar = (d) => d.accuracy?.local?.points_after_tile_offset?.zone1_measured?.rms_m;
  const rows = [
    ['Reconstruction type', '3D mesh or point cloud',
      perRun(runs, all, (d) => {
        const faces = d.zones?.mesh?.faces, pts = d.formats?.assets?.points?.points_full;
        return { text: `${n(faces)} triangles · ${n(pts)} points`, status: faces > 0 && pts > 0 ? 'pass' : 'fail' };
      }),
      () => 'A textured mesh and a classified point cloud are both written.'],

    ['Processing time', 'Under 15 min for a 10-minute video',
      perRun(runs, all, (d) => {
        const p = d.projection || {};
        return { text: `${clock(d.time?.total_s)} for ${clock(d.time?.video_s)} of video → <b>${clock(p.projected_total_s)}</b> forecast for 10 min`,
          status: p.projected_total_s != null && p.budget_s != null && p.projected_total_s <= p.budget_s ? 'pass' : 'fail' };
      }),
      () => `Measured on a 3-core slice of one GPU, with frame selection decoding every frame and the stages running one after another.
        Selecting from the video's own keyframes, overlapping the stages and running on a full GPU are the route to the target;
        none of them changes what is produced.`],

    ['Spatial accuracy', 'Within 1 m',
      perRun(runs, all, (d) => {
        const rms = d.georeferencing?.rms_inliers_m, shape = lidar(d);
        return { text: `cameras vs GPS <b>${num(rms, 2)} m</b>${shape != null ? ` · shape vs lidar, Zone 1 RMS <b>${num(shape, 2)} m</b>` : ''}`,
          status: rms != null && rms <= PS.accuracy_m ? 'pass' : 'fail' };
      }),
      () => `Placement comes from the drone's own GPS and no ground control points were used, so it cannot beat the GPS.
        Shape is measured against independent lidar after removing each tile's placement offset (Esri only: no survey covers the other site).
        RTK/PPK corrections or a few surveyed points, both accepted by the pipeline, are the route to 1 m.`],

    ['Coverage', 'The entire visible scene',
      perRun(runs, all, (d) => ({ text: `<b>${num(d.coverage?.coverage_pct, 1)}%</b> of the ground the cameras saw`,
        status: d.coverage?.coverage_pct != null && d.coverage.coverage_pct >= 100 ? 'pass' : 'warn' })),
      (runsList, allData) => {
        const g = runsList.map((r) => { const z = allData[r.key]?.zones?.gaps; return z ? `${esc(RUN_LABEL[r.key] || r.key)} ${n(z.count)} gaps, ${n(z.total_m2)} m²` : null; }).filter(Boolean);
        return `Ground the cameras never observed is outlined and reported rather than invented${g.length ? ': ' + g.join('; ') : ''}.`;
      }],

    ['Output formats', 'OBJ, PLY, LAS, GeoTIFF, GLB, FBX',
      perRun(runs, all, (d) => {
        const k = d.formats?.produced?.length;
        return { text: `<b>${n(k)} of ${PS.formats}</b> written and re-opened`, status: k === PS.formats ? 'pass' : 'warn' };
      }),
      () => 'Each format is re-opened after writing and checked for its geometry, its coordinates and its extra layers.'],

    ['Visualization', 'Web-based or desktop viewer',
      perRun(runs, all, (d) => ({ text: 'web viewer, all six formats', status: d.viewer && Object.keys(d.viewer).length ? 'pass' : 'warn' })),
      () => 'The Outputs page loads every format through its own reader; the pipeline also writes a standalone viewer with each run.'],
  ];
  return rows.map(([target, required, measured, why]) => ({ target, required, measured, why: why(runs, all), status: measured.status }));
}

export function buildTargets(runs, all) {
  const box = $('land-targets');
  if (!box) return;
  const body = targetRows(runs, all).map((r) => `<tr>
      <td data-label="Target"><b>${esc(r.target)}</b></td>
      <td data-label="Required">${esc(r.required)}</td>
      <td data-label="Measured">${r.measured.html}</td>
      <td data-label="Status"><span class="tag ${r.status}">${WORD[r.status]}</span></td>
      <td data-label="Why" class="why">${r.why}</td></tr>`).join('');
  box.innerHTML = `<div class="tablewrap"><table class="data targets"><thead><tr>
      <th>Target</th><th>Required</th><th>Measured</th><th>Status</th><th>Why</th></tr></thead><tbody>${body}</tbody></table></div>`;
}

const SCORES = [['ingest', 'Ingest'], ['condition', 'Conditioning'], ['occlusion', 'Occlusion'], ['recon', 'Reconstruction'], ['geo_export', 'Geo / export']];

export function buildFlights(runs, all, open) {
  const box = $('land-flights');
  if (!box) return;
  const cards = runs.map((r) => {
    const d = all[r.key], [kind, blurb] = RUN_BLURB[r.key] || ['Flight', ''];
    if (!d) return `<article class="flight panel pad"><h3>${esc(r.video || r.key)}</h3><p class="hint">This flight's data did not load.</p></article>`;
    const sc = d.scorecards || {};
    const scores = SCORES.filter(([k]) => sc[k] && sc[k].score != null)
      .map(([k, label]) => `<span class="tag ${scoreColour(sc[k].score)}">${label} ${Math.round(sc[k].score)}</span>`).join('');
    return `<article class="flight panel">
      <div class="fthumb"><img src="runs/${esc(r.key)}/assets/ortho.png" alt="" loading="lazy"><span class="badge tag accent">${esc(kind)}</span></div>
      <div class="fbody">
        <h3>${esc(d.video)}</h3><p class="hint">${esc(blurb)}</p>
        <div class="statgrid">
          ${stat('Length', clock(d.time?.video_s))}${stat('Processed in', clock(d.time?.total_s))}
          ${stat('Cameras placed', n(d.georeferencing?.cameras))}${stat('Coverage', `${num(d.coverage?.coverage_pct, 1)}%`)}
          ${stat('Cameras vs GPS', `${num(d.georeferencing?.rms_inliers_m, 2)} m`)}
        </div>
        <div class="scorerow"><span class="cap">Stage scores</span>${scores || '<span class="hint">—</span>'}</div>
        <div class="actions"><button class="btn primary" data-open="${esc(r.key)}">Open this flight →</button></div>
      </div></article>`;
  });
  // One line on why the low flight costs more time, from the two runs' own stage timings; omitted if either is missing.
  const lo = all.dji, hi = all.esri;
  const why = lo && hi && lo.time?.stage_seconds?.track_a != null && hi.time?.stage_seconds?.track_a != null
    ? `<p class="flight-why"><b>Why the lower flight takes longer.</b> ${esc(lo.video)} sees far more detail in every frame, so dense matching and texturing dominate:
       reconstruction alone took ${clock(lo.time.stage_seconds.track_a)} of ${clock(lo.time.total_s)}, against ${clock(hi.time.stage_seconds.track_a)} of
       ${clock(hi.time.total_s)} for ${esc(hi.video)}.</p>` : '';
  box.innerHTML = `<div class="flights">${cards.join('')}</div>${why}`;
  box.querySelectorAll('[data-open]').forEach((b) => { b.onclick = () => open(b.dataset.open); });
}

// Fetch every prepared flight once. A run that fails to load is null, and its cells render "—".
export async function loadAll(runs) {
  const all = {};
  await Promise.all(runs.map(async (r) => {
    try { all[r.key] = await (await fetch(`runs/${r.key}/data.json`)).json(); } catch { all[r.key] = null; }
  }));
  return all;
}
