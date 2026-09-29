/* Charts and infographics for the Metrics panel: inline SVG and CSS, no library.
   The form follows the data's job: a ratio is a meter, a continuous series is one line, signed offsets are diverging bars,
   magnitudes are bars, a part-to-whole is a stacked bar. One axis each; thin marks with 4 px rounded data-ends; a 2 px surface
   gap between touching fills; labels only where they are sparing; every chart has a table view and a hover tooltip.
   Colour: one accent blue for the subject, a de-emphasis gray for context, a validated blue/orange pair for polarity.
   Everything plotted is read from data.json (and residuals.json); a chart with no data says "Not recorded for this flight." */
import { esc, num, clock, val, mb } from './ui.js';

const ZONE = { 1: '#29a645', 2: '#e8a521', 3: '#d93838' };            // reserved: the same zone colours as the viewers and legends
// Colours are CSS variables (styles.css), so an open dialog re-themes live. Validated pairs: light #1565C0 / #C2410C, dark #3D8BE0 / #D9692A.
const POS = 'var(--c-pos)', NEG = 'var(--c-neg)';                    // diverging poles
const GRAY = 'var(--c-gray)';                                         // de-emphasis
const W = 640;

const kpi = (d, proc, key) => (d.processes || []).find((p) => p.process === proc)?.kpis.find((k) => k.key === key);
const niceStep = (raw) => { const p = 10 ** Math.floor(Math.log10(raw || 1)), n = raw / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p; };
const niceMax = (v) => { const p = 10 ** Math.floor(Math.log10(v || 1)), n = v / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p; };
// Chart labels keep their decimals (4.97 is "5.0", not "5"): a fixed count reads as a measurement, a trimmed one as a rounding.
const fmt = (v, d = 1) => (v == null || Number.isNaN(+v) ? '—' : Number(v).toFixed(d));
const MB = mb;

const card = (title, sub, body, table, legend = '') =>
  `<figure class="vcard"><figcaption><h5>${esc(title)}</h5>${sub ? `<p>${esc(sub)}</p>` : ''}${legend}</figcaption>${body}` +
  `${table ? `<details class="vt"><summary>Table view</summary>${table}</details>` : ''}</figure>`;
const empty = (title) => card(title, '', '<p class="vnone">Not recorded for this flight.</p>');
const miniTable = (heads, rows) => `<div class="mtable"><table class="metrics"><thead><tr>${heads.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead>` +
  `<tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const legend = (items) => `<div class="vlegend">${items.map(([c, t]) => `<span><i style="background:${c}"></i>${esc(t)}</span>`).join('')}</div>`;

// ── meters: a ratio against its target, fill coloured by the check's own status ─────────
function meters(d, items) {
  const rows = items.map(([proc, key, mode]) => {
    const k = kpi(d, proc, key); if (!k) return '';
    const raw = mode === 'pct' ? (k.value == null ? null : k.value * 100) : (k.unit === '%' ? k.value : null);
    const shown = mode === 'pct' ? (k.value == null ? '—' : `${fmt(k.value * 100, 1)}%`) : `${val(k.value)}${k.unit ? ' ' + k.unit : ''}`;
    return `<div class="meter"><div class="lbl">${esc(k.label)}</div><div class="track"><i class="fill ${esc(k.status)}" style="width:${raw == null ? 0 : Math.max(0, Math.min(100, raw))}%"></i></div>` +
      `<div class="val">${esc(shown)}</div><div class="tgt">target ${esc(k.target || '—')}</div></div>`;
  }).join('');
  return rows;
}

// ── a part-to-whole: stacked bar with a 2 px surface gap and a legend ───────────────────
function stack(parts, title, sub) {          // parts: [[label, value, colour]]
  const live = parts.filter(([, v]) => v > 0);
  const bar = live.map(([l, v, c]) => `<i style="flex:${v} 1 0;background:${c}" data-tip="${esc(`${l}: ${num(v, 0)}`)}"></i>`).join('');
  const lg = `<div class="zlegend">${parts.map(([l, v, c]) => `<div class="row"><span class="sw" style="background:${c}"></span>${esc(l)} <b>${num(v, 1)}</b></div>`).join('')}</div>`;
  return card(title, sub, `<div class="zonebar">${bar}</div>${lg}`, miniTable(['Part', 'Value'], parts.map(([l, v]) => [l, num(v, 1)])));
}

// ── time per stage ────────────────────────────────────────────────────────────────────
function timeBars(d, hi) {
  const s = d.time?.stage_seconds || {}, names = { preflight: 'Input check', ingest: 'Ingest', condition: 'Conditioning', track_a: 'Reconstruction', fusion: 'Occlusion handling', geo: 'Georeferencing', export: 'Outputs' };
  const peak = Math.max(...Object.values(s), 1);
  const rows = Object.entries(s).map(([k, v]) => `<div class="tbar${k === hi ? ' cur' : ''}" data-tip="${esc(`${names[k] || k}: ${clock(v)}`)}"><div class="lbl">${esc(names[k] || k)}</div><div class="track"><i style="width:${(v / peak) * 100}%"></i></div><div class="val">${clock(v)}</div></div>`).join('');
  return card('Where the time went', 'Measured seconds per stage on this flight; bars are relative to the longest.', rows,
    miniTable(['Stage', 'Time'], Object.entries(s).map(([k, v]) => [names[k] || k, clock(v)])));
}

// ── horizontal bars: magnitudes, one series, value at the tip ─────────────────────────
function hBars(title, sub, items, unit, tableHeads) {     // items: [{label, v, text, tip, cells}]
  const peak = Math.max(...items.map((i) => i.v), 1e-9);
  const rows = items.map((i) => `<div class="hb" data-tip="${esc(i.tip || `${i.label}: ${i.text}`)}"><div class="lbl">${esc(i.label)}</div><div class="track"><i style="width:${Math.max((i.v / peak) * 100, 1.5)}%"></i></div><div class="val">${esc(i.text)}</div></div>`).join('');
  return card(title, sub, rows, miniTable(tableHeads, items.map((i) => i.cells || [i.label, i.text])));
}

// ── emphasis: the run with the cleanup on (accent) against off (gray) per kind of damage ─
function bench(d) {
  const b = d.benchmarks?.degradation; if (!b?.rows?.length) return empty('Cleanup on and off, on damaged footage');
  const NAME = { motion_blur: 'Motion blur', compression: 'Heavy compression', low_light: 'Low light', shadow: 'Hard shadows', gps_noise: 'Noisy GPS', dynamic_objects: 'Moving vehicles' };
  const by = {}; b.rows.forEach((r) => { if (r.kind === 'clean') return; (by[r.kind] ||= {})[r.conditioning ? 'on' : 'off'] = r; });
  const kinds = Object.keys(NAME).filter((k) => by[k]);
  const dev = (r) => r?.camera_deviation?.rms_m;
  const peak = Math.max(...kinds.flatMap((k) => [dev(by[k].on), dev(by[k].off)]).filter((v) => v != null), 1);
  const bar = (r, c, lab) => { const v = dev(r); return `<div class="gb"><div class="track"><i style="width:${v == null ? 0 : Math.max((v / peak) * 100, 1.2)}%;background:${c}"></i></div><div class="val">${v == null ? 'n/a' : `${fmt(v, 1)} m`}</div></div>`; };
  const rows = kinds.map((k) => `<div class="gpair" data-tip="${esc(`${NAME[k]}. Cleanup on: ${fmt(dev(by[k].on), 1)} m, ${by[k].on?.registered ?? '—'} frames placed. Cleanup off: ${fmt(dev(by[k].off), 1)} m, ${by[k].off?.registered ?? '—'} frames placed.`)}">` +
    `<div class="lbl">${esc(NAME[k])}</div><div class="gbars">${bar(by[k].on, 'var(--c-on)')}${bar(by[k].off, GRAY)}</div></div>`).join('');
  const base = b.rows.find((r) => r.kind === 'clean');
  return card('Cleanup on and off, on damaged footage', `Same flight re-run with damage injected (${(b.severities || []).join(', ')}). Bars show how far the camera path moved from the undamaged run; shorter is better.${base ? ` Undamaged baseline: ${base.registered} frames placed.` : ''}`,
    rows, miniTable(['Damage', 'Cleanup on (m)', 'Cleanup off (m)', 'Frames placed on / off'], kinds.map((k) => [NAME[k], fmt(dev(by[k].on), 1), fmt(dev(by[k].off), 1), `${by[k].on?.registered ?? '—'} / ${by[k].off?.registered ?? '—'}`])),
    legend([['var(--c-on)', 'Cleanup on'], [GRAY, 'Cleanup off']]));
}

// ── one continuous series as a line: camera residual along the flight ──────────────────
function residualLine(res, target) {
  const f = res?.frames; if (!f?.length) return empty('Camera position against the GPS track, along the flight');
  const H = 230, L = 46, R = 56, T = 16, B = 30, iw = W - L - R, ih = H - T - B;
  const ys = f.map((r) => r[1]), top = Math.max(...ys, target) * 1.05, step = niceStep(top / 4), ymax = Math.ceil(top / step) * step;
  const x = (i) => L + (f.length === 1 ? 0 : (i / (f.length - 1)) * iw), y = (v) => T + ih - (v / ymax) * ih;
  const line = f.map((r, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(r[1]).toFixed(1)}`).join(' ');
  const area = `${line} L${x(f.length - 1).toFixed(1)} ${y(0)} L${x(0).toFixed(1)} ${y(0)} Z`;
  const ticks = Array.from({ length: Math.round(ymax / step) + 1 }, (_, k) => k * step).map((v) => `<line class="gl" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="ax" x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${fmt(v, step < 1 ? 1 : 0)}</text>`).join('');
  const last = f[f.length - 1];
  const pts = f.map((r, i) => ({ x: x(i), y: y(r[1]), tip: `${r[0].replace(/\.jpg$/, '')} · ${fmt(r[1], 2)} m${r[2] ? '' : ' · not used in the fit'}` }));
  const svg = `<svg class="vsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Camera residual against the GPS track along the flight, from ${fmt(ys[0], 1)} to ${fmt(last[1], 1)} metres" data-line='${JSON.stringify(pts).replace(/'/g, '&#39;')}'>
    ${ticks}<line class="gl ref" x1="${L}" x2="${W - R}" y1="${y(target)}" y2="${y(target)}"/><text class="ax ref" x="${W - R + 6}" y="${y(target) + 4}">${target} m target</text>
    <path d="${area}" fill="var(--c-area)" fill-opacity=".10"/><path d="${line}" fill="none" stroke="var(--c-line)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${x(f.length - 1)}" cy="${y(last[1])}" r="5" fill="var(--c-line)" stroke="var(--panel)" stroke-width="2"/><text class="ax end" x="${x(f.length - 1) + 10}" y="${y(last[1]) - 8}">${fmt(last[1], 1)} m</text>
    <text class="ax" x="${L}" y="${H - 6}">first frame</text><text class="ax" x="${W - R}" y="${H - 6}" text-anchor="end">last frame</text>
    <line class="xh" x1="0" x2="0" y1="${T}" y2="${T + ih}" hidden/><circle class="xd" r="5" fill="var(--c-line)" stroke="var(--panel)" stroke-width="2" hidden/><rect class="hit" x="${L}" y="${T}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;
  return card('Camera position against the GPS track, along the flight', `How far each camera sits from the GPS track after the fit, in flight order (${res.count} cameras). Where the line is high, the reconstructed path and the GPS disagree most.`,
    svg, miniTable(['Frame', 'Residual (m)', 'Used in the fit'], f.map((r) => [r[0].replace(/\.jpg$/, ''), fmt(r[1], 2), r[2] ? 'yes' : 'no'])));
}

// ── signed offsets: vertical offset of each 100 m tile against the lidar, diverging bars ─
function tileOffsets(d) {
  const t = d.accuracy?.local?.tile_offsets; if (!t?.length) return empty('Vertical offset of each tile against the lidar survey');
  const H = 230, L = 46, R = 16, T = 18, B = 32, iw = W - L - R, ih = H - T - B, mid = T + ih / 2;
  const top = Math.max(...t.map((o) => Math.abs(o.up_m)), 0.5) * 1.05, tstep = niceStep(top / 2), peak = Math.ceil(top / tstep) * tstep;
  const y = (v) => mid - (v / peak) * (ih / 2), slot = iw / t.length, w = Math.min(24, slot - 2);
  const path = (cx, v) => { const y1 = y(v), h = Math.abs(y1 - mid), r = Math.min(4, h / 2), x0 = cx - w / 2, x1 = cx + w / 2;
    return v >= 0 ? `M${x0} ${mid} V${y1 + r} Q${x0} ${y1} ${x0 + r} ${y1} H${x1 - r} Q${x1} ${y1} ${x1} ${y1 + r} V${mid} Z`
                  : `M${x0} ${mid} V${y1 - r} Q${x0} ${y1} ${x0 + r} ${y1} H${x1 - r} Q${x1} ${y1} ${x1} ${y1 - r} V${mid} Z`; };
  const ext = t.reduce((a, o) => (Math.abs(o.up_m) > Math.abs(a.up_m) ? o : a), t[0]);
  const bars = t.map((o, i) => { const cx = L + slot * (i + 0.5);
    return `<path d="${path(cx, o.up_m)}" fill="${o.up_m >= 0 ? POS : NEG}" data-tip="${esc(`Tile ${i + 1}: ${fmt(o.up_m, 2)} m up · ${fmt(o.east_m, 2)} m east · ${fmt(o.north_m, 2)} m north`)}"/><text class="ax" x="${cx}" y="${H - 10}" text-anchor="middle">${i + 1}</text>`; }).join('');
  const ticks = Array.from({ length: Math.round((2 * peak) / tstep) + 1 }, (_, k) => -peak + k * tstep).map((v) => `<line class="gl" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="ax" x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${fmt(v, tstep < 1 ? 1 : 0)}</text>`).join('');
  const ei = t.indexOf(ext), ecx = L + slot * (ei + 0.5);
  const svg = `<svg class="vsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Vertical offset of each 100 metre tile against the lidar survey, from ${fmt(Math.min(...t.map((o) => o.up_m)), 1)} to ${fmt(Math.max(...t.map((o) => o.up_m)), 1)} metres">${ticks}<line class="zero" x1="${L}" x2="${W - R}" y1="${mid}" y2="${mid}"/>${bars}<text class="ax end" x="${ecx}" y="${ext.up_m >= 0 ? y(ext.up_m) - 6 : y(ext.up_m) + 15}" text-anchor="middle">${fmt(ext.up_m, 1)} m</text></svg>`;
  const dm = d.accuracy?.local?.doming;
  return card('Vertical offset of each tile against the lidar survey', `Each ${num(d.accuracy.local.tile_m, 0)} m tile's height against the reference, in order along the flight. ${dm ? `The two ends sit about ${fmt(Math.abs(dm.sag_m), 1)} m ${dm.sag_m < 0 ? 'below' : 'above'} the middle, a tilt of ${fmt(dm.tilt_m_per_km, 0)} m per kilometre.` : ''}`,
    svg, miniTable(['Tile', 'Up (m)', 'East (m)', 'North (m)'], t.map((o, i) => [String(i + 1), fmt(o.up_m, 2), fmt(o.east_m, 2), fmt(o.north_m, 2)])), legend([[POS, 'Above the lidar'], [NEG, 'Below the lidar']]));
}

function zoneAccuracy(d) {
  const p = d.accuracy?.local?.points_after_tile_offset; if (!p) return empty('Shape accuracy by how well each surface was observed');
  const N = [['zone1_measured', 'Zone 1 · measured'], ['zone2_measured', 'Zone 2 · fewer views'], ['zone2_fill', 'Zone 2 · filled from depth'], ['all_points', 'All points']].filter(([k]) => p[k]);
  return hBars('Shape accuracy by how well each surface was observed', `Root-mean-square height difference against ${d.accuracy?.reference?.source || 'an independent survey'}, after removing each tile's placement offset. Shorter is better.`,
    N.map(([k, l]) => ({ label: l, v: p[k].rms_m, text: `${fmt(p[k].rms_m, 2)} m`, tip: `${l}: RMS ${fmt(p[k].rms_m, 2)} m · NMAD ${fmt(p[k].nmad_m, 2)} m · ${num(p[k].n, 0)} points`, cells: [l, fmt(p[k].rms_m, 2), fmt(p[k].nmad_m, 2), num(p[k].n, 0)] })),
    'm', ['Surface', 'RMS (m)', 'NMAD (m)', 'Points']);
}

function fileSizes(d) {
  const f = d.formats?.files || {}, set = [['model.glb', 'GLB'], ['model.obj', 'OBJ'], ['model.fbx', 'FBX'], ['cloud.ply', 'PLY'], ['cloud.las', 'LAS'], ['orthophoto.tif', 'GeoTIFF orthophoto'], ['dsm.tif', 'GeoTIFF height model']].filter(([k]) => f[k] != null);
  if (!set.length) return empty('Size of each file written');
  return hBars('Size of each file written', 'What the pipeline wrote for this flight, largest first.',
    set.map(([k, l]) => ({ label: l, v: f[k], text: MB(f[k]), cells: [l, MB(f[k])] })).sort((a, b) => b.v - a.v), 'MB', ['File', 'Size']);
}

// ── forecast: measured (gray) against the 10-minute forecast (accent), the stage's allowance ticked ──
export function forecastChart(d, rows) {                  // rows: [{ label, measured, forecast, budget }]
  const peak = Math.max(...rows.flatMap((r) => [r.measured, r.forecast, r.budget || 0]), 1);
  const bar = (v, c) => `<div class="gb"><div class="track"><i style="width:${Math.max((v / peak) * 100, 0.8)}%;background:${c}"></i></div><div class="val">${clock(v)}</div></div>`;
  const body = rows.map((r) => `<div class="gpair" data-tip="${esc(`${r.label}. Measured ${clock(r.measured)}, forecast ${clock(r.forecast)}${r.budget ? `, allowance ${clock(r.budget)}` : ''}.`)}">` +
    `<div class="lbl">${esc(r.label)}</div><div class="gbars"><div class="gbw">${bar(r.forecast, 'var(--c-on)')}${r.budget ? `<b class="bmark" style="left:${(r.budget / peak) * 100}%" title="allowance ${esc(clock(r.budget))}"></b>` : ''}</div>${bar(r.measured, GRAY)}</div></div>`).join('');
  return card('Measured against the forecast, by stage', 'Bars share one time axis. The tick on each forecast bar is the time that stage is allowed in a 15-minute run.', body,
    miniTable(['Stage', 'Measured', 'Forecast (10 min)', 'Allowance'], rows.map((r) => [r.label, clock(r.measured), clock(r.forecast), r.budget ? clock(r.budget) : '—'])),
    legend([['var(--c-on)', 'Forecast for a 10-minute video'], [GRAY, 'Measured on this flight']]));
}

// ── assembly ───────────────────────────────────────────────────────────────────────────
export function stageViz(d, id, extra = {}) {
  const cards = [];
  if (id === 'check') {
    const c = { pass: 0, warn: 0, fail: 0, info: 0 }; (d.input_check?.checks || []).forEach((x) => { c[x.status] = (c[x.status] || 0) + 1; });
    cards.push(stack([['Met', c.pass, 'var(--ok)'], ['Near target', c.warn, 'var(--warn)'], ['Below target', c.fail, 'var(--bad)'], ['Recorded', c.info, GRAY]], 'Input checks by result', 'Every check the file went through before any compute was spent.'));
    const s = d.input_check?.sync;
    if (s?.r_best != null) cards.push(card('Video and GPS agreement', 'How well the motion seen in the image tracks the ground speed the GPS reports.', `<div class="meter"><div class="lbl">Agreement</div><div class="track"><i class="fill pass" style="width:${Math.round(s.r_best * 100)}%"></i></div><div class="val">${Math.round(s.r_best * 100)}%</div><div class="tgt">read at a ${fmt(s.lag_s, 2)} s offset, applied automatically</div></div>`));
  } else if (id === 'ingest') {
    cards.push(card('Selection, blur and telemetry', 'Each bar is a measured share; its colour is the check\'s own status.', meters(d, [['Frame selection', 'overlap_median', 'pct'], ['Frame selection', 'decode_fraction', 'pct'], ['Blur gate', 'blur_reject_fraction', 'pct'], ['Telemetry', 'gps_frame_coverage', 'pct']])));
  } else if (id === 'condition') {
    cards.push(card('Artifacts, illumination and GPS', 'Each bar is a measured share; its colour is the check\'s own status.', meters(d, [['Artifact suppression', 'artifact_correct_fraction', 'pct'], ['Artifact suppression', 'blockiness_reduction', 'pct'], ['Illumination', 'shadow_fraction_mean', 'pct'], ['Illumination', 'saturated_fraction', 'pct'], ['Illumination', 'exposure_clamped_fraction', 'pct'], ['GPS conditioning', 'gps_outlier_fraction', 'pct']])));
    cards.push(bench(d));
  } else if (id === 'recon') {
    cards.push(timeBars(d, 'track_a'));
    cards.push(card('Solution quality', 'Each bar is a measured share; its colour is the check\'s own status.', meters(d, [['Sparse (SfM)', 'registered_fraction', 'pct'], ['Track B (VGGT depth)', 'frames_anchored', 'pct']])));
  } else if (id === 'fusion') {
    const z = d.zones || {};
    if (z.zone1_pct != null) cards.push(stack([['Zone 1 · measured', z.zone1_pct, ZONE[1]], ['Zone 2 · thin or anchored', z.zone2_pct, ZONE[2]], ['Zone 3 · never observed', z.zone3_pct, ZONE[3]]], 'How well the ground was observed',
      `Share of the ground the cameras saw. ${z.gaps?.count != null ? `${num(z.gaps.count, 0)} gap outlines covering ${num(z.gaps.total_m2, 0)} m² were written to gaps.geojson and never filled.` : ''}`));
    else cards.push(empty('How well the ground was observed'));
  } else if (id === 'geo') {
    cards.push(residualLine(extra.residuals, 1), tileOffsets(d), zoneAccuracy(d));
  } else if (id === 'export') {
    cards.push(fileSizes(d));
  }
  return `<div class="vgrid">${cards.join('')}</div>`;
}

// ── behaviour: one tooltip for every mark, and a crosshair on the line chart ───────────
export function wireViz(root) {
  let tip = document.getElementById('vtip');
  if (!tip) { tip = document.createElement('div'); tip.id = 'vtip'; tip.className = 'vtip'; tip.setAttribute('role', 'tooltip'); tip.hidden = true; document.body.appendChild(tip); }
  const place = (e, text) => { tip.textContent = text; tip.hidden = false; const w = tip.offsetWidth; tip.style.left = `${Math.min(e.clientX + 14, innerWidth - w - 8)}px`; tip.style.top = `${e.clientY + 16}px`; };
  root.querySelectorAll('[data-tip]').forEach((n) => { n.addEventListener('pointermove', (e) => place(e, n.dataset.tip)); n.addEventListener('pointerleave', () => { tip.hidden = true; }); });
  root.querySelectorAll('svg[data-line]').forEach((svg) => {
    const pts = JSON.parse(svg.dataset.line), xh = svg.querySelector('.xh'), xd = svg.querySelector('.xd'), hit = svg.querySelector('.hit');
    hit.addEventListener('pointermove', (e) => {
      const r = svg.getBoundingClientRect(), sx = ((e.clientX - r.left) / r.width) * W;
      const p = pts.reduce((a, b) => (Math.abs(b.x - sx) < Math.abs(a.x - sx) ? b : a));
      xh.setAttribute('x1', p.x); xh.setAttribute('x2', p.x); xd.setAttribute('cx', p.x); xd.setAttribute('cy', p.y); xh.hidden = xd.hidden = false;
      place(e, p.tip);
    });
    hit.addEventListener('pointerleave', () => { xh.hidden = xd.hidden = true; tip.hidden = true; });
  });
}
