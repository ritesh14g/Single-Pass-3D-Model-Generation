/* Single-pass drone 3D — demo console.
   Four steps (input, input check, pipeline, outputs) plus a quality report, all driven by data.json
   written by scripts/build_site.py from finished pipeline runs. Nothing here recomputes anything:
   every number shown is read from the run it belongs to. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

// ── helpers ───────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const mb = (b) => b == null ? '—' : (b >= 1e6 ? (b / 1048576).toFixed(1) + ' MB' : (b / 1024).toFixed(0) + ' KB');
const num = (v, d = 2) => (v == null || v === '' || Number.isNaN(+v)) ? '—' : (typeof v === 'number' ? (+v).toFixed(Math.abs(v) >= 100 ? 0 : d).replace(/\.?0+$/, '') : String(v));
const clock = (s) => s == null ? '—' : (s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${s < 10 ? s.toFixed(1) : Math.round(s)} s`);
const val = (v) => typeof v === 'boolean' ? (v ? 'yes' : 'no') : (typeof v === 'number' ? num(v, 3) : (v == null ? '—' : String(v)));
const STATUS_WORD = { pass: 'Met', warn: 'Near target', fail: 'Below target', info: 'Recorded' };

// ── shared components (styles.css: .chips, .progress, .statgrid, .kpirow) ─────
// Return HTML strings or nodes only; nothing here reads or computes a measured value.
const chips = (list) => `<div class="chips">${list.map((c) => {
  const [text, tip] = Array.isArray(c) ? c : [c];
  return `<span class="chip"${tip ? ` title="${esc(tip)}"` : ''}>${esc(text)}</span>`;
}).join('')}</div>`;
const stat = (label, value, context) => `<div class="stat"><div class="k">${esc(label)}</div>` +
  `<div class="v">${esc(value)}</div>${context ? `<div class="c">${esc(context)}</div>` : ''}</div>`;
const kpiRow = (k) => `<div class="kpirow"><div><div class="lbl">${esc(k.label)}</div>` +
  `<div class="det">${esc(k.detail || '')}</div></div>` +
  `<div class="v">${esc(val(k.value))}${k.unit ? ' ' + esc(k.unit) : ''}<div class="tgt">target ${esc(k.target || '—')}</div></div>` +
  `<div><span class="tag ${esc(k.status)}">${STATUS_WORD[k.status] || ''}</span></div></div>`;
// steps: [{ id, title, accent? }]; each segment carries its own data-accent so it recolours itself.
function progressBar(steps, currentId, onPick) {
  const at = steps.findIndex((s) => s.id === currentId);
  const nav = el('nav'); nav.className = 'progress'; nav.setAttribute('aria-label', 'Pipeline progress');
  const ol = el('ol');
  steps.forEach((s, i) => {
    const li = el('li', i < at ? 'done' : (i === at ? 'current' : ''));
    li.dataset.accent = s.accent || s.id;
    const b = el('button', '', `<span class="seg"></span><span class="lbl"><i>${i + 1}</i>${esc(s.title)}</span>`);
    b.type = 'button';
    if (i === at) b.setAttribute('aria-current', 'step');
    b.onclick = () => onPick(s.id);
    li.appendChild(b); ol.appendChild(li);
  });
  nav.appendChild(ol);
  return nav;
}

const state = { runs: [], data: null, step: 0, format: 'glb', qaTab: 'processes', own: {} };

// ── boot ──────────────────────────────────────────────────────────────────────
// Anything that goes wrong is shown on the page rather than leaving a blank panel.
const errors = [];
function report(what, e) {
  errors.push(`${what}: ${e?.message || e}`);
  let box = $('errbox');
  if (!box) { box = el('div', 'callout'); box.id = 'errbox'; box.style.borderLeftColor = 'var(--bad)'; box.style.background = 'var(--bad-bg)'; document.querySelector('main').prepend(box); }
  box.innerHTML = `<b>Something did not load.</b> ${errors.map(esc).join('<br>')}`;
}
addEventListener('error', (e) => report('Script', e.error || e.message));
addEventListener('unhandledrejection', (e) => report('Load', e.reason));

init().catch((e) => report('Startup', e));

async function init() {
  const idx = await (await fetch('runs/index.json')).json();
  state.runs = idx.runs || [];
  buildRail();
  buildRunCards();
  wireDropzones();
  wireQA();
  $('foot').innerHTML = `Every figure on this page is read from a completed pipeline run: its input check, its
    quality report and the files it wrote. Processed on an NVIDIA H100 80&nbsp;GB MIG&nbsp;2g.20gb slice
    (19.6&nbsp;GB, 3 CPU cores). Reference surface for the accuracy comparison: USGS 3DEP lidar.
    <a href="https://github.com/ritesh14g/Single-Pass-3D-Model-Generation">Source code</a>.`;
  // Deep link for the demo and for screenshots: ?run=dji&step=4&format=las&qa=speed
  const q = new URLSearchParams(location.search);
  const key = q.get('run') && state.runs.some((r) => r.key === q.get('run')) ? q.get('run') : (state.runs[0]?.key || 'esri');
  await loadRun(key, false);
  const fmt = q.get('format');
  if (fmt && FORMATS.includes(fmt)) state.format = fmt;
  const step = Math.min(Math.max(parseInt(q.get('step') || '1', 10) - 1, 0), STEPS.length - 1);
  goto(step);
  if (q.get('qa')) {
    if (QA_TABS.some(([k]) => k === q.get('qa'))) state.qaTab = q.get('qa');
    buildQA(); $('qa').classList.add('on'); $('scrim').classList.add('on');
  }
}

// ── step rail ─────────────────────────────────────────────────────────────────
const STEPS = ['Input', 'Input check', 'Pipeline', 'Outputs'];
function buildRail() {
  const rail = $('rail');
  rail.innerHTML = '';
  STEPS.forEach((name, i) => {
    const b = el('button', i === state.step ? 'on' : (i < state.step ? 'done' : ''),
      `<i class="n">${i < state.step ? '✓' : i + 1}</i><span class="lbl">${name}</span>`);
    b.onclick = () => goto(i);
    rail.appendChild(b);
  });
}
function goto(i) {
  state.step = i;
  document.querySelectorAll('.step').forEach((s) => s.classList.toggle('on', +s.dataset.step === i));
  buildRail();
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (i === 2) runFlow();
  if (i === 3) showFormat(state.format);
}

// ── step 1: input ─────────────────────────────────────────────────────────────
function buildRunCards() {
  const wrap = $('runcards');
  wrap.innerHTML = '';
  const blurb = {
    esri: ['Surveillance aircraft', 'High, wide-angle pass over a river and woodland. Telemetry is MISB KLV carried inside the video stream itself — no separate file.'],
    dji: ['DJI quadcopter', 'Low, detailed pass along a 907 m straight line. Telemetry is a flight-log CSV recorded alongside the video.'],
  };
  state.runs.forEach((r) => {
    const [kind, text] = blurb[r.key] || ['Flight', ''];
    const card = el('button', 'runcard');
    card.innerHTML = `
      <div class="thumb">
        <img src="runs/${r.key}/assets/ortho.png" alt="" loading="lazy">
        <span class="badge tag accent">${esc(kind)}</span>
      </div>
      <div class="body">
        <h4>${esc(r.video)}</h4>
        <div class="sub">${esc(text)}</div>
        <div class="kvrow">
          <span>Length <b>${clock(r.video_s)}</b></span>
          <span>Processed in <b>${clock(r.total_s)}</b></span>
          <span>Coverage <b>${num(r.coverage_pct, 1)}%</b></span>
        </div>
      </div>`;
    card.onclick = () => loadRun(r.key, true);
    wrap.appendChild(card);
  });
}

function wireDropzones() {
  const mk = (zoneId, listId, kind) => {
    const zone = $(zoneId), list = $(listId);
    const input = el('input'); input.type = 'file'; input.hidden = true;
    input.accept = kind === 'video' ? 'video/*,.ts,.mpeg4,.h264' : '.srt,.csv,.txt,.gpx,.kml,.json,.geojson,.ulg,.bin,.tlog';
    zone.after(input);
    const accept = async (file) => {
      if (!file) return;
      $('own-note').hidden = false;
      list.innerHTML = '';
      const item = el('div', 'fileitem');
      item.innerHTML = `<span>📄</span><div><b>${esc(file.name)}</b><br><span class="k">${mb(file.size)} · reading…</span></div>`;
      list.appendChild(item);
      const detail = kind === 'video' ? await readVideo(file) : await readText(file);
      item.querySelector('.k').innerHTML = `${mb(file.size)} · ${detail}`;
    };
    zone.onclick = () => input.click();
    zone.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } };
    input.onchange = () => accept(input.files[0]);
    ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('hot'); }));
    ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('hot'); }));
    zone.addEventListener('drop', (e) => accept(e.dataTransfer.files[0]));
  };
  mk('dz-video', 'fl-video', 'video');
  mk('dz-tel', 'fl-tel', 'telemetry');
}

function readVideo(file) {
  return new Promise((resolve) => {
    const v = document.createElement('video');
    v.preload = 'metadata';
    v.onloadedmetadata = () => {
      const d = `${v.videoWidth}×${v.videoHeight}, ${clock(v.duration)} — the input check would read the codec,
        scan type, keyframe spacing and per-frame sharpness from this`;
      URL.revokeObjectURL(v.src); resolve(d);
    };
    v.onerror = () => resolve('this browser cannot decode the container; the pipeline reads it by content, not extension');
    v.src = URL.createObjectURL(file);
  });
}
async function readText(file) {
  const head = await file.slice(0, 4096).text().catch(() => '');
  const lines = head.split(/\r?\n/).filter(Boolean);
  let kind = 'unrecognised here';
  if (/-->/.test(head)) kind = 'looks like an SRT subtitle track (DJI writes telemetry this way)';
  else if (/^<\?xml|<gpx|<kml/i.test(head.trim())) kind = 'looks like GPX/KML';
  else if (/^\s*\{/.test(head)) kind = 'looks like JSON or GeoJSON';
  else if (lines[0] && lines[0].split(/[,;\t]/).length > 2) kind = `looks like a table, ${lines[0].split(/[,;\t]/).length} columns`;
  return `${lines.length}+ lines · ${kind}`;
}

// ── run loading ───────────────────────────────────────────────────────────────
async function loadRun(key, advance) {
  state.data = await (await fetch(`runs/${key}/data.json`)).json();
  buildChecks();
  buildFlow();
  buildFormatBar();
  buildQA();
  if (advance) goto(1);
}

// ── step 2: input check ───────────────────────────────────────────────────────
function buildChecks() {
  const d = state.data, ic = d.input_check, body = $('check-body');
  const counts = { pass: 0, warn: 0, fail: 0, info: 0 };
  (ic.checks || []).forEach((c) => { counts[c.status] = (counts[c.status] || 0) + 1; });
  const ok = ic.verdict !== 'BLOCKED';
  const t = ic.telemetry || {}, sync = ic.sync || {}, cam = ic.camera || {};

  body.innerHTML = '';
  const verdict = el('div', 'verdict');
  verdict.innerHTML = `
    <div class="ring" style="background:var(--${ok ? 'ok' : 'bad'}-bg);color:var(--${ok ? 'ok' : 'bad'})">${ok ? '✓' : '✕'}</div>
    <div style="flex:1;min-width:0">
      <h3 style="font-size:16px">${ok ? 'Ready to process' : 'Blocked'}${ic.verdict === 'READY_WITH_WARNINGS' ? ' — with points to note' : ''}</h3>
      <p style="margin:3px 0 0;color:var(--muted);font-size:13.5px">
        ${esc(d.video)} · ${counts.pass} checks met, ${counts.warn} near target, ${counts.info} recorded for information${counts.fail ? `, ${counts.fail} below target` : ''}
        · decided in ${clock(ic.timing_s?.total)}</p>
    </div>`;
  body.appendChild(verdict);

  const cards = el('div', 'grid cols-4'); cards.style.margin = '14px 0';
  const sourceLabel = (t.source || '').startsWith('klv') ? 'Embedded in the video (MISB KLV)' : (t.source || '—');
  [
    ['Position fixes read', `${t.rows ?? '—'}`, `over ${clock(t.duration_s)} · ${esc(sourceLabel)}`],
    ['Video ↔ GPS agreement', sync.r_best != null ? `${(sync.r_best * 100).toFixed(0)}%` : '—',
      `image motion correlated with ground speed at a ${num(sync.lag_s, 2)} s offset, applied automatically`],
    ['Camera', cam.model ? esc(cam.model) : `${num(cam.hfov_deg, 1)}° field of view`, esc(cam.source || '')],
    ['Optional inputs present', [t.has_gps && 'GPS', t.has_baro && 'barometer', t.has_attitude && 'attitude', t.has_focal && 'focal length', t.has_rtk && 'RTK'].filter(Boolean).join(', ') || '—',
      'each missing input has a declared fallback rather than a hard requirement'],
  ].forEach(([k, v, s]) => {
    const c = el('div', 'panel pad');
    c.innerHTML = `<div class="hint" style="margin-bottom:5px">${k}</div><div class="metric"><b style="font-size:19px">${v}</b></div><div class="hint" style="margin-top:5px">${s}</div>`;
    cards.appendChild(c);
  });
  body.appendChild(cards);

  const groups = [...new Set((ic.checks || []).map((c) => c.group))];
  const wrap = el('div', 'grid'); wrap.style.gap = '10px';
  groups.forEach((g, gi) => {
    const items = ic.checks.filter((c) => c.group === g);
    const gc = { pass: 0, warn: 0, fail: 0, info: 0 };
    items.forEach((c) => gc[c.status]++);
    const det = el('details', 'checkgroup');
    if (gi < 2 || gc.warn || gc.fail) det.open = true;
    const pills = ['fail', 'warn', 'info', 'pass'].filter((s) => gc[s]).map((s) => `<span class="tag ${s}">${gc[s]} ${STATUS_WORD[s].toLowerCase()}</span>`).join('');
    det.innerHTML = `<summary>${esc(g)} <span class="count">${pills}</span></summary>`;
    items.forEach((c) => {
      const row = el('div', 'check');
      row.innerHTML = `
        <div class="id">${esc(c.id)}</div>
        <div><div class="lbl">${esc(c.label)}</div><div class="det">${esc(c.detail)}</div>
          ${c.fix ? `<div class="fix"><b>If it matters:</b> ${esc(c.fix)}</div>` : ''}</div>
        <div class="val"><div>${esc(val(c.value))}</div><span class="tag ${c.status}" style="margin-top:4px">${STATUS_WORD[c.status]}</span></div>`;
      det.appendChild(row);
    });
    wrap.appendChild(det);
  });
  body.appendChild(wrap);

  const note = el('div', 'callout plain'); note.style.marginTop = '14px';
  note.innerHTML = esc(ic.note || '');
  body.appendChild(note);

  const act = el('div', 'actions'); act.style.marginTop = '16px';
  act.innerHTML = `<button class="btn primary">Run the pipeline →</button><button class="btn ghost">← Back to input</button>`;
  act.children[0].onclick = () => goto(2);
  act.children[1].onclick = () => goto(0);
  body.appendChild(act);
}

// ── step 3: pipeline ──────────────────────────────────────────────────────────
const STAGE_TEXT = {
  preflight: ['Input check', 'Reads the container, the position log and whether the two agree, before any time is spent.'],
  ingest: ['Ingest', 'Decodes the video and keeps the frames that overlap each other enough to reconstruct from, rejecting blurred ones.'],
  condition: ['Conditioning', 'Cleans each kept frame: compression blocking, exposure drift, shadows, moving objects, and noisy GPS.'],
  track_a: ['Reconstruction', 'Solves where the camera was for every frame, then builds the dense surface and paints the photographs onto it.'],
  fusion: ['Occlusion handling', 'Labels every surface by how well it was really seen, fills thin areas from anchored depth, and outlines what was never seen at all.'],
  geo: ['Georeferencing', 'Places the model on the map from the GPS track and resolves what the heights are measured from.'],
  export: ['Export', 'Writes all six formats, classifies the point cloud and renders the height model and orthophoto.'],
  qa: ['Quality report', 'Scores every process against its target, compares against a reference surface and records the timings.'],
};
function buildFlow() {
  const d = state.data, stages = d.time?.stage_seconds || {}, body = $('flow-body');
  body.innerHTML = '';
  const total = d.time?.total_s || Object.values(stages).reduce((a, b) => a + b, 0);

  const bar = el('div', 'panel pad'); bar.style.marginBottom = '14px';
  bar.innerHTML = `<div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center">
      <div class="metric"><b>${clock(total)}</b><span>total, measured</span></div>
      <div class="metric"><b>${clock(d.time?.video_s)}</b><span>of video</span></div>
      <div class="metric"><b>${d.viewer?.faces ? (d.viewer.faces / 1e6).toFixed(2) + ' M' : '—'}</b><span>surface triangles produced</span></div>
      <div style="margin-left:auto" class="actions">
        <button class="btn sm" id="replay">Replay</button>
        <button class="btn primary sm" id="toout">See the outputs →</button>
      </div>
    </div>`;
  body.appendChild(bar);
  $('replay') && ($('replay').onclick = () => runFlow(true));
  $('toout') && ($('toout').onclick = () => goto(3));

  const flow = el('div', 'flow'); flow.id = 'flow';
  Object.entries(stages).forEach(([key, secs], i) => {
    const [name, text] = STAGE_TEXT[key] || [key, ''];
    const n = el('div', 'node'); n.dataset.stage = key;
    n.innerHTML = `<div class="mark">${i + 1}</div>
      <div><h4>${esc(name)}</h4><p>${esc(text)}</p><div class="bar"><i></i></div></div>
      <div class="t">${clock(secs)}</div>`;
    flow.appendChild(n);
  });
  body.appendChild(flow);

  const after = el('div', 'callout'); after.style.marginTop = '14px';
  after.innerHTML = `The quality report scores each of these by the <b>process</b> inside it, not by the stage as a
    whole, and states what every score does and does not account for. Open it from the button at the top right.`;
  body.appendChild(after);
}
let flowTimer = null;
function runFlow(force) {
  const flow = $('flow'); if (!flow) return;
  const nodes = [...flow.children];
  if (!force && nodes.some((n) => n.classList.contains('done'))) return;
  clearTimeout(flowTimer);
  nodes.forEach((n) => { n.className = 'node'; n.querySelector('.bar > i').style.width = '0'; });
  const secs = Object.values(state.data.time?.stage_seconds || {});
  const total = secs.reduce((a, b) => a + b, 0) || 1;
  let i = 0;
  const next = () => {
    if (i >= nodes.length) return;
    const n = nodes[i], ms = Math.max(260, (secs[i] / total) * 5200);
    n.classList.add('active');
    const fill = n.querySelector('.bar > i');
    requestAnimationFrame(() => { fill.style.transition = `width ${ms}ms linear`; fill.style.width = '100%'; });
    flowTimer = setTimeout(() => { n.classList.remove('active'); n.classList.add('done'); n.querySelector('.mark').textContent = '✓'; i++; next(); }, ms);
  };
  next();
}

// ── step 4: outputs ───────────────────────────────────────────────────────────
const FORMATS = ['glb', 'obj', 'fbx', 'ply', 'las', 'geotiff'];
function buildFormatBar() {
  const bar = $('fmtbar'); bar.innerHTML = '';
  FORMATS.forEach((f) => {
    const info = state.data.formats.info[f] || [f.toUpperCase()];
    const b = el('button', f === state.format ? 'on' : '', `${info[0]}`);
    b.onclick = () => showFormat(f);
    bar.appendChild(b);
  });
}

function showFormat(f) {
  state.format = f;
  [...$('fmtbar').children].forEach((b, i) => b.classList.toggle('on', FORMATS[i] === f));
  buildFormatSide(f);
  if (f === 'geotiff') showRaster();
  else if (f === 'ply' || f === 'las') loadPoints(f);
  else loadMesh(f);
}

function buildFormatSide(f) {
  const d = state.data, a = d.formats.assets, side = $('fmtside');
  const info = d.formats.info[f] || [f.toUpperCase(), '', ''];
  const real = { glb: 'model.glb', obj: 'model.obj', fbx: 'model.fbx', ply: 'cloud.ply', las: 'cloud.las', geotiff: 'dsm.tif' }[f];
  const realBytes = d.formats.files[real];
  const webAsset = f === 'geotiff' ? null : (a.mesh?.[f] || a.points?.[f]);
  side.innerHTML = '';

  const head = el('div', 'panel pad');
  head.innerHTML = `<div style="display:flex;align-items:center;gap:9px;margin-bottom:6px">
      <h3 style="font-size:16px">${esc(info[0])}</h3><span class="tag accent">${esc(info[1])}</span>
      ${d.formats.produced.includes(f) ? '<span class="tag pass" style="margin-left:auto">written &amp; re-opened</span>' : ''}
    </div><p class="hint" style="margin:0">${esc(info[2])}</p>`;
  side.appendChild(head);

  const facts = el('div', 'panel pad');
  let rows = '';
  const kv = (k, v) => { rows += `<div class="k">${k}</div><div class="v">${v}</div>`; };
  if (f === 'geotiff') {
    const r = a.rasters || {};
    kv('Products', 'height model + orthophoto');
    if (r.dsm) { kv('Height model', `${mb(r.dsm.source_bytes)} · ${num(r.dsm.res_m?.[0], 2)} m/px`); kv('Height range', `${num(r.dsm.min_m, 1)} to ${num(r.dsm.max_m, 1)} m`); }
    if (r.ortho) kv('Orthophoto', `${mb(r.ortho.source_bytes)} · ${num(r.ortho.res_m?.[0], 3)} m/px`);
    kv('Coordinate system', esc(r.dsm?.crs || d.georeferencing?.crs || '—'));
    kv('Zone map', mb(d.formats.files['zone_map.tif']));
  } else if (f === 'ply' || f === 'las') {
    const p = a.points || {};
    kv('Points in the export', (p.points_full || 0).toLocaleString());
    kv('File size', mb(realBytes));
    kv('Shown here', `${(p.points_web || 0).toLocaleString()} (every ${Math.max(1, Math.round((p.points_full || 1) / (p.points_web || 1)))}ᵗʰ point)`);
    if (f === 'las') {
      const c = p.classes || {};
      kv('Ground points', ((c[2] || 0)).toLocaleString());
      kv('Above ground', ((c[1] || 0)).toLocaleString());
      kv('Low noise', ((c[7] || 0)).toLocaleString());
      if (d.formats.tiles) kv('Tiles written', `${d.formats.tiles} × 100 m`);
    } else {
      kv('Extra layers', 'per-point confidence, observation zone');
      kv('Mean confidence', num(p.confidence_mean, 2));
    }
  } else {
    const m = a.mesh || {};
    kv('Triangles in the export', (d.viewer?.faces || 0).toLocaleString());
    kv('File size', mb(realBytes));
    kv('Shown here', `${(m.faces || 0).toLocaleString()} triangles (${mb(webAsset?.bytes)})`);
    kv('Texture', mb(d.formats.files['textured_material_00_map_Kd.jpg']));
    if (f === 'glb') kv('Layers carried', 'texture, confidence, zone');
  }
  facts.innerHTML = `<div class="kv">${rows}</div>`;
  side.appendChild(facts);

  const dl = el('div', 'panel pad');
  dl.innerHTML = `<h4 style="font-size:13.5px;margin-bottom:8px">Take a copy</h4>
    <div class="actions">
      ${webAsset ? `<a class="btn sm" download href="runs/${d.key}/assets/${webAsset.file}">Download ${esc(info[0])} (${mb(webAsset.bytes)})</a>` : ''}
      ${f === 'geotiff' && a.rasters?.dsm ? `<a class="btn sm" download href="runs/${d.key}/assets/dsm.png">Height preview</a>
        <a class="btn sm" download href="runs/${d.key}/assets/ortho.png">Orthophoto preview</a>` : ''}
      ${f === 'glb' ? `<a class="btn sm" target="_blank" href="${d.key}/viewer/">Open the viewer the pipeline wrote ↗</a>` : ''}
    </div>
    <p class="hint" style="margin:9px 0 0">The web copy above is reduced so it loads over a browser connection.
      The full ${esc(info[0])} listed as “file size” is the one the pipeline wrote${f === 'geotiff' ? '' : `, ${mb(realBytes)}`}.</p>`;
  side.appendChild(dl);

  const why = el('div', 'panel pad');
  why.innerHTML = `<h4 style="font-size:13.5px;margin-bottom:6px">Why this format is in the set</h4>
    <p class="hint" style="margin:0">${esc({
      obj: 'It is the lowest common denominator for 3D geometry: if a tool opens any mesh at all, it opens OBJ.',
      ply: 'It keeps arbitrary per-point attributes, so confidence and observation zone travel with every point.',
      las: 'It is what surveyors and GIS analysts already use, with a standard classification scheme and tiling.',
      geotiff: 'A height model and an orthophoto drop straight into QGIS, ArcGIS or any mapping stack with no conversion.',
      glb: 'One self-contained file that any browser, phone or game engine can display without a plug-in.',
      fbx: 'The format Blender, 3ds Max, Maya and Unreal expect for handing a model between tools.',
    }[f] || '')}</p>`;
  side.appendChild(why);
}

// ── 3-D viewers ───────────────────────────────────────────────────────────────
let R = null;   // renderer context, created once
function ctx() {
  if (R) return R;
  const host = $('stage3d');
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 20000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.position.set(1, 2, 1.4); scene.add(sun);
  const resize = () => {
    const r = host.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix();
  };
  addEventListener('resize', resize); resize();
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
  R = { renderer, scene, camera, controls, host, resize, object: null };
  return R;
}
function clearObject() { const c = ctx(); if (c.object) { c.scene.remove(c.object); c.object = null; } }
function busy(on, text) {
  $('loading').hidden = !on;
  if (text) $('loading').querySelector('.txt').textContent = text;
}
function frameObject(obj) {
  const c = ctx();
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 1);
  const vHalf = THREE.MathUtils.degToRad(c.camera.fov / 2);
  const hHalf = Math.atan(Math.tan(vHalf) * c.camera.aspect);
  const dist = radius / Math.tan(Math.min(vHalf, hHalf)) * 0.82;     // fill the panel, leave a small margin
  c.camera.near = Math.max(radius / 4000, 0.01); c.camera.far = radius * 80; c.camera.updateProjectionMatrix();
  c.camera.position.copy(centre).add(new THREE.Vector3(0, dist * 0.72, dist * 0.68));
  c.controls.target.copy(centre); c.controls.update();
}
function setBg() {
  ctx().scene.background = new THREE.Color(0x0a1120);     // --bg-2, so the canvas meets the panel edge cleanly
}

function loadMesh(f) {
  const d = state.data, asset = d.formats.assets.mesh?.[f];
  $('rasterview').hidden = true;
  if (!asset) { busy(true, `No ${f.toUpperCase()} asset for this run`); return; }
  busy(true, `Loading the ${f.toUpperCase()} through its own reader…`);
  clearObject(); setBg();
  const url = `runs/${d.key}/assets/${asset.file}`;
  const done = (obj) => {
    obj.traverse((o) => {
      if (!o.isMesh) return;
      o.material = new THREE.MeshLambertMaterial({ vertexColors: !!o.geometry.attributes.color, color: o.geometry.attributes.color ? 0xffffff : 0xb9bec6, side: THREE.DoubleSide });
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    });
    ctx().scene.add(obj); ctx().object = obj;
    frameObject(obj); busy(false);
    meshModes(obj, f);
    $('viewhint').textContent = `Drag to orbit · right-drag to pan · scroll to zoom. Loaded from ${asset.file} with three.js ${f.toUpperCase()}Loader.`;
  };
  const fail = (e) => busy(true, `Could not read the ${f.toUpperCase()}: ${e?.message || e}`);
  if (f === 'glb') new GLTFLoader().load(url, (g) => done(g.scene), null, fail);
  else if (f === 'obj') new OBJLoader().load(url, done, null, fail);
  else new FBXLoader().load(url, done, null, fail);
}

function meshModes(root, f) {
  const bar = $('mode-bar'), legend = $('legend'), hud = $('hud');
  let mesh = null; root.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
  const attrs = mesh?.geometry?.attributes || {};
  // glTF custom attributes arrive lower-cased: _CONFIDENCE -> _confidence, _ZONE -> _zone.
  const conf = attrs._confidence || attrs._CONFIDENCE, zone = attrs._zone || attrs._ZONE;
  const has = { photo: !!attrs.color, confidence: !!conf, zone: !!zone };
  const modes = [['photo', 'Photo'], ['confidence', 'Confidence'], ['zone', 'Zones']].filter(([k]) => has[k]);
  bar.hidden = modes.length < 2; bar.innerHTML = '';
  const base = attrs.color ? attrs.color.clone() : null;
  const apply = (m) => {
    [...bar.children].forEach((b) => b.classList.toggle('on', b.dataset.m === m));
    if (m === 'photo' && base) mesh.geometry.setAttribute('color', base);
    else if (m === 'confidence') mesh.geometry.setAttribute('color', ramp(conf, confColour));
    else if (m === 'zone') mesh.geometry.setAttribute('color', ramp(zone, zoneColour));
    mesh.geometry.attributes.color.needsUpdate = true;
    drawLegend(m);
  };
  modes.forEach(([k, label]) => { const b = el('button', '', label); b.dataset.m = k; b.onclick = () => apply(k); bar.appendChild(b); });
  hud.hidden = false;
  hud.innerHTML = `<div class="hint" style="margin:0"><b>${(state.data.formats.assets.mesh.faces || 0).toLocaleString()}</b> triangles shown<br>
    of <b>${(state.data.viewer?.faces || 0).toLocaleString()}</b> in the ${f.toUpperCase()} file</div>`;
  legend.hidden = modes.length < 2;
  apply(modes.length ? modes[0][0] : 'photo');
}
const confColour = (v) => v < 0 ? [0.55, 0.57, 0.6] : [1 - 0.85 * v, 0.28 + 0.55 * v, 0.35 + 0.2 * v];
const zoneColour = (v) => v === 1 ? [0.16, 0.65, 0.27] : v === 2 ? [0.91, 0.65, 0.13] : v === 3 ? [0.85, 0.22, 0.22] : [0.5, 0.5, 0.5];
function ramp(attr, fn) {
  const n = attr.count, out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const c = fn(attr.getX(i)); out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2]; }
  return new THREE.BufferAttribute(out, 3);
}
function drawLegend(mode) {
  const l = $('legend');
  if (mode === 'confidence') {
    l.innerHTML = `<b>Confidence</b><div class="ramp" style="background:linear-gradient(90deg,#ff4747,#e8a521,#2aa64a)"></div>
      <div class="row" style="justify-content:space-between;color:var(--muted)"><span>seen by one camera</span><span>seen by many</span></div>
      <div class="row" style="color:var(--muted)"><span class="sw" style="background:#8c9299"></span>no measured point nearby</div>`;
  } else if (mode === 'zone') {
    l.innerHTML = `<b>How well it was observed</b>
      <div class="row"><span class="sw" style="background:#29a645"></span>Zone 1 — measured from several wide angles</div>
      <div class="row"><span class="sw" style="background:#e8a521"></span>Zone 2 — thinly seen, or filled from anchored depth</div>
      <div class="row"><span class="sw" style="background:#d93838"></span>Zone 3 — inferred, excluded from accuracy</div>`;
  } else {
    l.innerHTML = `<b>Photo colour</b><div class="row" style="color:var(--muted)">Colour sampled from the texture the pipeline painted onto the surface.</div>`;
  }
}

function loadPoints(f) {
  const d = state.data, asset = d.formats.assets.points?.[f];
  $('rasterview').hidden = true;
  if (!asset) { busy(true, `No ${f.toUpperCase()} asset for this run`); return; }
  busy(true, `Loading the ${f.toUpperCase()}…`);
  clearObject(); setBg();
  const url = `runs/${d.key}/assets/${asset.file}`;
  const show = (geom, extra) => {
    geom.computeBoundingBox();
    const c = geom.boundingBox.getCenter(new THREE.Vector3());
    geom.translate(-c.x, -c.y, -c.z);
    const size = geom.boundingBox.getSize(new THREE.Vector3()).length();
    const mat = new THREE.PointsMaterial({ size: size / 1400, vertexColors: true, sizeAttenuation: true });
    const pts = new THREE.Points(geom, mat);
    ctx().scene.add(pts); ctx().object = pts;
    frameObject(pts); busy(false);
    pointModes(pts, extra, f);
    $('viewhint').textContent = `Drag to orbit · scroll to zoom. Loaded from ${asset.file}` +
      (f === 'las' ? ' and decoded here with a LAS 1.2 reader written for this page.' : ' with three.js PLYLoader.');
  };
  if (f === 'ply') {
    new PLYLoader().load(url, (g) => {
      if (!g.attributes.color) g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 3).fill(0.7), 3));
      show(g, null);
    }, null, (e) => busy(true, `Could not read the PLY: ${e?.message || e}`));
  } else {
    fetch(url).then((r) => r.arrayBuffer()).then((buf) => {
      const las = parseLAS(buf);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(las.xyz, 3));
      g.setAttribute('color', new THREE.BufferAttribute(las.rgb, 3));
      show(g, las);
    }).catch((e) => busy(true, `Could not read the LAS: ${e?.message || e}`));
  }
}

/* Minimal LAS 1.2 reader: header, then fixed-size point records. Enough for point formats 2 and 3,
   which is what the export writes (x, y, z, classification, colour). */
function parseLAS(buffer) {
  const dv = new DataView(buffer);
  if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'LASF') throw new Error('not a LAS file');
  const offset = dv.getUint32(96, true), fmt = dv.getUint8(104) & 0x3f, len = dv.getUint16(105, true);
  let count = dv.getUint32(107, true);
  const sx = dv.getFloat64(131, true), sy = dv.getFloat64(139, true), sz = dv.getFloat64(147, true);
  const ox = dv.getFloat64(155, true), oy = dv.getFloat64(163, true), oz = dv.getFloat64(171, true);
  if (!count) count = Math.floor((buffer.byteLength - offset) / len);
  const hasRGB = fmt === 2 || fmt === 3 || fmt === 5 || fmt === 7 || fmt === 8;
  const rgbAt = fmt === 3 || fmt === 5 ? 28 : (fmt === 2 ? 20 : 30);
  const xyz = new Float32Array(count * 3), rgb = new Float32Array(count * 3), cls = new Uint8Array(count);
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < count; i++) {
    const p = offset + i * len;
    const x = dv.getInt32(p, true) * sx + ox, y = dv.getInt32(p + 4, true) * sy + oy, z = dv.getInt32(p + 8, true) * sz + oz;
    xyz[i * 3] = x; xyz[i * 3 + 1] = z; xyz[i * 3 + 2] = -y;      // to three.js Y-up
    mx += x; my += y; mz += z;
    cls[i] = dv.getUint8(p + 15) & 0x1f;
    if (hasRGB) { rgb[i * 3] = dv.getUint16(p + rgbAt, true) / 65535; rgb[i * 3 + 1] = dv.getUint16(p + rgbAt + 2, true) / 65535; rgb[i * 3 + 2] = dv.getUint16(p + rgbAt + 4, true) / 65535; }
    else { rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 0.7; }
  }
  return { xyz, rgb, cls, count, fmt, centre: [mx / count, my / count, mz / count] };
}

function pointModes(pts, las, f) {
  const bar = $('mode-bar'), legend = $('legend'), hud = $('hud');
  const base = pts.geometry.attributes.color.clone();
  const pos = pts.geometry.attributes.position;
  const modes = [['photo', 'True colour'], ['height', 'Height']];
  if (las) modes.push(['class', 'Classification']);
  bar.hidden = false; bar.innerHTML = '';
  const CLASS_COLOUR = { 1: [0.36, 0.55, 0.85], 2: [0.72, 0.55, 0.3], 7: [0.85, 0.3, 0.32] };
  const apply = (m) => {
    [...bar.children].forEach((b) => b.classList.toggle('on', b.dataset.m === m));
    const n = pos.count, out = new Float32Array(n * 3);
    if (m === 'photo') pts.geometry.setAttribute('color', base);
    else {
      if (m === 'height') {
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < n; i++) { const y = pos.getY(i); if (y < lo) lo = y; if (y > hi) hi = y; }
        for (let i = 0; i < n; i++) {
          const t = Math.min(1, Math.max(0, (pos.getY(i) - lo) / Math.max(hi - lo, 1e-6)));
          out[i * 3] = Math.min(1, 1.6 * t - 0.3 < 0 ? 0 : 1.6 * t - 0.3);
          out[i * 3 + 1] = Math.min(1, 1.4 * Math.pow(t, 0.8) * 0.85 + 0.12);
          out[i * 3 + 2] = Math.min(1, Math.max(0, 1.1 - 1.3 * t) * 0.8 + 0.15);
        }
      } else {
        for (let i = 0; i < n; i++) { const c = CLASS_COLOUR[las.cls[i]] || [0.6, 0.6, 0.6]; out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2]; }
      }
      pts.geometry.setAttribute('color', new THREE.BufferAttribute(out, 3));
    }
    pts.geometry.attributes.color.needsUpdate = true;
    legend.hidden = false;
    legend.innerHTML = m === 'class'
      ? `<b>ASPRS classification, read from the file</b>
         <div class="row"><span class="sw" style="background:#b88c4d"></span>2 — ground</div>
         <div class="row"><span class="sw" style="background:#5c8cd9"></span>1 — above ground (buildings, trees)</div>
         <div class="row"><span class="sw" style="background:#d94d52"></span>7 — low noise</div>`
      : m === 'height'
        ? `<b>Height above the vertical datum</b><div class="ramp" style="background:linear-gradient(90deg,#2a85b0,#43a047,#c9a227,#c25b3a)"></div>`
        : `<b>True colour</b><div class="row" style="color:var(--muted)">Colour carried per point, sampled from the photographs.</div>`;
  };
  modes.forEach(([k, label]) => { const b = el('button', '', label); b.dataset.m = k; b.onclick = () => apply(k); bar.appendChild(b); });
  const p = state.data.formats.assets.points || {};
  hud.hidden = false;
  hud.innerHTML = `<div class="hint" style="margin:0"><b>${(p.points_web || 0).toLocaleString()}</b> points shown<br>
    of <b>${(p.points_full || 0).toLocaleString()}</b> in the ${f.toUpperCase()} file</div>`;
  apply('photo');
}

function showRaster() {
  const d = state.data, r = d.formats.assets.rasters || {};
  clearObject(); busy(false);
  $('mode-bar').hidden = false; $('hud').hidden = false; $('legend').hidden = true;
  $('rasterview').hidden = false;
  const bar = $('mode-bar'); bar.innerHTML = '';
  const pick = (which) => {
    [...bar.children].forEach((b) => b.classList.toggle('on', b.dataset.m === which));
    const info = r[which]; if (!info) return;
    $('rasterimg').src = `runs/${d.key}/assets/${info.file}`;
    $('hud').innerHTML = `<div class="hint" style="margin:0">
      <b>${esc(info.crs || '')}</b><br>${num(info.res_m?.[0], 2)} m per pixel · ${info.px?.[0]}×${info.px?.[1]} shown<br>
      east ${num(info.bounds?.[0], 0)}–${num(info.bounds?.[2], 0)} m · north ${num(info.bounds?.[1], 0)}–${num(info.bounds?.[3], 0)} m</div>`;
    if (which === 'dsm' && info.min_m != null) {
      $('legend').hidden = false;
      $('legend').innerHTML = `<b>Surface height</b>
        <div class="ramp" style="background:linear-gradient(90deg,#2a85b0,#43a047,#c9a227,#c25b3a)"></div>
        <div class="row" style="justify-content:space-between;color:var(--muted)"><span>${num(info.ramp_low_m, 0)} m</span><span>${num(info.ramp_high_m, 0)} m</span></div>`;
    } else $('legend').hidden = true;
    $('viewhint').textContent = `Rendered from ${info.source_file} (${mb(info.source_bytes)}), which carries its coordinate system so it opens directly in QGIS or ArcGIS.`;
  };
  [['dsm', 'Height model'], ['ortho', 'Orthophoto']].forEach(([k, label]) => {
    if (!r[k]) return;
    const b = el('button', '', label); b.dataset.m = k; b.onclick = () => pick(k); bar.appendChild(b);
  });
  pick(r.dsm ? 'dsm' : 'ortho');
}

// ── quality report ────────────────────────────────────────────────────────────
function wireQA() {
  const open = () => { $('qa').classList.add('on'); $('scrim').classList.add('on'); };
  const close = () => { $('qa').classList.remove('on'); $('scrim').classList.remove('on'); };
  $('qa-open').onclick = open; $('qa-close').onclick = close; $('scrim').onclick = close;
  addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
}

const QA_TABS = [['processes', 'By process'], ['accuracy', 'Accuracy'], ['speed', 'Speed & 10-minute forecast'], ['formats', 'Outputs']];
function buildQA() {
  const d = state.data;
  $('qa-sub').textContent = `${d.video} · ${d.run} · processed on ${d.device === 'cuda' ? 'GPU' : d.device} in ${clock(d.time?.total_s)}`;
  const tabs = $('qa-tabs'); tabs.innerHTML = '';
  QA_TABS.forEach(([k, label]) => {
    const b = el('button', k === state.qaTab ? 'on' : '', label);
    b.onclick = () => { state.qaTab = k; buildQA(); $('qa-body').scrollTop = 0; };
    tabs.appendChild(b);
  });
  const body = $('qa-body'); body.innerHTML = '';
  ({ processes: qaProcesses, accuracy: qaAccuracy, speed: qaSpeed, formats: qaFormats }[state.qaTab] || qaProcesses)(body, d);
}

function scoreColour(s) { return s == null ? 'info' : s >= 85 ? 'pass' : s >= 60 ? 'warn' : 'fail'; }

function qaProcesses(body, d) {
  const scored = d.processes.filter((p) => p.score != null);
  const mean = scored.length ? scored.reduce((a, p) => a + p.score, 0) / scored.length : null;
  const intro = el('div', 'callout');
  intro.innerHTML = `<b>How to read this.</b> The video passes through the processes below in order. Each one is
    scored only on what it can actually measure on this flight: a check that meets its target counts fully, one
    that lands near it counts half, one below target counts nothing, and anything recorded for information only
    is shown but not scored. Every process states what its score covers and what it does not, so a high score is
    never mistaken for something it did not test.
    <div style="margin-top:9px">Across all ${scored.length} scored processes: <b>${num(mean, 1)} / 100</b>.</div>`;
  body.appendChild(intro);

  d.processes.forEach((p) => {
    const det = el('details', 'proc');
    const cls = scoreColour(p.score);
    det.innerHTML = `<summary>
        <div class="ring tag ${cls}" style="border-radius:50%">${p.score == null ? '—' : Math.round(p.score)}</div>
        <div><div class="st">${esc(p.stage)}</div><h4>${esc(p.process)}</h4></div>
        <div style="display:flex;gap:4px">${['fail', 'warn', 'info', 'pass'].filter((s) => p.counts[s]).map((s) => `<span class="tag ${s}">${p.counts[s]}</span>`).join('')}</div>
      </summary>`;
    const inner = el('div', 'inner');
    inner.innerHTML = `
      <p class="blurb">${esc(p.what)}</p>
      <p class="blurb"><b>The score accounts for:</b> ${esc(p.covers)}</p>
      <p class="blurb"><b>It does not measure:</b> ${esc(p.lacks)}</p>`;
    p.kpis.forEach((k) => {
      const row = el('div', 'kpi');
      row.innerHTML = `<div><div class="lbl">${esc(k.label)}</div><div class="det">${esc(k.detail || '')}</div></div>
        <div class="v">${esc(val(k.value))}${k.unit ? ' ' + esc(k.unit) : ''}<div class="tgt">target ${esc(k.target || '—')}</div></div>
        <div><span class="tag ${k.status}">${STATUS_WORD[k.status]}</span></div>`;
      inner.appendChild(row);
    });
    det.appendChild(inner);
    body.appendChild(det);
  });
}

function qaAccuracy(body, d) {
  const a = d.accuracy || {}, local = a.local || {}, geo = d.georeferencing || {}, z = d.zones || {};
  body.appendChild(el('div', 'callout', `<b>Two different questions.</b> <i>Shape</i> asks whether distances and
    heights within the model are right; <i>placement</i> asks whether the whole model sits in the right spot on
    the map. Shape comes from the reconstruction. Placement comes from the drone's own GPS, so it improves with
    better positioning data — RTK corrections or ground control points, both of which the pipeline accepts.`));

  const cards = el('div', 'grid cols-3');
  [['Cameras vs GPS', `${num(geo.rms_inliers_m, 2)} m`, `over ${geo.inliers ?? '—'} frames, after fitting the reconstruction to the track`],
   ['Reconstructed', `${num(z.coverage_pct ?? d.coverage?.coverage_pct, 1)}%`, `of the ground the cameras saw; the rest is outlined as gaps`],
   ['Coordinate system', esc(geo.crs || '—'), 'written into every georeferenced output']
  ].forEach(([k, v, s]) => {
    const c = el('div', 'panel pad');
    c.innerHTML = `<div class="hint">${k}</div><div class="metric"><b style="font-size:21px">${v}</b></div><div class="hint" style="margin-top:4px">${s}</div>`;
    cards.appendChild(c);
  });
  body.appendChild(cards);

  if (local.points_after_tile_offset) {
    const rows = [['zone1_measured', 'Zone 1 — measured from several wide angles'],
                  ['zone2_measured', 'Zone 2 — measured, but from fewer views'],
                  ['zone2_fill', 'Zone 2 — filled from anchored learned depth'],
                  ['all_points', 'All points together']]
      .filter(([k]) => local.points_after_tile_offset[k])
      .map(([k, label]) => {
        const s = local.points_after_tile_offset[k];
        return `<tr><td>${label}</td><td class="num">${num(s.rms_m, 2)}</td><td class="num">${num(s.nmad_m, 2)}</td><td class="num">${(s.n || 0).toLocaleString()}</td></tr>`;
      }).join('');
    const w = el('div', 'tablewrap');
    w.innerHTML = `<table class="data"><thead><tr><th>Surface, by how well it was observed</th><th>RMS (m)</th><th>NMAD (m)</th><th>Points</th></tr></thead><tbody>${rows}</tbody></table>`;
    body.appendChild(el('h4', '', 'Shape, against independent lidar'));
    body.appendChild(w);
    body.appendChild(el('p', 'hint', `Measured against ${esc((a.reference || {}).source || 'an independent reference surface')},
      after removing each 100 m tile's own placement offset so that what remains is shape alone. The ordering is the
      point: surfaces the system labelled well-observed really are the most accurate, and the filled surface lands
      as close as the thinly measured surface beside it. Zone 3 has no figure because nothing was measured there —
      it is reported as a gap instead of being filled.`));
  }
  if (!local.points_after_tile_offset) {
    body.appendChild(el('div', 'callout plain', `<b>No independent survey covers this site.</b> The comparison
      above needs a reference surface over the same ground — for the other flight that is public USGS lidar.
      Where none exists, the honest measures are the ones shown above: agreement between the reconstructed
      camera path and the GPS track, and the share of the scene that was actually observed. Supplying a lidar
      tile or a few surveyed ground points turns this into a full accuracy comparison without changing anything
      in the pipeline.`));
  }
  if (local.doming) {
    body.appendChild(el('div', 'callout plain', `<b>Known shape effect.</b> Over the ${num(local.doming.span_m, 0)} m
      length of this flight the two ends sit about ${num(Math.abs(local.doming.sag_m), 1)} m ${local.doming.sag_m < 0 ? 'below' : 'above'}
      the middle, with a ${num(local.doming.tilt_m_per_km, 0)} m/km tilt. This is the classic gentle bend a single
      straight strip produces, it is measured and reported on every run, and it is removed by the tile-by-tile
      comparison above. Flying a second crossing strip, or supplying ground control points, removes it outright.`));
  }
}

function qaSpeed(body, d) {
  const p = d.projection || {}, t = d.time || {};
  body.appendChild(el('div', 'callout', `<b>What the target asks.</b> The problem statement asks for a 10-minute
    video in under 15 minutes. This flight is ${clock(t.video_s)} long and took ${clock(t.total_s)}. The forecast
    column restates each measured time for a 10-minute flight on this same machine — a
    ${num(p.factor, 1)}× longer flight keeps ${num(p.factor, 1)}× more frames, and the work scales with frames.`));

  const rows = (p.stages || []).map((s) => {
    const [name] = STAGE_TEXT[s.stage] || [s.stage];
    return `<tr><td>${esc(name)}</td><td class="num">${clock(s.seconds)}</td><td class="num">${clock(s.projected_s)}</td></tr>`;
  }).join('');
  const w = el('div', 'tablewrap');
  w.innerHTML = `<table class="data"><thead><tr><th>Stage</th><th>Measured on this flight</th><th>Forecast for a 10-minute flight</th></tr></thead>
    <tbody>${rows}<tr style="font-weight:700"><td>Total</td><td class="num">${clock(t.total_s)}</td><td class="num">${clock(p.projected_total_s)}</td></tr></tbody></table>`;
  body.appendChild(w);

  const over = p.projected_total_s && p.budget_s ? p.projected_total_s / p.budget_s : null;
  body.appendChild(el('div', 'callout plain', `<b>Where that leaves us.</b> On this hardware the forecast is
    ${clock(p.projected_total_s)} against the ${clock(p.budget_s)} target${over ? `, about ${num(over, 1)}× the target` : ''}.
    The machine here is a ${d.device === 'cuda' ? 'shared slice of one GPU with 3 CPU cores' : String(d.device)};
    the reference machine in the specification has a full GPU and eight or more cores, and several of the heaviest
    steps — decoding, meshing, texturing — are limited by those cores rather than by the GPU. The measured route to
    the target is to select frames straight from the video's own keyframes instead of decoding everything, to run
    the stages overlapped rather than one after another, and to run on the reference hardware. Each of those is a
    change in how the work is scheduled, not a reduction in what is produced.`));

  body.appendChild(el('p', 'hint', esc(p.basis)));

  const bench = (d.benchmarks || {}).degradation;
  if (bench && bench.rows) {
    const by = {};
    bench.rows.forEach((r) => {
      if (r.kind === 'clean') { by.clean = r; return; }
      (by[r.kind] = by[r.kind] || {})[r.conditioning ? 'on' : 'off'] = r;
    });
    const NAME = { motion_blur: 'Motion blur', compression: 'Heavy compression', low_light: 'Low light',
      shadow: 'Hard shadows', gps_noise: 'Noisy GPS', dynamic_objects: 'Moving vehicles' };
    const cell = (r, k, dp = 1) => r && r[k] != null ? num(r[k], dp) : '—';
    const rows = Object.keys(NAME).filter((k) => by[k]).map((k) => {
      const on = by[k].on, off = by[k].off;
      const a = on?.camera_deviation?.rms_m, b = off?.camera_deviation?.rms_m;
      const better = a != null && b != null ? (a < b ? 'on' : (b < a ? 'off' : null)) : null;
      return `<tr><td>${NAME[k]}</td>
        <td class="num"><b${better === 'on' ? ' style="color:var(--ok)"' : ''}>${cell(on, 'camera_deviation.rms_m') === '—' ? num(a, 1) : num(a, 1)} m</b></td>
        <td class="num"><b${better === 'off' ? ' style="color:var(--ok)"' : ''}>${num(b, 1)} m</b></td>
        <td class="num">${cell(on, 'registered', 0)} / ${cell(off, 'registered', 0)}</td>
        <td class="num">${cell(on, 'coverage_pct')}% / ${cell(off, 'coverage_pct')}%</td></tr>`;
    }).join('');
    body.appendChild(el('h4', '', 'Behaviour on damaged footage'));
    body.appendChild(el('p', 'hint', `The same flight was reprocessed ${bench.rows.length} times with damage injected
      into the frames at ${esc((bench.severities || []).join(', '))} severity — once with the conditioning stage on
      and once with it off — to measure what that stage is worth. The figure is how far the camera path moved from
      the clean run: smaller is better, and green marks the setting that held up.`));
    const w = el('div', 'tablewrap');
    w.innerHTML = `<table class="data"><thead><tr><th>Damage</th><th>Cleanup on</th><th>Cleanup off</th>
      <th>Frames placed</th><th>Coverage</th></tr></thead><tbody>${rows}</tbody></table>`;
    body.appendChild(w);
    body.appendChild(el('p', 'hint', `Baseline with no damage: ${by.clean?.registered ?? '—'} frames placed,
      ${num(by.clean?.coverage_pct, 1)}% coverage, cameras ${num(by.clean?.cam_vs_gps_rms_m, 2)} m from GPS.
      Cleanup earns its place on blur and compression, where the solve fails without it. On low light and noisy
      GPS this run came out better without it — both are being worked on, and both are recorded here rather
      than left out.`));
  }
}

function qaFormats(body, d) {
  body.appendChild(el('div', 'callout', `<b>Every format is verified by reading it back.</b> After writing each
    file the pipeline re-opens it and checks it carries the geometry, the coordinates and the extra layers it
    should. The sizes below are the files this run produced.`));
  const rows = FORMATS.map((f) => {
    const info = d.formats.info[f] || [f.toUpperCase(), '', ''];
    const real = { glb: 'model.glb', obj: 'model.obj', fbx: 'model.fbx', ply: 'cloud.ply', las: 'cloud.las', geotiff: 'dsm.tif' }[f];
    const ok = d.formats.produced.includes(f);
    return `<tr><td><b>${esc(info[0])}</b><div class="det" style="color:var(--muted);font-size:12.3px">${esc(info[2])}</div></td>
      <td class="num">${mb(f === 'geotiff' ? (d.formats.files['dsm.tif'] || 0) + (d.formats.files['orthophoto.tif'] || 0) : d.formats.files[real])}</td>
      <td><span class="tag ${ok ? 'pass' : 'info'}">${ok ? 'verified' : '—'}</span></td></tr>`;
  }).join('');
  const w = el('div', 'tablewrap');
  w.innerHTML = `<table class="data"><thead><tr><th>Format</th><th>Size written</th><th>Re-opened</th></tr></thead><tbody>${rows}</tbody></table>`;
  body.appendChild(w);

  const z = d.zones || {};
  body.appendChild(el('h4', '', 'What travels with the model'));
  const extra = el('div', 'tablewrap');
  extra.innerHTML = `<table class="data"><tbody>
    <tr><td>Observation zone per point and per triangle</td><td class="num">${num(z.zone1_pct, 1)}% / ${num(z.zone2_pct, 1)}% / ${num(z.zone3_pct, 1)}% of ground</td></tr>
    <tr><td>Confidence per point</td><td class="num">${num((d.formats.assets.points || {}).confidence_mean, 2)} mean</td></tr>
    <tr><td>Gap outlines, drawn where nothing was observed</td><td class="num">${(z.gaps?.count ?? 0).toLocaleString()} areas · ${num(z.gaps?.total_m2, 0)} m²</td></tr>
    <tr><td>Point cloud classified (ground / above ground / noise)</td><td class="num">${Object.entries((d.formats.assets.points || {}).classes || {}).map(([k, v]) => `${k}: ${v.toLocaleString()}`).join(' · ')}</td></tr>
    ${d.formats.tiles ? `<tr><td>Tiled for large sites</td><td class="num">${d.formats.tiles} tiles</td></tr>` : ''}
  </tbody></table>`;
  body.appendChild(extra);

  const own = el('div', 'callout plain');
  own.innerHTML = `The pipeline writes its own full report with every run — every KPI, every timing and the
    accuracy comparison — as a standalone HTML file next to the outputs.
    <div class="actions" style="margin-top:9px"><a class="btn sm" target="_blank" href="${d.key}/report.html">Open the report this run wrote ↗</a></div>`;
  body.appendChild(own);

  if ((d.limitations || []).length) {
    body.appendChild(el('h4', '', 'Stated on every run'));
    const ul = el('ul'); ul.style.cssText = 'margin:0;padding-left:19px;color:var(--muted);font-size:13px;line-height:1.6';
    d.limitations.forEach((l) => ul.appendChild(el('li', '', esc(l))));
    body.appendChild(ul);
  }
}
