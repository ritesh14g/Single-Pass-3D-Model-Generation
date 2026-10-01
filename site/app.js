/* Single-pass drone 3D: the demo console.
   Four views (landing, input, running, output) and, over the output, Metrics and Forecast. Everything shown is read from
   a completed pipeline run's data.json; nothing here recomputes a measurement. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { createHero } from './hero.js';
import { $, el, esc, mb, num, clock, val, STATUS_WORD, scoreColour, chips, stat, kpiRow, RUN_BLURB, RUN_LABEL, STAGES } from './ui.js';
import { buildFlights } from './landing.js';
import { makeSheet } from './sheet.js';
import { createPanels } from './panels.js';
import { gpu, runPolicy, wireGpuPanel, onGpuChange } from './gpu.js';

const state = { runs: [], data: null, view: 'home', format: 'glb', files: { video: null, tel: null }, panel: null, replay: null };
const VIEWS = ['home', 'input', 'run', 'output'];

// ── boot ──────────────────────────────────────────────────────────────────────
// Anything that goes wrong is shown on the page rather than leaving a blank panel.
const errors = [];
function report(what, e) {
  errors.push(`${what}: ${e?.message || e}`);
  let box = $('errbox');
  if (!box) { box = el('div', 'callout'); box.id = 'errbox'; box.style.borderLeftColor = 'var(--bad)'; box.style.background = 'var(--bad-bg)'; document.querySelector('main').prepend(box); }
  box.innerHTML = `<b>Something did not load.</b> ${errors.map(esc).join('<br>')}`;
  if (!document.body.dataset.view) show('home');         // a failed start still shows the landing page and this message
}
addEventListener('error', (e) => report('Script', e.error || e.message));
addEventListener('unhandledrejection', (e) => report('Load', e.reason));

init().catch((e) => report('Startup', e));

async function init() {
  state.runs = (await (await fetch('runs/index.json')).json()).runs || [];
  watchHeader();
  buildFlights(state.runs, openFlight);
  wireNav(); wireInput(); wireHome();
  wireTheme();
  wireGpuPanel(); gpuSheet = makeSheet($('gpu-sheet'));
  $('gpu-open').onclick = () => gpuSheet.open();
  panels = createPanels({ getData: () => state.data, getExtra: loadExtra, onState: (p) => { state.panel = p; syncUrl(true); } });
  $('metrics-open').onclick = () => panels.openMetrics();
  $('forecast-open').onclick = () => panels.openForecast();
  $('foot').innerHTML = `Every figure on this page is read from a completed pipeline run: its input check, its
    stage scores and the files it wrote. Processed on an NVIDIA H100 80&nbsp;GB MIG&nbsp;2g.20gb slice
    (19.6&nbsp;GB, 3 CPU cores). Reference surface for the accuracy comparison: USGS 3DEP lidar.
    <a href="https://github.com/ritesh14g/Single-Pass-3D-Model-Generation">Source code</a>.`;
  addEventListener('popstate', () => route().catch((e) => report('Navigation', e)));
  await route();
}
let gpuSheet = null, panels = null;

// ── theme ─────────────────────────────────────────────────────────────────────
// Set before first paint by a small script in <head> (the saved choice, else the system setting). Here: the toggle, and following the
// system live for as long as nothing has been chosen. The 3-D viewport is dark in both themes.
const THEME_KEY = 'sp3d.theme';
const currentTheme = () => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
function setTheme(name, save) {
  document.documentElement.dataset.theme = name;
  if (save) { try { localStorage.setItem(THEME_KEY, name); } catch { /* the choice just will not be remembered */ } }
  const b = $('theme-toggle'), label = name === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
  b.setAttribute('aria-pressed', String(name === 'dark')); b.setAttribute('aria-label', label); b.title = label;
  hero?.setTheme(name);
}
function wireTheme() {
  $('theme-toggle').onclick = () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark', true);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    let saved = null; try { saved = localStorage.getItem(THEME_KEY); } catch { /* none */ }
    if (saved !== 'light' && saved !== 'dark') setTheme(e.matches ? 'dark' : 'light', false);
  });
  setTheme(currentTheme(), false);
}

// The sticky header's real height, so anything pinned under it sits flush at any width.
function watchHeader() {
  const head = document.querySelector('header.top');
  const set = () => document.documentElement.style.setProperty('--head-h', head.offsetHeight + 'px');
  new ResizeObserver(set).observe(head); set();
}

// ── views and history ─────────────────────────────────────────────────────────
// ?view=home|input|output, ?run=<key>, ?format=<f>, ?panel=metrics|forecast. The old ?step=<id> links still land somewhere sensible.
const LEGACY = { input: 'input', export: 'output' };
async function route() {
  const q = new URLSearchParams(location.search);
  let view = q.get('view') || LEGACY[q.get('step')] || (q.get('step') && q.get('step') !== 'home' ? 'output' : 'home');
  let panel = q.get('panel') || (q.get('step') && !['home', 'input', 'export'].includes(q.get('step')) ? 'metrics' : null);
  if (!VIEWS.includes(view)) view = 'home';
  if (view === 'run') view = 'output';                    // a running view cannot be reopened from a link; the model it produced can
  if (view === 'output') {
    const key = state.runs.some((r) => r.key === q.get('run')) ? q.get('run') : (state.data?.key || state.runs[0]?.key);
    if (!key) view = 'home'; else if (state.data?.key !== key) await loadRun(key);
  }
  const fmt = q.get('format'); if (FORMATS.includes(fmt)) state.format = fmt;
  show(view);
  if (view === 'output' && (panel === 'metrics' || panel === 'forecast')) openPanel(panel);
}

function urlFor() {
  const q = new URLSearchParams();
  if (state.view !== 'home') q.set('view', state.view);
  if ((state.view === 'run' || state.view === 'output') && state.data) q.set('run', state.data.key);
  if (state.view === 'output') q.set('format', state.format);
  if (state.panel && state.view === 'output') q.set('panel', state.panel);
  return q.toString() ? `?${q}` : location.pathname;
}
function syncUrl(replace) {
  const u = urlFor();
  if (u === location.pathname + location.search || u === location.search) return;
  history[replace ? 'replaceState' : 'pushState']({}, '', u);
}
function go(view, opts = {}) { show(view); syncUrl(opts.replace); }

function show(view) {
  if (state.view === 'run' && view !== 'run') cancelReplay();
  if (view !== 'output') { panels?.closeAll(); state.panel = null; }
  state.view = view;
  document.body.dataset.view = view;
  document.querySelectorAll('.step').forEach((s) => s.classList.toggle('on', s.dataset.step === view));
  $('gpu-open').hidden = !(view === 'home' || view === 'input');
  $('metrics-open').hidden = view !== 'output';
  $('forecast-open').hidden = view !== 'output';
  $('scroll-to-flights').hidden = view !== 'home';
  if (view === 'home') $('scroll-to-flights').classList.remove('past');   // reset so it is on screen again the next time home is shown
  window.scrollTo({ top: 0, behavior: 'auto' });
  viewerLoop(view === 'output');
  if (view === 'output') showFormat(state.format);
  syncHero();
}

function wireNav() {
  $('home-link').onclick = (e) => { e.preventDefault(); go('home'); };
  document.addEventListener('click', (e) => { const b = e.target.closest('[data-go]'); if (b) go(b.dataset.go); });
  $('new-model').onclick = () => go('input');
}

// ── landing ───────────────────────────────────────────────────────────────────
// The hero owns its own WebGL context, created on first visit and paused whenever the landing page is not showing.
let hero = null;
function syncHero() {
  if (state.view === 'home') { hero ??= createHero($('hero-viz'), $('hero-fallback')); hero.setActive(true); }
  else hero?.setActive(false);
}
function wireHome() {
  $('hero-run').onclick = () => go('input');
  const cue = $('scroll-to-flights');
  cue.onclick = () => $('l-fl').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  addEventListener('scroll', () => cue.classList.toggle('past', scrollY > 24), { passive: true });
}

async function openFlight(key) {
  await loadRun(key);
  go('run');
  startReplay();
}

// ── input ─────────────────────────────────────────────────────────────────────
function wireInput() {
  const mk = (zoneId, listId, kind) => {
    const zone = $(zoneId), list = $(listId);
    const input = el('input'); input.type = 'file'; input.hidden = true;
    input.accept = kind === 'video' ? 'video/*,.ts,.mpeg4,.h264' : '.srt,.csv,.txt,.gpx,.kml,.json,.geojson,.ulg,.bin,.tlog';
    zone.after(input);
    const clear = () => { state.files[kind] = null; list.innerHTML = ''; input.value = ''; paintRun(); };
    const accept = async (file) => {
      if (!file) return;
      state.files[kind] = file; paintRun();
      list.innerHTML = '';
      const item = el('div', 'fileitem');
      item.innerHTML = `<span>📄</span><div><b>${esc(file.name)}</b><br><span class="k">${mb(file.size)} · reading…</span></div><button class="x" type="button" aria-label="Remove ${esc(file.name)}">✕</button>`;
      item.querySelector('.x').onclick = clear;
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
  $('run-btn').onclick = startOwnRun;
  onGpuChange(paintRun);
  paintRun();
}
// Run stays disabled until there is a video; the line beside it says what will happen if it is pressed.
function paintRun() {
  const has = !!state.files.video, ok = runPolicy().allowed;
  $('run-btn').disabled = !has;
  const hint = $('run-hint');
  hint.className = 'runhint' + (has && !ok ? ' warn' : '');
  hint.textContent = !has ? 'Choose a video to begin.' : (ok ? 'Ready to run.' : 'GPU not connected. A run needs one.');
  if (ok) $('run-alert').hidden = true;
}
function startOwnRun() {
  const policy = runPolicy(), box = $('run-alert');
  if (!policy.allowed) {
    box.hidden = false;
    box.innerHTML = `<div><b>GPU not connected.</b> ${policy.message}</div><button class="btn sm" type="button" id="alert-gpu">Connect GPU</button>`;
    $('alert-gpu').onclick = () => gpuSheet.open();
    return;
  }
  box.hidden = false;
  box.innerHTML = '<div><b>A GPU is connected,</b> but submitting a run needs the run server, which is not part of this build yet.</div>';
}

// ── reading the chosen files in the browser ───────────────────────────────────
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


// The one continuous series the charts need beyond data.json: each camera's residual along the flight.
const extraCache = {};
async function loadExtra() {
  const key = state.data.key;
  if (!(key in extraCache)) extraCache[key] = await fetch(`runs/${key}/residuals.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  return { residuals: extraCache[key] };
}

// ── run loading ───────────────────────────────────────────────────────────────
async function loadRun(key) {
  state.data = await (await fetch(`runs/${key}/data.json`)).json();
  buildFormatBar();
  $('out-eyebrow').textContent = `${state.data.video} · processed in ${clock(state.data.time?.total_s)}`;
}

// ── the run view ──────────────────────────────────────────────────────────────
// The strip fills stage by stage. On this build it replays the measured stage times of a completed run (and says so);
// the same strip is what a live run's progress events will drive.
function cancelReplay() { if (state.replay) state.replay.cancel = true; state.replay = null; }
async function startReplay() {
  cancelReplay();
  const d = state.data, token = { cancel: false }; state.replay = token;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const secs = STAGES.map((s) => s.seconds(d) ?? 0), total = secs.reduce((a, b) => a + b, 0) || 1;
  const nap = (ms) => new Promise((r) => setTimeout(r, ms));
  $('run-strip').innerHTML = STAGES.map((s) => `<li data-id="${s.id}"><div class="rs-top"><span class="rs-name">${esc(s.title)}</span><span class="rs-time">—</span></div><div class="rs-track"><i></i></div></li>`).join('');
  $('run-eyebrow').textContent = `Creating a 3D model from ${d.video}`;
  $('run-title').textContent = 'Working'; $('run-now').textContent = '';
  $('run-note').textContent = 'Replaying the measured stage times of a completed run.';
  $('run-skip').onclick = () => { cancelReplay(); go('output', { replace: true }); };
  const skip = $('run-skip'); skip.textContent = 'Skip to the model →'; skip.className = 'btn ghost sm';
  for (let i = 0; i < STAGES.length; i++) {
    if (token.cancel) return;
    const s = STAGES[i], li = $('run-strip').children[i], fill = li.querySelector('i'), ms = still ? 120 : Math.max(500, (secs[i] / total) * 7500);
    li.classList.add('active');
    $('run-title').textContent = `Stage ${i + 1} of ${STAGES.length}`;
    $('run-now').innerHTML = `<b>${esc(s.title)}</b> · ${esc(s.desc)}`;
    fill.style.transition = still ? 'none' : `width ${ms}ms linear`;
    await nap(30); fill.style.width = '100%';
    await nap(ms);
    if (token.cancel) return;
    li.classList.replace('active', 'done'); li.querySelector('.rs-time').textContent = clock(secs[i]);
  }
  $('run-title').textContent = 'Your model is ready'; $('run-now').textContent = 'All stages complete.';
  // Finishing does not navigate: the same button becomes the primary action and the person decides when to move on.
  if (token.cancel) return;
  state.replay = null;
  skip.textContent = 'View output →'; skip.className = 'btn primary'; skip.focus({ preventScroll: true });
  $('run-note').textContent = 'The run is complete.';
}

// ── Metrics and Forecast (panels.js, added in the next stages) ────────────────
function openPanel(name) { if (name === 'metrics') panels.openMetrics(); else if (name === 'forecast') panels.openForecast(); }

// ── step 4: outputs ───────────────────────────────────────────────────────────
const FORMATS = ['glb', 'obj', 'fbx', 'ply', 'las', 'geotiff'];
function buildFormatBar() {
  const bar = $('fmtbar'); bar.innerHTML = '';
  FORMATS.forEach((f) => {
    const info = state.data.formats.info[f] || [f.toUpperCase(), '', ''];
    const ok = state.data.formats.produced.includes(f);
    const b = el('button', f === state.format ? 'on' : '',
      `<span class="fn">${esc(info[0])}</span><span class="fk">${esc(info[1])}</span><span class="fv${ok ? '' : ' no'}">${ok ? '✓ written &amp; re-opened' : 'not written'}</span>`);
    b.type = 'button'; b.dataset.format = f;
    b.onclick = () => showFormat(f);
    bar.appendChild(b);
  });
}

function showFormat(f) {
  state.format = f; syncUrl(true);
  [...$('fmtbar').children].forEach((b, i) => b.classList.toggle('on', FORMATS[i] === f));
  buildFormatSide(f);
  if (f === 'geotiff') showRaster();
  else if (f === 'ply' || f === 'las') loadPoints(f);
  else loadMesh(f);
}

function buildFormatSide(f) {
  const d = state.data, a = d.formats.assets, side = $('fmtside'), below = $('fmtbelow');
  const info = d.formats.info[f] || [f.toUpperCase(), '', ''];
  const real = { glb: 'model.glb', obj: 'model.obj', fbx: 'model.fbx', ply: 'cloud.ply', las: 'cloud.las', geotiff: 'dsm.tif' }[f];
  const realBytes = d.formats.files[real];
  const webAsset = f === 'geotiff' ? null : (a.mesh?.[f] || a.points?.[f]);
  side.innerHTML = ''; below.innerHTML = '';

  let rows = '';
  const kv = (k, v) => { rows += `<div class="kpirow two"><div class="lbl">${k}</div><div class="v">${v}</div></div>`; };
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
  const card = el('div', 'factcard glass');
  card.innerHTML = `<div class="fhead"><h3>${esc(info[0])}</h3><span class="tag accent">${esc(info[1])}</span>` +
    `${d.formats.produced.includes(f) ? '<span class="tag pass">written &amp; re-opened</span>' : ''}` +
    `<button class="ftoggle" type="button" aria-expanded="true" title="Show or hide the details">Details</button></div>` +
    `<div class="fbodytxt"><p class="hint">${esc(info[2])}</p><div class="factrows">${rows}</div></div>`;
  card.querySelector('.ftoggle').onclick = (e) => {
    const off = side.classList.toggle('collapsed'); e.currentTarget.setAttribute('aria-expanded', String(!off));
  };
  side.appendChild(card);

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
  below.appendChild(dl);

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
  below.appendChild(why);
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
  const loop = () => { controls.update(); renderer.render(scene, camera); };
  renderer.setAnimationLoop(loop);
  R = { renderer, scene, camera, controls, host, resize, loop, object: null };
  return R;
}
// The viewer renders only while the Outputs step is showing; no loop is left running on any other page.
function viewerLoop(on) {
  if (!R) return;
  R.renderer.setAnimationLoop(on ? R.loop : null);
  if (on) R.resize();
}
function clearObject() { const c = ctx(); if (c.object) { c.scene.remove(c.object); c.object = null; } }
function busy(on, text, failed) {
  const box = $('loading');
  box.hidden = !on; box.classList.toggle('err', !!failed);
  if (text) box.querySelector('.txt').textContent = text;
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
  if (!asset) { busy(true, `No ${f.toUpperCase()} asset for this run`, true); return; }
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
  const fail = (e) => busy(true, `Could not read ${asset.file} with ${{ glb: 'GLTFLoader', obj: 'OBJLoader', fbx: 'FBXLoader' }[f]}: ${e?.message || e}`, true);
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
  if (!asset) { busy(true, `No ${f.toUpperCase()} asset for this run`, true); return; }
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
    }, null, (e) => busy(true, `Could not read ${asset.file} with PLYLoader: ${e?.message || e}`, true));
  } else {
    fetch(url).then((r) => r.arrayBuffer()).then((buf) => {
      const las = parseLAS(buf);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(las.xyz, 3));
      g.setAttribute('color', new THREE.BufferAttribute(las.rgb, 3));
      show(g, las);
    }).catch((e) => busy(true, `Could not read ${asset.file} with the LAS reader: ${e?.message || e}`, true));
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
  $('mode-bar').hidden = false; $('hud').hidden = true; $('legend').hidden = true;     // the raster's numbers sit beside the image, not in the corner HUD
  $('rasterview').hidden = false;
  const bar = $('mode-bar'); bar.innerHTML = '';
  const pick = (which) => {
    [...bar.children].forEach((b) => b.classList.toggle('on', b.dataset.m === which));
    const info = r[which]; if (!info) return;
    $('rasterimg').src = `runs/${d.key}/assets/${info.file}`;
    const row = (k, v) => `<div class="kpirow two"><div class="lbl">${k}</div><div class="v">${v}</div></div>`;
    $('rastercap').textContent = which === 'dsm' ? 'Height model' : 'Orthophoto';
    $('rasterinfo').innerHTML = `<div class="rinfo-h">${which === 'dsm' ? 'Height model' : 'Orthophoto'}</div>` +
      row('Coordinate system', esc(info.crs || '—')) + row('Resolution', `${num(info.res_m?.[0], 2)} m per pixel`) +
      row('Shown here', `${info.px?.[0]}×${info.px?.[1]} px`) +
      row('East', `${num(info.bounds?.[0], 0)}–${num(info.bounds?.[2], 0)} m`) + row('North', `${num(info.bounds?.[1], 0)}–${num(info.bounds?.[3], 0)} m`);
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

