/* Metrics and Forecast: two full-size dialogs over the output page, in one shared layout.
   Left two-thirds scrolls (one table per stage: metric, value, target, status; the same four columns the pipeline scores use);
   the right third is a fixed summary of the stage in view. Everything is read from data.json.
   Limitations are deliberately not shown in Metrics; they belong with the forecast. */
import { el, esc, num, clock, val, STATUS_WORD, scoreColour, STAGES } from './ui.js';
import { makeSheet } from './sheet.js';
import { stageViz, wireViz, forecastChart } from './charts.js';

const procsFor = (d, s) => (d.processes || []).filter((p) => (s.process ? p.process === s.process : s.processes ? s.processes.includes(p.process) : p.stage === s.stage));
const firstSentence = (t) => { const m = String(t || '').match(/^.*?[.!?](?=\s|$)/); return (m ? m[0] : String(t || '')).trim(); };
const count = (rows) => rows.reduce((a, r) => { a[r.status] = (a[r.status] || 0) + 1; return a; }, { pass: 0, warn: 0, fail: 0, info: 0 });

// One metric row: keyword and its one-line explanation, the value in a rounded box, the target, the status.
const row = (m) => `<tr><td><div class="mk">${esc(m.label)}</div><div class="md">${esc(m.detail || '')}</div></td>` +
  `<td><span class="vbox">${esc(m.value)}${m.unit ? `<small> ${esc(m.unit)}</small>` : ''}</span></td>` +
  `<td class="mt">${esc(m.target || '—')}</td><td><span class="tag ${esc(m.status)}">${STATUS_WORD[m.status] || ''}</span></td></tr>`;
const table = (heads) => `<div class="mtable"><table class="metrics"><thead><tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr></thead><tbody>${heads}</tbody></table></div>`;
const group = (title, score, rows) => `<tr class="grp"><th colspan="4"><span>${esc(title)}</span>${score == null ? '' : `<span class="pbox tag ${scoreColour(score)}">${Math.round(score)}</span>`}</th></tr>${rows.join('')}`;

// What the input check found, in words, from its own fields: the position log, the camera and how well video and GPS agree.
function inputFacts(ic) {
  const t = ic.telemetry || {}, cam = ic.camera || {}, sync = ic.sync || {}, bits = [];
  if (t.rows != null) bits.push(`${t.rows} position fixes over ${clock(t.duration_s)}${t.source ? `, from ${String(t.source).startsWith('klv') ? 'the video stream itself (MISB KLV)' : t.source}` : ''}`);
  if (cam.model || cam.hfov_deg != null) bits.push(`camera ${cam.model || `${num(cam.hfov_deg, 1)}° field of view`}${cam.source ? ` (${cam.source})` : ''}`);
  if (sync.r_best != null) bits.push(`image motion agrees with GPS speed at ${(sync.r_best * 100).toFixed(0)}%, at a ${num(sync.lag_s, 2)} s offset that is corrected automatically`);
  return bits.length ? `Found: ${bits.join('; ')}.` : `Verdict: ${String(ic.verdict || '').replace(/_/g, ' ').toLowerCase()}.`;
}

// The rows and the right-hand summary for one stage.
function stageContent(d, s) {
  if (s.id === 'check') {                                            // the input check has its own checks, grouped, and no per-process score
    const ic = d.input_check || {}, checks = ic.checks || [], c = count(checks);
    const groups = [...new Set(checks.map((x) => x.group))];
    const body = groups.map((g) => group(g, null, checks.filter((x) => x.group === g).map((x) => row({ label: x.label, detail: x.detail, value: val(x.value), target: '—', status: x.status })))).join('');
    return { table: table(body), counts: c, total: checks.length,
      summary: [`${s.desc} ${checks.length} checks were made in ${clock(ic.timing_s?.total)}: ${c.pass} met their target, ${c.warn} came near it, ${c.fail} fell below it and ${c.info} were recorded for information.`,
        inputFacts(ic)] };
  }
  const procs = procsFor(d, s), kp = procs.flatMap((p) => p.kpis), c = count(kp);
  const body = procs.map((p) => group(p.process, p.score, p.kpis.map((k) => row({ label: k.label, detail: k.detail, value: val(k.value), unit: k.unit, target: k.target, status: k.status })))).join('');
  return { table: table(body), counts: c, total: kp.length,
    summary: [`${s.desc} ${kp.length} measurements across ${procs.length} ${procs.length === 1 ? 'process' : 'processes'}: ${c.pass} met their target, ${c.warn} came near it, ${c.fail} fell below it and ${c.info} were recorded for information.`,
      procs.map((p) => `<span class="pl"><b>${esc(p.process)}</b>${p.score == null ? '' : ` (${Math.round(p.score)})`}: ${esc(firstSentence(p.what))}</span>`).join('')].filter(Boolean),
    raw: true };
}

// Measured time for a 10-minute video, from the pipeline's own projection in data.json. Each stage's allowance is its share of
// the 15-minute budget (Reconstruction owns three: Track B's and the refinement's are handed to Track A when they do not run).
const FSTAGES = [['preflight', 'Input check', ['preflight']], ['ingest', 'Ingest', ['ingest']], ['condition', 'Conditioning', ['condition']],
  ['track_a', 'Reconstruction', ['track_b', 'refine_ba', 'track_a_mvs']], ['fusion', 'Occlusion handling', ['fusion']], ['geo', 'Georeferencing', ['geo']], ['export', 'Outputs', ['export']]];
const KPI = (d, proc, key) => (d.processes || []).find((p) => p.process === proc)?.kpis.find((k) => k.key === key);

function forecastContent(d) {
  const p = d.projection || {}, budgets = d.time?.stage_budgets_s || {}, by = Object.fromEntries((p.stages || []).map((s) => [s.stage, s]));
  const rows = FSTAGES.filter(([k]) => by[k]).map(([k, label, keys]) => {
    const parts = keys.map((x) => budgets[x]), budget = parts.every((v) => v != null) ? parts.reduce((a, v) => a + v, 0) : null;
    return { label, measured: by[k].seconds, forecast: by[k].projected_s, budget };
  });
  const verdict = (f, b) => (b == null ? 'info' : f <= b ? 'pass' : 'fail');
  const time = rows.map((r) => row({ label: r.label, detail: `Measured ${clock(r.measured)} on this ${clock(p.video_s)} flight`, value: clock(r.forecast), target: r.budget != null ? `≤ ${clock(r.budget)}` : '—', status: verdict(r.forecast, r.budget) }));
  time.push(row({ label: 'Total', detail: `Measured ${clock(p.total_s)}, scaled by ${num(p.factor, 1)}× for a ${clock(p.target_video_s)} video`, value: clock(p.projected_total_s), target: `≤ ${clock(p.budget_s)}`, status: verdict(p.projected_total_s, p.budget_s) }));
  const kept = KPI(d, 'Frame selection', 'frames_selected'), same = [KPI(d, 'Coverage and gaps (§6.4)', 'coverage_pct'), KPI(d, 'Metric (vs telemetry)', 'cam_vs_gps_rms_m')].filter(Boolean);
  const vol = kept && p.factor ? [row({ label: 'Frames kept', detail: `${num(kept.value, 0)} kept on this flight, scaled by the same ${num(p.factor, 1)}×`, value: `≈ ${num(Math.round(kept.value * p.factor), 0)}`, target: '—', status: 'info' })] : [];
  const keep = same.map((k) => row({ label: k.label, detail: 'Depends on the flight, not on how long it is', value: val(k.value), unit: k.unit, target: k.target, status: k.status }));
  const body = group('Processing time for a 10-minute video', null, time) + (vol.length ? group('Volume', null, vol) : '') + (keep.length ? group('Unchanged by length', null, keep) : '');
  const over = p.projected_total_s && p.budget_s ? p.projected_total_s / p.budget_s : null;
  return { rows, table: table(body),
    summary: [`${p.basis || ''}`,
      `On this hardware the forecast is <b>${clock(p.projected_total_s)}</b> against the ${clock(p.budget_s)} target${over ? `, about ${num(over, 1)}× the target` : ''}. The machine that ran this flight is ${d.device === 'cuda' ? 'a shared slice of one GPU with 3 CPU cores' : esc(String(d.device))}; the reference machine has a full GPU and eight or more cores, and several of the heaviest steps (decoding, meshing, texturing) are limited by those cores rather than by the GPU.`,
      'The route to the target is to select frames straight from the video\'s own keyframes instead of decoding everything, to run the stages overlapped rather than one after another, and to run on the reference hardware. Each is a change in how the work is scheduled, not in what is produced.'] };
}

export function createPanels({ getData, getExtra, onState }) {
  const mRoot = document.getElementById('metrics-sheet');
  let stage = 0, extra = {};
  let silent = false;                                    // closing because the page is navigating must not rewrite the URL history is using
  const msheet = makeSheet(mRoot, { onClose: () => { if (!silent) onState(null); } });

  mRoot.innerHTML = `<div class="sheet-box" role="dialog" aria-modal="true" aria-labelledby="m-title">
    <header class="sheet-head"><div><h3 id="m-title">Metrics</h3><p class="hint" id="m-sub"></p></div>
      <div class="m-nav"><button class="btn sm" type="button" data-prev aria-label="Previous stage">←</button><button class="btn sm" type="button" data-next aria-label="Next stage">→</button>
      <button class="btn ghost sm" type="button" data-close data-autofocus>Close ✕</button></div></header>
    <ol class="mstrip" id="m-strip" aria-label="Stages"></ol>
    <div class="mbody"><div class="mleft" id="m-left" tabindex="0" aria-label="Metrics for the stage in view"></div><aside class="mright" id="m-right" aria-label="Summary of the stage in view"></aside></div></div>`;
  const $m = (s) => mRoot.querySelector(s);

  function render() {
    const d = getData(), s = STAGES[stage], c = stageContent(d, s);
    $m('#m-sub').textContent = `${d.video} · stage ${stage + 1} of ${STAGES.length}`;
    $m('#m-strip').innerHTML = STAGES.map((x, i) => `<li class="${i === stage ? 'current' : 'done'}"><button type="button" data-i="${i}" aria-label="Stage ${i + 1} of ${STAGES.length}: ${esc(x.title)}"${i === stage ? ' aria-current="step"' : ''}><span class="seg"></span><span class="lbl"><i>${i + 1}</i><span class="t">${esc(x.title)}</span></span></button></li>`).join('');
    $m('#m-strip').querySelectorAll('button').forEach((b) => { b.onclick = () => { stage = +b.dataset.i; render(); }; });
    const t = c.counts;
    $m('#m-left').innerHTML = `<div class="mhead"><h2>${esc(s.title)}</h2><div class="mcounts">` +
      ['pass', 'warn', 'fail', 'info'].filter((k) => t[k]).map((k) => `<span class="tag ${k}">${t[k]} ${STATUS_WORD[k].toLowerCase()}</span>`).join('') + `</div></div>` +
      `<div id="m-viz">${stageViz(d, s.id, extra)}</div>${c.table}`;
    wireViz($m('#m-left'));
    $m('#m-right').innerHTML = `<h4>Summary</h4>` + c.summary.map((p) => `<p>${c.raw ? p : esc(p)}</p>`).join('');
    $m('#m-left').scrollTop = 0; $m('#m-right').scrollTop = 0;
    $m('[data-prev]').disabled = stage === 0; $m('[data-next]').disabled = stage === STAGES.length - 1;
  }
  const step = (n) => { const to = Math.max(0, Math.min(STAGES.length - 1, stage + n)); if (to !== stage) { stage = to; render(); } };
  $m('[data-prev]').onclick = () => step(-1); $m('[data-next]').onclick = () => step(1);
  mRoot.addEventListener('keydown', (e) => {
    if (e.target.closest('input, select, textarea')) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); step(1); } else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
  });
  // Swipe right for the next stage, left for the previous (a mostly horizontal drag, so vertical scrolling is left alone).
  let sx = null, sy = null;
  const body = $m('.mbody');
  body.addEventListener('pointerdown', (e) => { sx = e.clientX; sy = e.clientY; });
  body.addEventListener('pointerup', (e) => {
    if (sx == null) return;
    const dx = e.clientX - sx, dy = e.clientY - sy; sx = sy = null;
    if (Math.abs(dx) > 70 && Math.abs(dx) > 1.6 * Math.abs(dy)) step(dx > 0 ? 1 : -1);
  });
  body.addEventListener('pointercancel', () => { sx = sy = null; });

  // Forecast: the same dialog layout as Metrics, without a stage strip.
  const fRoot = document.getElementById('forecast-sheet');
  let fsilent = false;
  const fsheet = makeSheet(fRoot, { onClose: () => { if (!fsilent) onState(null); } });
  fRoot.innerHTML = `<div class="sheet-box" role="dialog" aria-modal="true" aria-labelledby="f-title">
    <header class="sheet-head"><div><h3 id="f-title">Forecast</h3><p class="hint" id="f-sub"></p></div>
      <div class="m-nav"><button class="btn ghost sm" type="button" data-close data-autofocus>Close ✕</button></div></header>
    <div class="mbody"><div class="mleft" id="f-left" tabindex="0" aria-label="Forecast for a 10-minute video"></div><aside class="mright" id="f-right" aria-label="How to read the forecast"></aside></div></div>`;
  function renderForecast() {
    const d = getData(), c = forecastContent(d), p = d.projection || {};
    fRoot.querySelector('#f-sub').textContent = `${d.video} · measured on ${clock(p.video_s)} of video, forecast for ${clock(p.target_video_s)}`;
    fRoot.querySelector('#f-left').innerHTML = `<div class="mhead"><h2>A 10-minute video</h2></div><div class="vgrid">${forecastChart(d, c.rows)}</div>${c.table}`;
    fRoot.querySelector('#f-right').innerHTML = `<h4>How to read this</h4>` + c.summary.map((t) => `<p>${t}</p>`).join('');
    wireViz(fRoot.querySelector('#f-left'));
  }

  return {
    openForecast() { renderForecast(); fsheet.open(); onState('forecast'); },
    async openMetrics(at) { if (typeof at === 'number') stage = at; extra = (await getExtra?.()) || {}; render(); msheet.open(); onState('metrics'); },
    closeAll() { silent = true; fsilent = true; msheet.close(); fsheet.close(); silent = false; fsilent = false; },
  };
}
