/* Metrics and Forecast: two full-size dialogs over the output page, in one shared layout.
   Left two-thirds scrolls (one table per stage: metric, value, target, status; the same four columns the pipeline scores use);
   the right third is a fixed summary of the stage in view. Everything is read from data.json.
   Limitations are deliberately not shown in Metrics; they belong with the forecast. */
import { el, esc, num, clock, val, STATUS_WORD, scoreColour, STAGES } from './ui.js';
import { makeSheet } from './sheet.js';
import { stageViz, wireViz } from './charts.js';

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

  return {
    async openMetrics(at) { if (typeof at === 'number') stage = at; extra = (await getExtra?.()) || {}; render(); msheet.open(); onState('metrics'); },
    closeAll() { silent = true; msheet.close(); silent = false; },
  };
}
