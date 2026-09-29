/* Shared helpers and components for the demo console. Formatting and markup only: nothing in here
   reads, computes or invents a measured value; every number arrives from data.json through the caller. */

// ── helpers ───────────────────────────────────────────────────────────────────
export const $ = (id) => document.getElementById(id);
export const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const mb = (b) => b == null ? '—' : (b >= 1e6 ? (b / 1048576).toFixed(1) + ' MB' : (b / 1024).toFixed(0) + ' KB');
// Trailing zeros are trimmed only after a decimal point: 88.0 -> "88", but 100 stays "100" and 617000 stays "617000".
export const num = (v, d = 2) => {
  if (v == null || v === '' || Number.isNaN(+v)) return '—';
  if (typeof v !== 'number') return String(v);
  const s = v.toFixed(Math.abs(v) >= 100 ? 0 : d);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
};
export const clock = (s) => s == null ? '—' : (s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${s < 10 ? s.toFixed(1) : Math.round(s)} s`);
export const val = (v) => typeof v === 'boolean' ? (v ? 'yes' : 'no') : (typeof v === 'number' ? num(v, 3) : (v == null ? '—' : String(v)));
export const STATUS_WORD = { pass: 'Met', warn: 'Near target', fail: 'Below target', info: 'Recorded' };
export const RUN_BLURB = {
  esri: ['Surveillance aircraft', 'High, wide-angle pass over a river and woodland. Telemetry is MISB KLV carried inside the video stream itself — no separate file.'],
  dji: ['DJI quadcopter', 'Low, detailed pass along a 907 m straight line. Telemetry is a flight-log CSV recorded alongside the video.'],
};
export const RUN_LABEL = { esri: 'Esri', dji: 'DJI' };
export function scoreColour(s) { return s == null ? 'info' : s >= 85 ? 'pass' : s >= 60 ? 'warn' : 'fail'; }

// ── shared components (styles.css: .chips, .progress, .statgrid, .kpirow) ─────
export const chips = (list) => `<div class="chips">${list.map((c) => {
  const [text, tip] = Array.isArray(c) ? c : [c];
  return `<span class="chip"${tip ? ` title="${esc(tip)}"` : ''}>${esc(text)}</span>`;
}).join('')}</div>`;
export const stat = (label, value, context) => `<div class="stat"><div class="k">${esc(label)}</div>` +
  `<div class="v">${esc(value)}</div>${context ? `<div class="c">${esc(context)}</div>` : ''}</div>`;
export const kpiRow = (k) => `<div class="kpirow"><div><div class="lbl">${esc(k.label)}</div>` +
  `<div class="det">${esc(k.detail || '')}</div></div>` +
  `<div class="v">${esc(val(k.value))}${k.unit ? ' ' + esc(k.unit) : ''}<div class="tgt">target ${esc(k.target || '—')}</div></div>` +
  `<div><span class="tag ${esc(k.status)}">${STATUS_WORD[k.status] || ''}</span></div></div>`;
// steps: [{ id, title, accent? }]; each segment carries its own data-accent so it recolours itself.
export function progressBar(steps, currentId, onPick) {
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
