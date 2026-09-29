/* The tested flights on the landing page: one card each, straight from runs/index.json. Nothing is computed here. */
import { $, esc, num, clock, stat, RUN_BLURB } from './ui.js';

export function buildFlights(runs, open) {
  const box = $('land-flights');
  if (!box) return;
  if (!runs.length) { box.innerHTML = '<p class="hint">No tested flights are available.</p>'; return; }
  box.innerHTML = `<div class="flights">${runs.map((r) => {
    const [kind, blurb] = RUN_BLURB[r.key] || ['Flight', ''];
    return `<article class="flight panel">
      <div class="fthumb"><img src="runs/${esc(r.key)}/assets/ortho.png" alt="" loading="lazy"><span class="badge tag accent">${esc(kind)}</span></div>
      <div class="fbody">
        <h3>${esc(r.video)}</h3><p class="hint">${esc(blurb)}</p>
        <div class="statgrid">${stat('Video length', clock(r.video_s))}${stat('Processed in', clock(r.total_s))}${stat('Ground covered', `${num(r.coverage_pct, 1)}%`)}</div>
        <div class="actions"><button class="btn primary" type="button" data-open="${esc(r.key)}">Open this flight →</button></div>
      </div></article>`;
  }).join('')}</div>`;
  box.querySelectorAll('[data-open]').forEach((b) => { b.onclick = () => open(b.dataset.open); });
}
