/* Landing hero: a procedural node terrain that reads as a mountain ridge, with a geolocation marker on
   its saddle. Generated here in JavaScript (no model file), so the page paints immediately and the hero
   never touches runs/*\/assets/. Purely decorative: it shows no measured value.

   Resource rules (the outputs viewer owns another WebGL context):
     - the renderer is created lazily, on first activation;
     - the loop runs only while the landing page is the active step AND the hero is on screen;
     - pixel ratio is capped at 2;
     - prefers-reduced-motion renders single static frames (no rotation, pulse or intro);
     - no WebGL, or a lost context, swaps in the SVG fallback in the page. */
import * as THREE from 'three';

const NX = 44, NZ = 32;                                   // node grid: ~1,400 nodes
const WORLD = { x: 1.7, z: 1.25, y: 1.15 };               // half-width, half-depth, peak height
const RAMP = ['#1b3a8f', '#38bdf8', '#eaf6ff'];           // valley → slope → peak

// deterministic value noise, so the ridge is identical on every load
const hash = (i, j) => { let h = (Math.imul(i, 374761393) + Math.imul(j, 668265263)) ^ 0x5bd1e995; h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };
const smooth = (t) => t * t * (3 - 2 * t);
function vnoise(x, z) {
  const xi = Math.floor(x), zi = Math.floor(z), u = smooth(x - xi), v = smooth(z - zi);
  const a = hash(xi, zi), b = hash(xi + 1, zi), c = hash(xi, zi + 1), d = hash(xi + 1, zi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
const bump = (x, z, cx, cz, sx, sz, a) => a * Math.exp(-(((x - cx) ** 2) / sx + ((z - cz) ** 2) / sz));
const PEAK = [0.16, -0.06], SECOND = [-0.58, 0.16];       // dominant peak and its neighbour: the saddle lies between
function rawHeight(x, z) {                                // x, z in [-1, 1]
  const peaks = bump(x, z, PEAK[0], PEAK[1], 0.09, 0.13, 1.0) + bump(x, z, SECOND[0], SECOND[1], 0.07, 0.10, 0.66)
    + bump(x, z, 0.66, 0.30, 0.06, 0.09, 0.50) + bump(x, z, -0.05, 0.55, 0.05, 0.07, 0.28);
  const ridges = 0.10 * Math.sin(3.1 * x + 1.2) * Math.cos(2.4 * z + 0.4) + 0.06 * Math.sin(6.3 * x - 2.0 * z);
  const rough = 0.16 * (vnoise(x * 4.2 + 7, z * 4.2 + 3) - 0.5) + 0.07 * (vnoise(x * 9 + 1, z * 9 + 5) - 0.5);
  const edge = (1 - Math.abs(x) ** 3.4) * (1 - Math.abs(z) ** 3.4);
  return Math.max(0, (peaks + ridges + rough) * (0.25 + 0.75 * edge) * Math.sqrt(edge) + 0.02);
}

function terrain() {
  const pos = new Float32Array(NX * NZ * 3), col = new Float32Array(NX * NZ * 3), raw = new Float32Array(NX * NZ);
  const xs = new Float32Array(NX * NZ), zs = new Float32Array(NX * NZ);
  let lo = Infinity, hi = -Infinity;
  for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++) {
    const n = j * NX + i, inner = i > 0 && i < NX - 1 && j > 0 && j < NZ - 1;
    const jx = inner ? (hash(i, j) - 0.5) * 0.7 : 0, jz = inner ? (hash(j + 91, i + 17) - 0.5) * 0.7 : 0;
    xs[n] = -1 + 2 * (i + jx) / (NX - 1); zs[n] = -1 + 2 * (j + jz) / (NZ - 1);
    raw[n] = rawHeight(xs[n], zs[n]); lo = Math.min(lo, raw[n]); hi = Math.max(hi, raw[n]);
  }
  const ramp = RAMP.map((c) => new THREE.Color(c)), tmp = new THREE.Color();
  for (let n = 0; n < NX * NZ; n++) {
    const t = (raw[n] - lo) / (hi - lo);
    if (t < 0.55) tmp.copy(ramp[0]).lerp(ramp[1], t / 0.55); else tmp.copy(ramp[1]).lerp(ramp[2], (t - 0.55) / 0.45);
    pos.set([xs[n] * WORLD.x, t * WORLD.y, zs[n] * WORLD.z], n * 3); col.set([tmp.r, tmp.g, tmp.b], n * 3);
  }
  const idx = [];
  for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++) {
    const n = j * NX + i;
    if (i < NX - 1) idx.push(n, n + 1);
    if (j < NZ - 1) idx.push(n, n + NX);
    if (i < NX - 1 && j < NZ - 1) ((i + j) & 1) ? idx.push(n, n + NX + 1) : idx.push(n + 1, n + NX);   // alternating diagonals: reads as a triangulated mesh
  }
  // the saddle: lowest point on the ridge line between the two main peaks
  let saddle = { t: 0.5, h: Infinity };
  for (let t = 0.25; t <= 0.75; t += 0.01) {
    const h = rawHeight(PEAK[0] + (SECOND[0] - PEAK[0]) * t, PEAK[1] + (SECOND[1] - PEAK[1]) * t);
    if (h < saddle.h) saddle = { t, h };
  }
  const sx = PEAK[0] + (SECOND[0] - PEAK[0]) * saddle.t, sz = PEAK[1] + (SECOND[1] - PEAK[1]) * saddle.t;
  const marker = new THREE.Vector3(sx * WORLD.x, ((saddle.h - lo) / (hi - lo)) * WORLD.y, sz * WORLD.z);
  return { pos, col, idx, marker };
}

function sprite() {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d'), grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)'); grad.addColorStop(0.35, 'rgba(255,255,255,.85)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

const additive = { transparent: true, blending: THREE.AdditiveBlending, depthWrite: false };
function circle(radius, segs = 128) {
  const p = []; for (let k = 0; k < segs; k++) { const a = (k / segs) * Math.PI * 2; p.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius)); }
  return new THREE.BufferGeometry().setFromPoints(p);
}

export function createHero(host, fallback) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  // `hidden` is an HTML-element property; an <svg> needs the attribute itself removed.
  const showFallback = () => { host.querySelector('canvas')?.setAttribute('hidden', ''); fallback?.removeAttribute('hidden'); };
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' }); }
  catch { showFallback(); return { setActive() {}, dispose() {} }; }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  host.prepend(renderer.domElement);
  renderer.domElement.setAttribute('aria-hidden', 'true');

  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 60);
  const wrap = new THREE.Group(), world = new THREE.Group();
  wrap.add(world); scene.add(wrap);

  const t = terrain(), fade = [];                                     // fade: [material, base opacity]
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(t.pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(t.col, 3));
  const lineGeo = geo.clone(); lineGeo.setIndex(t.idx);
  const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true, opacity: 0.22, ...additive }));
  const map = sprite();
  const nodes = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.085, map, vertexColors: true, sizeAttenuation: true, opacity: 0.95, ...additive }));
  world.add(lines, nodes); fade.push([lines.material, 0.22], [nodes.material, 0.95]);

  // graticule on the ground plane: two rings, compass ticks, and a faint meridian arc over the ridge
  const ink = new THREE.Color('#7dd3fc');
  [[2.1, 0.22], [2.65, 0.14]].forEach(([r, o]) => {
    const m = new THREE.LineBasicMaterial({ color: ink, opacity: o, ...additive }); world.add(new THREE.LineLoop(circle(r), m)); fade.push([m, o]);
  });
  const ticks = []; for (let k = 0; k < 36; k++) { const a = (k / 36) * Math.PI * 2, r0 = 2.65, r1 = k % 9 === 0 ? 2.95 : 2.8;
    ticks.push(new THREE.Vector3(Math.cos(a) * r0, 0, Math.sin(a) * r0), new THREE.Vector3(Math.cos(a) * r1, 0, Math.sin(a) * r1)); }
  const tickMat = new THREE.LineBasicMaterial({ color: ink, opacity: 0.3, ...additive });
  world.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(ticks), tickMat)); fade.push([tickMat, 0.3]);
  const arc = []; for (let k = 0; k <= 64; k++) { const a = (k / 64) * Math.PI; arc.push(new THREE.Vector3(0, Math.sin(a) * 1.95, Math.cos(a) * 2.65)); }
  const arcGeo = new THREE.BufferGeometry().setFromPoints(arc), arcMat = new THREE.LineDashedMaterial({ color: ink, dashSize: 0.07, gapSize: 0.06, opacity: 0.2, ...additive });
  const meridian = new THREE.Line(arcGeo, arcMat); meridian.computeLineDistances(); world.add(meridian); fade.push([arcMat, 0.2]);

  // geolocation marker on the saddle: beam, spinning head, anchored ring, two pulsing rings
  const marker = new THREE.Group(); marker.position.copy(t.marker); world.add(marker);
  const beamMat = new THREE.MeshBasicMaterial({ color: '#7dd3fc', opacity: 0.7, ...additive });
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.95, 8), beamMat); beam.position.y = 0.475; marker.add(beam);
  const headMat = new THREE.MeshBasicMaterial({ color: '#e0f2fe' });
  const head = new THREE.Mesh(new THREE.OctahedronGeometry(0.075), headMat); head.position.y = 1.0; marker.add(head);
  const ringGeo = new THREE.RingGeometry(0.16, 0.178, 64);
  const mkRing = (o) => { const m = new THREE.MeshBasicMaterial({ color: '#bae6fd', opacity: o, side: THREE.DoubleSide, depthTest: false, ...additive });
    const r = new THREE.Mesh(ringGeo, m); r.rotation.x = -Math.PI / 2; r.position.y = 0.012; r.renderOrder = 5; marker.add(r); return r; };
  const still = mkRing(0.45), pulseA = mkRing(0.9), pulseB = mkRing(0.9);
  fade.push([beamMat, 0.7], [still.material, 0.45]);

  const basePulse = { a: pulseA.material.opacity, b: pulseB.material.opacity };
  let intro = reduced ? 1 : 0, active = false, visible = true, lost = false, raf = false;
  const tilt = { x: 0, y: 0, tx: 0, ty: 0 };
  const rad = (d) => (d * Math.PI) / 180;

  function size() {
    const w = Math.max(host.clientWidth, 1), h = Math.max(host.clientHeight, 1);
    renderer.setSize(w, h, false); camera.aspect = w / h;
    const vHalf = rad(camera.fov / 2), need = 2.7 / (Math.tan(vHalf) * camera.aspect);    // keep the graticule in frame on narrow panels (outer ticks may just touch the edge)
    const dist = Math.max(5.6, need);
    camera.position.set(0, dist * 0.36, dist); camera.lookAt(0, 0.45, 0); camera.updateProjectionMatrix();
  }
  function frame(now) {
    const dt = Math.min((now - (frame.last ?? now)) / 1000, 0.1); frame.last = now;
    if (!reduced) {
      intro = Math.min(1, intro + dt / 1.6);
      world.rotation.y += dt * 0.05;
      tilt.x += (tilt.tx - tilt.x) * Math.min(1, dt * 4); tilt.y += (tilt.ty - tilt.y) * Math.min(1, dt * 4);
      wrap.rotation.y = tilt.y; wrap.rotation.x = tilt.x;
      const s = now / 1000, k = (s % 2.4) / 2.4, k2 = ((s + 1.2) % 2.4) / 2.4;             // 2.4 s pulse cycle
      pulseA.scale.setScalar(0.5 + 2.2 * k); pulseA.material.opacity = basePulse.a * (1 - k) * intro;
      pulseB.scale.setScalar(0.5 + 2.2 * k2); pulseB.material.opacity = basePulse.b * (1 - k2) * intro;
      head.rotation.y = s * 1.4; head.position.y = 1.0 + Math.sin(s * 2.1) * 0.025;
    } else { pulseA.visible = pulseB.visible = false; }
    const e = 1 - (1 - intro) ** 3;
    fade.forEach(([m, o]) => { m.opacity = o * e; });
    world.scale.setScalar(0.92 + 0.08 * e); headMat.opacity = e; headMat.transparent = e < 1;
    renderer.render(scene, camera);
  }
  function sync() {                                                     // the one place the loop is started or stopped
    if (lost) return;
    const run = active && visible && !document.hidden;
    if (reduced) { renderer.setAnimationLoop(null); raf = false; if (run) frame(performance.now()); return; }
    if (run && !raf) { frame.last = undefined; renderer.setAnimationLoop(frame); raf = true; }
    else if (!run && raf) { renderer.setAnimationLoop(null); raf = false; }
  }

  const ro = new ResizeObserver(() => { size(); if (reduced && active) frame(performance.now()); }); ro.observe(host);
  const io = new IntersectionObserver((es) => { visible = es[es.length - 1].isIntersecting; sync(); }); io.observe(host);
  const onMove = (ev) => { const r = host.getBoundingClientRect();
    tilt.ty = rad(6) * Math.max(-1, Math.min(1, ((ev.clientX - r.left) / r.width - 0.5) * 2));
    tilt.tx = rad(6) * Math.max(-1, Math.min(1, ((ev.clientY - r.top) / r.height - 0.5) * 2)); };
  const onLeave = () => { tilt.tx = tilt.ty = 0; };
  if (!reduced) { host.addEventListener('pointermove', onMove); host.addEventListener('pointerleave', onLeave); }
  const onVis = () => sync(); document.addEventListener('visibilitychange', onVis);
  renderer.domElement.addEventListener('webglcontextlost', (ev) => { ev.preventDefault(); lost = true; renderer.setAnimationLoop(null); raf = false; showFallback(); });
  size();

  return {
    setActive(on) { active = on; if (on) size(); sync(); },
    dispose() {
      renderer.setAnimationLoop(null); ro.disconnect(); io.disconnect(); document.removeEventListener('visibilitychange', onVis);
      host.removeEventListener('pointermove', onMove); host.removeEventListener('pointerleave', onLeave);
      [geo, lineGeo, arcGeo, ringGeo, beam.geometry, head.geometry].forEach((g) => g.dispose()); map.dispose(); renderer.dispose(); renderer.forceContextLoss();
    },
  };
}
