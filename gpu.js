/* Connect GPU: the connection target and the rule that follows from it.

   This build has no run server, so nothing here can connect yet, and the panel says so. What it does do for real:
   it keeps a target (server address, device, and SSH or Jupyter details), it probes `<server>/api/health` when asked, and
   it enforces the rule that a run only starts with a GPU attached. The health reply it expects is
   { ok: true, gpu: { name, memory_gb } }, so the server can be added later without changing this file.
   Secrets are never stored: an SSH key is a *path* the server reads, and a Jupyter token lives in memory for the session. */
import { $ } from './ui.js';

const KEY = 'sp3d.gpu.target';
export const gpu = { connected: false, name: '', memoryGb: null, checked: false, message: '' };
const listeners = [];
export const onGpuChange = (fn) => listeners.push(fn);
const set = (patch) => { Object.assign(gpu, patch); listeners.forEach((f) => f(gpu)); };

// The rule: no GPU, no run. The pipeline would otherwise fall back to the CPU and take many hours.
export function runPolicy() {
  return gpu.connected ? { allowed: true }
    : { allowed: false, reason: 'no-gpu', message: 'Running the pipeline on the CPU would take many hours, so this run has not been started.' };
}

const store = {
  read() { try { return JSON.parse(localStorage.getItem(KEY) || 'null'); } catch { return null; } },
  write(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); return true; } catch { return false; } },
};

let kind = 'ssh';
const val = (id) => $(id).value.trim();
function collect() {
  return { server: val('gpu-server-url'), device: $('gpu-device').value, memoryGb: val('gpu-mem') || null, kind,
    ssh: { host: val('ssh-host'), port: val('ssh-port'), user: val('ssh-user'), keyPath: val('ssh-key'), dir: val('ssh-dir') },
    jupyter: { url: val('jup-url'), dir: val('jup-dir') } };                    // the token is deliberately not part of what is saved
}
function fill(t) {
  if (!t) return;
  $('gpu-server-url').value = t.server || ''; $('gpu-device').value = t.device || 'auto'; $('gpu-mem').value = t.memoryGb || '';
  $('ssh-host').value = t.ssh?.host || ''; $('ssh-port').value = t.ssh?.port || 22; $('ssh-user').value = t.ssh?.user || '';
  $('ssh-key').value = t.ssh?.keyPath || ''; $('ssh-dir').value = t.ssh?.dir || '';
  $('jup-url').value = t.jupyter?.url || ''; $('jup-dir').value = t.jupyter?.dir || '';
  setKind(t.kind || 'ssh');
}
function setKind(k) {
  kind = k;
  document.querySelectorAll('#gpu-sheet .seg button').forEach((b) => { const on = b.dataset.kind === k; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
  $('gpu-ssh').hidden = k !== 'ssh'; $('gpu-jupyter').hidden = k !== 'jupyter';
}

function validate(t) {
  if (t.server && !/^https?:\/\/\S+$/i.test(t.server)) return 'The run server address must start with http:// or https://.';
  if (t.kind === 'ssh' && (t.ssh.host || t.ssh.user) && !(t.ssh.host && t.ssh.user)) return 'An SSH target needs both a host and a user.';
  if (t.kind === 'jupyter' && t.jupyter.url && !/^https?:\/\/\S+$/i.test(t.jupyter.url)) return 'The Jupyter server URL must start with http:// or https://.';
  return '';
}

export async function testConnection() {
  const t = collect(), server = (t.server || '').replace(/\/$/, '');
  if (!server) { set({ connected: false, checked: true, message: 'Enter the address of the run server first.' }); return; }
  set({ connected: false, checked: false, message: 'Testing the connection…' });
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 3500);
  try {
    const r = await fetch(`${server}/api/health`, { signal: ctl.signal, mode: 'cors' });
    const j = await r.json();
    if (j?.ok && j.gpu?.name) { set({ connected: true, checked: true, name: j.gpu.name, memoryGb: j.gpu.memory_gb ?? null, message: '' }); $('gpu-found').textContent = `${j.gpu.name}${j.gpu.memory_gb ? ` · ${j.gpu.memory_gb} GB` : ''}`; }
    else set({ connected: false, checked: true, message: j?.ok ? 'The run server answered, but it sees no CUDA GPU.' : 'The run server answered, but reported a problem.' });
  } catch {
    set({ connected: false, checked: true, message: `No run server answered at ${server}. The run server is not part of this build yet, so nothing can connect for now.` });
  } finally { clearTimeout(timer); }
}

function paintStatus() {
  const box = $('gpu-status'), dot = $('gpu-dot'), label = $('gpu-open').querySelector('span');
  box.className = 'gpu-status ' + (gpu.connected ? 'ok' : (gpu.message.startsWith('Testing') ? 'info' : 'warn'));
  box.innerHTML = gpu.connected
    ? `<b>Connected</b> · ${gpu.name}${gpu.memoryGb ? ` · ${gpu.memoryGb} GB` : ''}`
    : `<b>GPU not connected.</b> Runs are not started without one: on the CPU the pipeline would take many hours.` +
      (gpu.message ? `<div class="gpu-msg">${gpu.message.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))}</div>` : '');
  dot.className = 'gpu-dot ' + (gpu.connected ? 'ok' : 'off');
  label.textContent = gpu.connected ? `GPU · ${gpu.name}` : 'Connect GPU';
}

export function wireGpuPanel() {
  fill(store.read());
  document.querySelectorAll('#gpu-sheet .seg button').forEach((b) => { b.onclick = () => setKind(b.dataset.kind); });
  $('gpu-test').onclick = testConnection;
  $('gpu-save').onclick = () => {
    const t = collect(), err = validate(t), out = $('gpu-saved');
    if (err) { out.textContent = err; out.className = 'hint bad'; return; }
    out.textContent = store.write(t) ? 'Saved in this browser. The Jupyter token is not saved.' : 'This browser would not let the target be saved.';
    out.className = 'hint';
  };
  onGpuChange(paintStatus); paintStatus();
}
