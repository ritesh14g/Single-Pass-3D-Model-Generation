/* One implementation of a modal dialog, shared by Connect GPU, Metrics and Forecast.
   Opening makes the page behind inert and moves focus in; Tab wraps inside; Escape or a click on the backdrop closes;
   closing hands focus back to whatever opened it. Only one sheet is open at a time. */
const BEHIND = 'header.top, main, footer.foot, .fab';
let current = null;

export function makeSheet(root, { onClose, onOpen } = {}) {
  let opener = null;
  const focusables = () => [...root.querySelectorAll('button, a[href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])')]
    .filter((n) => !n.disabled && n.offsetParent !== null);
  const api = {
    get isOpen() { return !root.hidden; },
    open() {
      if (current && current !== api) current.close();
      opener = document.activeElement;
      document.querySelectorAll(BEHIND).forEach((n) => { n.inert = true; });
      root.hidden = false; current = api;
      (root.querySelector('[data-autofocus]') || focusables()[0])?.focus({ preventScroll: true });
      onOpen?.();
    },
    close() {
      if (root.hidden) return;
      root.hidden = true; current = null;
      document.querySelectorAll(BEHIND).forEach((n) => { n.inert = false; });
      if (opener && document.contains(opener) && !opener.closest('[hidden]')) opener.focus({ preventScroll: true });
      onClose?.();
    },
  };
  root.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); api.close(); return; }
    if (e.key !== 'Tab') return;
    const f = focusables(); if (!f.length) return;
    if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
    else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
  });
  root.addEventListener('mousedown', (e) => { if (e.target === root) api.close(); });
  root.addEventListener('click', (e) => { if (e.target.closest('[data-close]')) api.close(); });
  return api;
}
