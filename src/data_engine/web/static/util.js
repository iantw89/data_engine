// Small helpers shared by the dashboard and the task runner. No framework: elements are built with h().

export function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style' && typeof value === 'object') { for (const [k, v] of Object.entries(value)) k.startsWith('--') ? node.style.setProperty(k, v) : (node.style[k] = v); }
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else if (value === true) node.setAttribute(key, '');
    else node.setAttribute(key, value);
  }
  const add = (child) => {
    if (Array.isArray(child)) child.forEach(add);
    else if (child === null || child === undefined || child === false) return;
    else node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  };
  children.forEach(add);
  return node;
}

// replaceChildren that flattens arrays and drops null/false (plain replaceChildren stringifies them).
export function setChildren(node, ...kids) {
  const flat = kids.flat(Infinity).filter((k) => k !== null && k !== undefined && k !== false);
  node.replaceChildren(...flat);
}

export function svg(tag, attrs, ...children) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  children.flat().forEach((c) => c !== null && c !== undefined && node.append(c instanceof Node ? c : document.createTextNode(String(c))));
  return node;
}

export class ApiError extends Error {
  constructor(status, detail) { super(detail); this.status = status; }
}

export async function api(path, options = {}) {
  const init = { method: options.method || 'GET', headers: {} };
  if (options.body !== undefined) {
    init.body = JSON.stringify(options.body);
    init.headers['Content-Type'] = 'application/json';
  }
  const response = await fetch(path, init);
  let data = null;
  try { data = await response.json(); } catch (_) { /* empty body */ }
  if (!response.ok) {
    let detail = data && data.detail ? data.detail : response.statusText;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
    throw new ApiError(response.status, detail);
  }
  return data;
}

export function panelUrl(datasetId, panel, { w = 720, full = false } = {}) {
  return `/api/datasets/${encodeURIComponent(datasetId)}/panel?spec=${encodeURIComponent(JSON.stringify(panel))}&full=${full ? 1 : 0}&w=${w}`;
}

// ---- formatting
export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return '-';
  seconds = Math.round(seconds);
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`;
  return `${Math.floor(seconds / 3600)}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, '0')}m`;
}

export function fmtTime(epoch) {
  if (!epoch) return '-';
  const d = new Date(epoch * 1000);
  const sameDay = new Date().toDateString() === d.toDateString();
  const t = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  return sameDay ? t : `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${t}`;
}

export function fmtAgo(epoch) {
  if (!epoch) return '-';
  const s = Date.now() / 1000 - epoch;
  if (s < 5) return 'just now';
  return `${fmtDuration(s)} ago`;
}

export function fmtNumber(value, key = '') {
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString();
    if (value >= 0 && value <= 1 && /cover|frac|ratio|rate|share|percent|recall|precision/i.test(key)) return `${(value * 100).toFixed(1)}%`;
    return Math.abs(value) >= 100 ? value.toFixed(0) : Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  }
  return JSON.stringify(value);
}

export function prettyKey(key) { return String(key).replace(/_/g, ' '); }

// ---- reviewer name, toasts
export function reviewer() {
  try { return (localStorage.getItem('de.reviewer') || '').trim() || 'reviewer'; } catch (_) { return 'reviewer'; }
}

export function setupReviewerInput() {
  const input = document.getElementById('reviewer');
  try { input.value = localStorage.getItem('de.reviewer') || ''; } catch (_) { /* private mode */ }
  input.placeholder = 'your name';
  input.addEventListener('input', () => {
    try { localStorage.setItem('de.reviewer', input.value.trim()); } catch (_) { /* ignore */ }
  });
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === 'Escape') input.blur(); });
}

export function toast(message, kind = 'info', ms = 2600) {
  const box = document.getElementById('toasts');
  const node = h('div', { class: `toast ${kind}` }, message);
  box.append(node);
  setTimeout(() => node.remove(), kind === 'error' ? Math.max(ms, 6000) : ms);
  while (box.children.length > 4) box.firstChild.remove();
}

export function setCrumbs(parts) {
  const nav = document.getElementById('crumbs');
  setChildren(nav, parts.flatMap((p, i) => [
    i ? h('span', { class: 'sep', 'aria-hidden': 'true' }, '/') : null,
    p.href ? h('a', { href: p.href }, p.text) : h('span', { class: 'here', 'aria-current': 'page' }, p.text),
  ]));
}

export const PHASE_LABELS = {
  new: 'New', machine_pending: 'Machine work pending', running: 'Running', blocked: 'Blocked',
  answers_pending: 'Answers waiting for next round', waiting_for_human: 'Waiting for humans',
  ready_to_publish: 'Ready to publish', published: 'Published', error: 'Error',
};
export const PHASE_ACTOR = {
  new: 'operator', machine_pending: 'machine', running: 'machine', blocked: 'operator',
  answers_pending: 'machine', waiting_for_human: 'human', ready_to_publish: 'operator', published: 'none', error: 'operator',
};
