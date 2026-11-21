// Task runner: one task at a time, keyboard first, images load progressively.
import { h, setChildren, api, ApiError, panelUrl, panelOverlayUrl, toast, setCrumbs, reviewer, fmtDuration } from './util.js';

const PREF = {
  get(key, fallback) { try { const v = localStorage.getItem(`de.${key}`); return v === null ? fallback : JSON.parse(v); } catch (_) { return fallback; } },
  set(key, value) { try { localStorage.setItem(`de.${key}`, JSON.stringify(value)); } catch (_) { /* ignore */ } },
};

// Image sizes: a moderate first request, then a sharper one; sized by how many images share the screen.
function plan(task) {
  const p = task.payload;
  switch (task.ui) {
    case 'strip': { const n = (p.panels || []).length; return n <= 4 ? { lo: 480, hi: 900 } : n <= 8 ? { lo: 360, hi: 640 } : { lo: 300, hi: 520 }; }
    case 'decision': { const n = (p.panels || []).length; return n <= 1 ? { lo: 640, hi: 1100 } : n <= 2 ? { lo: 560, hi: 900 } : { lo: 420, hi: 700 }; }
    case 'reference_candidates': { const n = (p.candidates || []).length; return n <= 2 ? { lo: 520, hi: 880 } : { lo: 400, hi: 660 }; }
    case 'frame_picker': return { lo: 620, hi: 1000, thumb: 220 };
    default: return { lo: 480, hi: 800 };
  }
}

function allPanels(task) {
  const p = task.payload, out = [];
  (p.panels || []).forEach((x) => out.push(x));
  if (p.reference) (p.reference.panels || []).forEach((x) => out.push(x));
  (p.candidates || []).forEach((c) => (c.panels || []).forEach((x) => out.push(x)));
  return out;
}

export async function renderRunner(root, id) {
  const q = encodeURIComponent(id);
  const params = new URLSearchParams(location.search);
  const ctx = {
    id, q, type: params.get('type') || '', context: false, lens: PREF.get('lens', true),
    current: null, view: null, history: [], answered: 0, skipped: 0, remaining: null, byType: {},
    prefetch: null, busy: false, figures: new Set(), taxonomy: {}, readOnly: false,
  };
  setCrumbs([{ text: 'Datasets', href: '/' }, { text: id, href: `/d/${q}` }, { text: 'Review tasks' }]);

  try {
    const tax = await api('/api/taxonomy');
    ctx.taxonomy = Object.fromEntries(tax.task_types.map((t) => [t.name, t]));
    ctx.answerModel = tax.answer_model || 1;  // what this server understands; older servers drop image flags
  } catch (_) { /* titles fall back to ids */ }

  // ---------------------------------------------------------------- frame
  const typeSelect = h('select', { id: 'type-filter', 'aria-label': 'Filter by task type', onchange: () => { ctx.type = typeSelect.value; syncUrl(); ctx.prefetch = null; loadNext(null); } });
  const chips = h('div', { class: 'chips', 'aria-live': 'polite' });
  const stage = h('div', { class: 'task-stage' });
  const keybar = h('div', { class: 'keybar', 'aria-label': 'Keyboard shortcuts' });
  const bar = h('div', { class: 'runner-bar' },
    h('label', { class: 'type-filter' }, h('span', {}, 'Task type'), typeSelect), chips, h('div', { class: 'grow' }),
    h('button', { type: 'button', class: 'btn small', id: 'btn-context', 'aria-pressed': 'false', title: 'Show the whole frame with the box (C)', onclick: toggleContext }, 'Room view (C)'),
    h('button', { type: 'button', class: 'btn small', id: 'btn-lens', 'aria-pressed': String(ctx.lens), title: 'Hover magnifier (L)', onclick: toggleLens }, 'Magnifier (L)'),
    h('button', { type: 'button', class: 'btn small', onclick: () => toggleHelp(), 'aria-label': 'Show keyboard help', title: 'Keyboard help (?)' }, '? Keys'));
  root.replaceChildren(h('div', { class: 'runner' }, bar, stage, keybar));
  document.body.classList.add('runner-page');
  // Buttons must not keep focus: Enter submits the task, it must not re-click the last option.
  root.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setTimeout(() => b.blur(), 0); });

  function syncUrl() {
    const u = new URL(location.href);
    if (ctx.type) u.searchParams.set('type', ctx.type); else u.searchParams.delete('type');
    u.searchParams.delete('task');
    history.replaceState(null, '', u);
  }

  function renderChips() {
    const total = ctx.remaining;
    setChildren(chips,
      h('span', { class: 'chip ok', title: 'Answered in this browser session' }, `answered ${ctx.answered}`),
      ctx.skipped ? h('span', { class: 'chip', title: 'Skipped in this session' }, `skipped ${ctx.skipped}`) : null,
      h('span', { class: 'chip hot', title: 'Open tasks remaining' }, `remaining ${total === null ? '-' : total}`),
      Object.entries(ctx.byType).filter(([, n]) => n).map(([t, n]) => h('button', {
        type: 'button', class: `chip link ${ctx.type === t ? 'active' : ''}`, title: `Only ${titleOf(t)} tasks`,
        onclick: () => { ctx.type = ctx.type === t ? '' : t; typeSelect.value = ctx.type; syncUrl(); ctx.prefetch = null; loadNext(null); },
      }, `${titleOf(t)} ${n}`)));
    const known = Object.keys(ctx.byType).filter((t) => ctx.byType[t]);
    if (ctx.type && !known.includes(ctx.type)) known.push(ctx.type);
    setChildren(typeSelect,h('option', { value: '' }, 'All types'), known.map((t) => h('option', { value: t }, `${titleOf(t)} (${ctx.byType[t] || 0})`)));
    typeSelect.value = ctx.type;
  }
  const titleOf = (t) => (ctx.taxonomy[t] ? ctx.taxonomy[t].title : t);

  // ---------------------------------------------------------------- data flow
  function takeCounts(result) {
    if (!result) return;
    if (result.remaining !== undefined) ctx.remaining = result.remaining_by_type && !ctx.type ? Object.values(result.remaining_by_type).reduce((a, b) => a + b, 0) : result.remaining;
    if (result.remaining_by_type) ctx.byType = result.remaining_by_type;
    renderChips();
  }

  async function fetchNext(after) {
    const qs = new URLSearchParams();
    if (ctx.type) qs.set('type', ctx.type);
    if (after !== null && after !== undefined) qs.set('after', after);
    return api(`/api/datasets/${q}/tasks/next?${qs}`);
  }

  async function loadNext(after, { usePrefetch = false } = {}) {
    let result = null;
    if (usePrefetch && ctx.prefetch && ctx.prefetch.after === after && ctx.prefetch.result.task) {
      result = ctx.prefetch.result;
      fetchNext(after).then(takeCounts).catch(() => {});
    } else {
      stage.classList.add('busy');
      try { result = await fetchNext(after); } catch (e) { stage.classList.remove('busy'); return toast(e.message, 'error'); }
      takeCounts(result);
    }
    ctx.prefetch = null;
    stage.classList.remove('busy');
    if (!result.task) return showEmpty();
    show(result.task);
  }

  async function loadTask(taskId, restoreAnswer) {
    try {
      const task = await api(`/api/datasets/${q}/tasks/${taskId}`);
      show(task, restoreAnswer);
    } catch (e) { toast(e.message, 'error'); }
  }

  function showEmpty() {
    ctx.current = null; ctx.view = null; ctx.figures.clear();
    keybar.replaceChildren(h('span', {}, 'Backspace undo last'));
    stage.replaceChildren(h('div', { class: 'panel empty' },
      h('h2', {}, ctx.type ? `No open ${titleOf(ctx.type)} tasks` : 'No open tasks'),
      h('p', { class: 'muted' }, `You answered ${ctx.answered} in this session. Answers wait for the next round, which turns them into constraints and may create new tasks.`),
      h('p', {}, h('a', { class: 'btn primary', href: `/d/${q}` }, 'Back to dashboard'),
        ctx.type ? h('button', { class: 'btn', type: 'button', onclick: () => { ctx.type = ''; syncUrl(); loadNext(null); } }, 'Show all types') : null)));
  }

  function preloadNext(task) {
    fetchNext(task.id).then((result) => {
      if (ctx.current && ctx.current.id === task.id) {
        ctx.prefetch = { after: task.id, result };
        if (result.task) {
          const sizes = plan(result.task);
          allPanels(result.task).slice(0, 24).forEach((panel) => { new Image().src = panelUrl(ctx.id, panel, { w: result.task.ui === 'frame_picker' ? sizes.lo : sizes.lo }); });
        }
      }
    }).catch(() => {});
  }

  // ---------------------------------------------------------------- show a task
  function show(task, restoreAnswer) {
    ctx.current = task; ctx.figures.clear(); closeLightbox();
    const builder = BUILDERS[task.ui] || buildUnknown;
    const view = builder(ctx, task, plan(task));
    ctx.view = view;
    ctx.readOnly = !['open', 'answered'].includes(task.status);
    const notes = h('textarea', { id: 'notes', rows: 1, placeholder: 'Notes (optional) - press / to type', 'aria-label': 'Notes', maxlength: 2000 });
    view.notes = notes;
    const submitBtn = h('button', { type: 'button', class: 'btn primary big', id: 'btn-submit', onclick: submit, disabled: ctx.readOnly, title: 'Submit (Enter)' }, 'Submit ', h('kbd', {}, 'Enter'));
    const skipBtn = h('button', { type: 'button', class: 'btn', id: 'btn-skip', onclick: skip, disabled: task.status !== 'open', title: 'Skip (S)' }, 'Skip ', h('kbd', {}, 'S'));
    const backBtn = h('button', { type: 'button', class: 'btn', id: 'btn-back', onclick: undo, disabled: !ctx.history.length, title: 'Reopen the previous task (Backspace)' }, 'Back ', h('kbd', {}, 'Bksp'));
    const message = h('div', { class: 'msg-line', id: 'msg-line', role: 'status' });
    view.message = message;
    const t = task.payload;
    const info = h('header', { class: 'task-info' },
      h('div', { class: 'task-title-row' },
        h('h1', {}, t.title || titleOf(task.type)),
        h('span', { class: 'chip' }, `#${task.id}`), h('span', { class: 'chip' }, titleOf(task.type)),
        task.status !== 'open' ? h('span', { class: `chip st-${task.status}` }, task.status) : null,
        t.time_seconds !== null && t.time_seconds !== undefined ? h('span', { class: 'chip', title: 'Recording time' }, `t=${fmtDuration(t.time_seconds)}`) : null,
        h('span', { class: 'chip', title: 'Priority' }, `prio ${Number(task.priority).toFixed(1)}`)),
      h('p', { class: 'question' }, t.question || (ctx.taxonomy[task.type] || {}).question || ''),
      t.why ? h('p', { class: 'why' }, h('strong', {}, 'Why the machine could not decide: '), t.why) : null,
      t.evidence && Object.keys(t.evidence).length ? h('dl', { class: 'evidence' }, Object.entries(t.evidence).map(([k, v]) => h('div', {}, h('dt', {}, k.replace(/_/g, ' ')), h('dd', {}, typeof v === 'number' ? (Number.isInteger(v) ? v : +v.toFixed(3)) : String(v))))) : null,
      task.status !== 'open' ? h('p', { class: 'warn-text' }, ctx.readOnly ? `This task is ${task.status}; its answer is already part of a round and cannot be changed here.` : `Already answered by ${task.answered_by || 'someone'}; submitting replaces that answer.`) : null);
    stage.replaceChildren(h('article', { class: `task ui-${task.ui}`, dataset: { taskId: task.id, ui: task.ui } },
      info, view.el,
      h('footer', { class: 'actions' }, view.outcome || null, message, notes, h('div', { class: 'action-buttons' }, backBtn, skipBtn, submitBtn))));
    setChildren(keybar, [...(view.hints || []), ['Enter', 'submit'], ['S', 'skip'], ['Bksp', 'undo previous'], ['C', 'room view'], ['/', 'notes'], ['?', 'help']]
      .map(([k, l]) => h('span', { class: 'hint' }, h('kbd', {}, k), ' ', l)));
    const answer = restoreAnswer || (task.status === 'answered' ? task.answer : null);
    if (answer && view.restore) { view.restore(answer); if (answer.notes) notes.value = answer.notes; }
    else if (!answer && task.status === 'open' && view.prefill && task.known_flags) view.prefill(task.known_flags);
    window.scrollTo(0, 0);
    document.title = `#${task.id} ${t.title || ''} - Data Engine`;
    if (task.status === 'open' || task.status === 'answered') preloadNext(task);
  }

  function showMessage(text, kind = 'error') {
    const m = ctx.view && ctx.view.message;
    if (!m) return toast(text, kind);
    m.textContent = text; m.className = `msg-line ${kind}`;
    m.classList.remove('shake'); void m.offsetWidth; m.classList.add('shake');
  }

  // ---------------------------------------------------------------- actions
  async function submit() {
    if (!ctx.current || ctx.busy || ctx.readOnly) return;
    const problem = ctx.view.check ? ctx.view.check() : null;
    if (problem) return showMessage(problem);
    const answer = ctx.view.answer();
    answer.notes = ctx.view.notes.value.trim();
    const task = ctx.current;
    ctx.busy = true;
    try {
      await api(`/api/datasets/${q}/tasks/${task.id}/answer`, { method: 'POST', body: { answer, user: reviewer() } });
      ctx.answered++; ctx.history.push({ id: task.id, kind: 'answered' });
      toast(`#${task.id} saved`, 'ok', 1200);
      ctx.busy = false;
      await loadNext(task.id, { usePrefetch: true });
    } catch (e) {
      ctx.busy = false;
      if (e instanceof ApiError && e.status === 409) { toast(e.message, 'error'); await loadNext(task.id); }
      else showMessage(e.message || String(e));
    }
  }

  async function skip() {
    if (!ctx.current || ctx.busy || ctx.current.status !== 'open') return;
    const task = ctx.current;
    ctx.busy = true;
    try {
      await api(`/api/datasets/${q}/tasks/${task.id}/skip`, { method: 'POST' });
      ctx.skipped++; ctx.history.push({ id: task.id, kind: 'skipped' });
      toast(`#${task.id} skipped`, 'info', 1000);
      ctx.busy = false;
      await loadNext(task.id, { usePrefetch: true });
    } catch (e) { ctx.busy = false; toast(e.message, 'error'); }
  }

  async function undo() {
    if (ctx.busy) return;
    const last = ctx.history[ctx.history.length - 1];
    if (!last) return toast('Nothing to undo in this session', 'info');
    ctx.busy = true;
    try {
      const before = await api(`/api/datasets/${q}/tasks/${last.id}`);
      await api(`/api/datasets/${q}/tasks/${last.id}/reopen`, { method: 'POST', body: { user: reviewer() } });
      ctx.history.pop();
      if (last.kind === 'answered') ctx.answered = Math.max(0, ctx.answered - 1); else ctx.skipped = Math.max(0, ctx.skipped - 1);
      ctx.busy = false;
      const reopened = await api(`/api/datasets/${q}/tasks/${last.id}`);
      show(reopened, before.answer);
      toast(`#${last.id} reopened`, 'info', 1200);
      fetchNext(null).then(takeCounts).catch(() => {});
    } catch (e) { ctx.busy = false; toast(e.message, 'error'); }
  }

  function toggleContext() {
    ctx.context = !ctx.context;
    document.getElementById('btn-context').setAttribute('aria-pressed', String(ctx.context));
    document.getElementById('btn-context').classList.toggle('active', ctx.context);
    ctx.figures.forEach((fig) => fig.setFull(ctx.context));
  }
  function toggleLens() {
    ctx.lens = !ctx.lens; PREF.set('lens', ctx.lens);
    document.getElementById('btn-lens').setAttribute('aria-pressed', String(ctx.lens));
    document.getElementById('btn-lens').classList.toggle('active', ctx.lens);
    hideLens();
  }
  document.getElementById('btn-lens').classList.toggle('active', ctx.lens);

  // ---------------------------------------------------------------- help
  let helpNode = null;
  function toggleHelp(force) {
    if (helpNode) { helpNode.remove(); helpNode = null; if (!force) return; }
    if (force === false) return;
    const rows = [...(ctx.view && ctx.view.help ? ctx.view.help : []), ['Enter', 'Submit the answer'], ['S', 'Skip this task'], ['Backspace / Z', 'Reopen the previous task (answer prefilled)'],
      ['C', 'Toggle room view (whole frame with the box) for all images'], ['L', 'Toggle hover magnifier'], ['/', 'Focus notes (Esc leaves)'], ['?', 'This help']];
    helpNode = h('div', { class: 'overlay', role: 'dialog', 'aria-label': 'Keyboard help', onclick: () => toggleHelp() },
      h('div', { class: 'help-card', onclick: (e) => e.stopPropagation() }, h('h2', {}, 'Keyboard'),
        h('table', {}, rows.map(([k, d]) => h('tr', {}, h('th', {}, h('kbd', {}, k)), h('td', {}, d)))),
        h('p', { class: 'muted' }, '"Unsure" is always a valid, recorded answer and is never penalized. Prefer it over guessing.'),
        h('button', { class: 'btn', type: 'button', onclick: () => toggleHelp() }, 'Close (Esc)')));
    document.body.append(helpNode);
  }

  // ---------------------------------------------------------------- keyboard
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); } return; }
    const tag = (e.target.tagName || '').toLowerCase();
    const typing = tag === 'input' || tag === 'textarea' || tag === 'select';
    if (e.key === 'Escape') {
      if (closeFlagMenu) closeFlagMenu();
      if (lightbox) return closeLightbox();
      if (helpNode) return toggleHelp();
      if (typing) e.target.blur();
      return;
    }
    if (typing || lightbox) return;
    if (helpNode && e.key !== '?') return;
    if (ctx.view && ctx.view.onKey && ctx.view.onKey(e)) { e.preventDefault(); return; }
    const key = e.key.toLowerCase();
    if (e.key === 'Enter') { if (tag === 'a') return; e.preventDefault(); submit(); }
    else if (key === 's') { e.preventDefault(); skip(); }
    else if (e.key === 'Backspace' || key === 'z') { e.preventDefault(); undo(); }
    else if (key === 'c') { e.preventDefault(); toggleContext(); }
    else if (key === 'l') { e.preventDefault(); toggleLens(); }
    else if (key === '/') { e.preventDefault(); if (ctx.view) ctx.view.notes.focus(); }
    else if (e.key === '?') { e.preventDefault(); toggleHelp(); }
  });

  // ---------------------------------------------------------------- go
  try {
    const first = await fetchNext(null);
    takeCounts(first);
    const wanted = params.get('task');
    if (wanted) await loadTask(Number(wanted));
    else if (first.task) { show(first.task); }
    else showEmpty();
  } catch (e) { stage.replaceChildren(h('div', { class: 'panel error-box', role: 'alert' }, `Cannot load tasks: ${e.message}`)); }
}

// ====================================================================== images
let lightbox = null, lensNode = null;

function hideLens() { if (lensNode) { lensNode.remove(); lensNode = null; } }

function closeLightbox() { if (lightbox) { lightbox.remove(); lightbox = null; } }

// The lightbox has no magnifier, so in both crop and room view the skeleton hides while the cursor is over it.
function openLightbox(ctx, panel, full) {
  closeLightbox();
  let showFull = full;
  const img = h('img', { alt: 'Enlarged evidence image' });
  const layer = skeletonLayer(img);
  const stage = h('div', { class: 'lightbox-stage' }, img, layer.svg);
  const set = () => {
    const opts = { w: showFull ? 2400 : 1800, full: showFull };
    layer.load(panelOverlayUrl(ctx.id, panel, opts));
    img.onload = () => layer.ready();
    img.src = panelUrl(ctx.id, panel, opts);
    toggle.textContent = showFull ? 'Show crop' : 'Show room';
  };
  const toggle = h('button', { type: 'button', class: 'btn small', onclick: (e) => { e.stopPropagation(); showFull = !showFull; set(); } });
  stage.addEventListener('mousemove', (e) => layer.setHidden(layer.over(e.clientX, e.clientY)));
  stage.addEventListener('mouseleave', () => layer.setHidden(false));
  lightbox = h('div', { class: 'overlay lightbox', role: 'dialog', 'aria-label': 'Enlarged image', onclick: closeLightbox },
    h('div', { class: 'lightbox-bar' }, h('span', {}, `${panel.camera}  frame ${panel.frame} ${panel.label || ''}`), toggle,
      h('button', { type: 'button', class: 'btn small', onclick: closeLightbox }, 'Close (Esc)')), stage);
  set();
  document.body.append(lightbox);
}

// ---- skeleton overlay: panel images are skeleton-free and the skeletons are SVG, so hovering can hide them.
const SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs) { const n = document.createElementNS(SVG_NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; }

// Same look as render.draw_skeleton: dark halo under a thin coloured line, dots at joints, dashed for projections.
function drawSkeletons(svg, overlay) {
  svg.replaceChildren();
  const [W, H] = overlay.size || [1, 1];
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const t = overlay.thickness || 1, r = overlay.radius || 2, color = overlay.color || '#5ae65a', halo = '#141414';
  for (const sk of overlay.skeletons || []) {
    const P = sk.joints.map((j) => (j ? [j[0] * W, j[1] * H] : null));
    const dash = sk.dashed ? { 'stroke-dasharray': '9 6' } : {};
    for (const [stroke, width] of [[halo, t + 2], [color, t]]) {
      for (const [a, b] of overlay.bones || []) {
        if (P[a] && P[b]) svg.append(svgEl('line', { x1: P[a][0], y1: P[a][1], x2: P[b][0], y2: P[b][1], stroke, 'stroke-width': width, ...dash }));
      }
    }
    for (const p of P) {
      if (!p) continue;
      if (sk.dashed) {
        svg.append(svgEl('circle', { cx: p[0], cy: p[1], r: r + 1, fill: 'none', stroke: halo, 'stroke-width': 1 }));
        svg.append(svgEl('circle', { cx: p[0], cy: p[1], r, fill: 'none', stroke: color, 'stroke-width': 1 }));
      } else {
        svg.append(svgEl('circle', { cx: p[0], cy: p[1], r: r + 1, fill: halo }));
        svg.append(svgEl('circle', { cx: p[0], cy: p[1], r, fill: color }));
      }
    }
  }
}

// Joint bounding boxes ([x0, y0, x1, y1] fractions); "over the skeleton" means inside one (plus a margin).
function skeletonBoxes(overlay) {
  return (overlay.skeletons || []).map((sk) => {
    const js = sk.joints.filter(Boolean);
    if (!js.length) return null;
    const xs = js.map((j) => j[0]), ys = js.map((j) => j[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }).filter(Boolean);
}

// An SVG skeleton layer kept exactly over ``img`` (same offset parent). load() fetches the geometry, ready()
// reveals it once the image is shown, over() says whether a client point is on a skeleton (box + margin).
function skeletonLayer(img) {
  const svg = svgEl('svg', { class: 'skel pending', 'aria-hidden': 'true', preserveAspectRatio: 'none' });
  const layer = { svg, overlay: { skeletons: [] }, boxes: [], hidden: false, token: 0 };
  layer.place = () => {
    svg.style.left = `${img.offsetLeft}px`; svg.style.top = `${img.offsetTop}px`;
    svg.style.width = `${img.offsetWidth}px`; svg.style.height = `${img.offsetHeight}px`;
  };
  if (window.ResizeObserver) new ResizeObserver(layer.place).observe(img);
  layer.load = (url) => {
    const t = ++layer.token;
    svg.classList.add('pending'); svg.replaceChildren(); layer.overlay = { skeletons: [] }; layer.boxes = []; layer.setHidden(false);
    fetch(url).then((r) => (r.ok ? r.json() : null)).then((data) => {
      if (t !== layer.token || !data) return;
      layer.overlay = data; layer.boxes = skeletonBoxes(data); drawSkeletons(svg, data);
    }).catch(() => { /* no overlay: the image alone is still useful */ });
    return t;
  };
  layer.ready = () => { layer.place(); svg.classList.remove('pending'); };
  layer.setHidden = (value) => { layer.hidden = value; svg.classList.toggle('hidden', value); };
  layer.over = (clientX, clientY, pad = 14) => {
    const r = img.getBoundingClientRect();
    const x = clientX - r.left, y = clientY - r.top;
    if (x < 0 || y < 0 || x > r.width || y > r.height) return false;
    return layer.boxes.some(([x0, y0, x1, y1]) => x >= x0 * r.width - pad && x <= x1 * r.width + pad && y >= y0 * r.height - pad && y <= y1 * r.height + pad);
  };
  return layer;
}

function panelFigure(ctx, panel, o = {}) {
  const lo = o.lo || 480, hi = o.hi || 800;
  let full = ctx.context, token = 0;
  const img = h('img', { alt: `${panel.camera} frame ${panel.frame}${panel.label ? ', ' + panel.label : ''}`, draggable: 'false', class: 'loading' });
  const layer = skeletonLayer(img), svg = layer.svg;
  const roomBtn = h('button', { type: 'button', class: 'fig-btn', title: 'Toggle room view for this image', 'aria-label': 'Toggle room view for this image', 'aria-pressed': String(full),
    onclick: (e) => { e.stopPropagation(); fig.setFull(!full); } }, '▣');
  const zoomBtn = h('button', { type: 'button', class: 'fig-btn zoom', title: 'Enlarge', 'aria-label': 'Enlarge image', onclick: (e) => { e.stopPropagation(); openLightbox(ctx, panel, full); } }, '⤢');
  const fig = h('figure', { class: `pfig ${o.cls || ''}` }, img, svg, o.noButtons ? null : [roomBtn, zoomBtn]);

  function load() {
    const t = ++token;
    const scale = full ? 1.5 : 1;
    const loW = Math.round((o.lo || lo) * scale);
    const loUrl = panelUrl(ctx.id, panel, { w: loW, full });
    const hiUrl = hi > lo ? panelUrl(ctx.id, panel, { w: Math.round(hi * scale), full }) : null;
    img.classList.add('loading'); img.classList.remove('failed'); img.dataset.res = 'lo';
    layer.load(panelOverlayUrl(ctx.id, panel, { w: loW, full }));
    const pre = new Image();
    pre.onload = () => { if (t === token) { img.dataset.res = 'hi'; img.src = hiUrl; } };
    img.onload = () => {
      img.classList.remove('loading'); layer.place();
      if (t === token) layer.ready();
      if (img.dataset.res === 'lo' && t === token && hiUrl) pre.src = hiUrl;
    };
    img.onerror = () => { img.classList.remove('loading'); img.classList.add('failed'); };
    img.src = loUrl;
  }
  fig.setFull = (value) => { if (value === full) return; full = value; roomBtn.setAttribute('aria-pressed', String(full)); fig.classList.toggle('room', full); load(); };
  if (full) fig.classList.add('room');

  fig.addEventListener('mousemove', (e) => {
    const r = img.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const inside = x >= 0 && y >= 0 && x <= r.width && y <= r.height && !e.target.closest('.fig-btn');
    // Room view: hide the skeleton only while the cursor is over it. Crops keep it; their magnifier is bare.
    if (full) layer.setHidden(inside && layer.over(e.clientX, e.clientY));
    if (o.noLens) return;
    if (!ctx.lens || !inside || !img.complete || img.classList.contains('loading') || img.classList.contains('failed')) return hideLens();
    const Z = 2.6, size = 230;
    if (!lensNode) { lensNode = h('div', { class: 'lens', 'aria-hidden': 'true' }); document.body.append(lensNode); }
    const zoomUrl = panelUrl(ctx.id, panel, { w: 1800, full });
    if (lensNode.dataset.src !== zoomUrl) { lensNode.dataset.src = zoomUrl; lensNode.style.backgroundImage = `url("${zoomUrl}")`; }
    const ox = -(x * Z - size / 2), oy = -(y * Z - size / 2);
    lensNode.style.backgroundSize = `${r.width * Z}px ${r.height * Z}px`;
    lensNode.style.backgroundPosition = `${ox}px ${oy}px`;
    // Room view: the magnifier mirrors the image (skeleton shown unless the cursor is over it).
    let lensSvg = lensNode.firstChild;
    const overlay = layer.overlay;
    if (full && !layer.hidden && overlay.skeletons && overlay.skeletons.length) {
      if (!lensSvg || lensSvg._overlay !== overlay) {
        lensSvg = svgEl('svg', { class: 'skel', preserveAspectRatio: 'none' }); lensSvg._overlay = overlay;
        drawSkeletons(lensSvg, overlay); lensNode.replaceChildren(lensSvg);
      }
      Object.assign(lensSvg.style, { left: `${ox}px`, top: `${oy}px`, width: `${r.width * Z}px`, height: `${r.height * Z}px` });
    } else if (lensSvg) lensNode.replaceChildren();
    const left = e.clientX + 28 + size > innerWidth ? e.clientX - 28 - size : e.clientX + 28;
    const top = Math.min(Math.max(8, e.clientY - size / 2), innerHeight - size - 8);
    lensNode.style.left = `${left}px`; lensNode.style.top = `${top}px`;
  });
  fig.addEventListener('mouseleave', () => { layer.setHidden(false); if (!o.noLens) hideLens(); });
  if (o.clickZoom !== false) fig.addEventListener('click', () => openLightbox(ctx, panel, full));
  ctx.figures.add(fig);
  load();
  return fig;
}

const panelRow = (ctx, panels, o) => h('div', { class: 'panel-row' }, (panels || []).map((p) => panelFigure(ctx, p, o)));
const kbd = (k) => h('kbd', {}, k);

// ====================================================================== UI builders
const BUILDERS = {
  strip: buildStrip, reference_candidates: buildReference, decision: buildDecision, frame_picker: buildPicker, cvat: buildCvat,
};

// ---------------------------------------------------------------- strip
// A time-ordered strip of crops of one single-camera tracklet. The answer marks every crop at which a
// different person starts (several are possible), and for a change after the first, optionally who it is: an
// earlier person again, or nobody seen before. Single crops can be taken out: unclear, not a person, two
// people in one pose. (validate.strip_segments mirrors segments() below.)
const STRIP_MARKS = { unclear: ['X', 'unclear', 'Unclear'], not_person: ['N', 'not a person', 'Not a person'], mixed: ['M', 'two people in one pose', 'Two people in one pose'] };
const PERSON_COLORS = ['', 'var(--diff)', '#e0b34a', '#6fb3ff', '#c58bff', '#5fd3c4', '#ff9d6f', '#9be06f'];

function buildStrip(ctx, task, sizes) {
  const panels = task.payload.panels || [];
  const n = panels.length;
  const s = { switches: new Set(), who: {}, marks: {}, unsureAll: false, mode: 'switch', cursor: Math.min(1, n - 1) };
  function segments() {
    const bounds = [0, ...[...s.switches].sort((a, b) => a - b), n];
    const out = [];
    let following = 2;
    for (let k = 0; k + 1 < bounds.length; k++) {
      const start = bounds[k];
      const earlier = [...new Set(out.map((g) => g.person))];
      const options = k >= 2 ? earlier.filter((person) => person !== out[k - 1].person) : [];
      const said = s.who[start];
      let person, how;
      if (k === 0) { person = 1; how = 'first'; }
      else if (typeof said === 'number' && options.includes(said)) { person = said; how = 'back'; }
      else { person = following++; how = said === 'new' && k >= 2 ? 'new' : 'switch'; }
      out.push({ start, stop: bounds[k + 1], person, said: how, options });
    }
    return out;
  }
  const cards = panels.map((panel, i) => {
    const badge = h('span', { class: 'num-badge' }, i + 1);
    const state = h('span', { class: 'crop-state' });
    const who = h('div', { class: 'crop-who', style: { display: 'flex', gap: '4px', justifyContent: 'center', flexWrap: 'wrap', padding: '0 2px 3px' }, onclick: (e) => e.stopPropagation() });
    const card = h('div', { class: 'crop', role: 'button', tabindex: '0', 'aria-label': `Crop ${i + 1}`,
      onclick: (e) => { if (e.target.closest('.fig-btn')) return; clickCrop(i, e.shiftKey); },
      onkeydown: (e) => { if (e.key === ' ') { e.preventDefault(); clickCrop(i, e.shiftKey); } } },
      panelFigure(ctx, { ...panel, label: panel.label || '' }, { lo: sizes.lo, hi: sizes.hi, clickZoom: false }), badge, state, who);
    card._state = state; card._who = who;
    return card;
  });
  const status = h('div', { class: 'strip-status', 'aria-live': 'polite' });
  const modes = [['switch', 'a different person starts here', 'B'], ...Object.entries(STRIP_MARKS).map(([kind, [key, , label]]) => [kind, label.toLowerCase(), key])];
  const modeBtns = modes.map(([mode, label, key]) => h('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', dataset: { mode },
    onclick: () => { s.mode = mode; refresh(); } }, label, ' ', kbd(key)));
  const modeRow = h('div', { class: 'seg', role: 'group', 'aria-label': 'What a click on a crop marks' }, h('span', { class: 'muted small', style: { alignSelf: 'center', marginRight: '6px' } }, 'A click marks:'), modeBtns);
  const resetBtn = h('button', { type: 'button', class: 'btn', onclick: () => { s.switches.clear(); s.who = {}; s.unsureAll = false; refresh(); } }, 'All the same person ', kbd('0'));
  const unsureBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', onclick: () => toggleUnsure() }, 'Unsure about this strip ', kbd('U'));

  function toggleUnsure() { s.unsureAll = !s.unsureAll; if (s.unsureAll) { s.switches.clear(); s.who = {}; } refresh(); }
  function toggleSwitch(i) {
    if (i === 0) { toast('Crop 1 is the reference; mark each crop at which a different person starts', 'info'); return; }
    if (s.switches.has(i)) { s.switches.delete(i); delete s.who[i]; } else s.switches.add(i);
    s.unsureAll = false;
  }
  function toggleMark(i, kind) { if (s.marks[i] === kind) delete s.marks[i]; else s.marks[i] = kind; }
  function clickCrop(i, shift) {
    s.cursor = i;
    if (shift) toggleMark(i, 'unclear'); else if (s.mode === 'switch') toggleSwitch(i); else toggleMark(i, s.mode);
    refresh();
  }
  // Who the person is that starts at a change after the first: cycles not said -> an earlier person -> new.
  function cycleWho(i) {
    const seg = segments().find((g) => i >= g.start && i < g.stop);
    if (!seg || !seg.options.length) { toast('Only a change after the first one can be an earlier person again', 'info', 1800); return; }
    const choices = [undefined, ...seg.options, 'new'];
    const next = choices[(choices.indexOf(s.who[seg.start]) + 1) % choices.length];
    if (next === undefined) delete s.who[seg.start]; else s.who[seg.start] = next;
  }
  const personText = (g) => `person ${g.person}${g.said === 'back' ? ' again' : g.said === 'new' ? ', not seen before' : ''}`;
  function refresh() {
    const segs = segments();
    cards.forEach((card, i) => {
      const seg = segs.find((g) => i >= g.start && i < g.stop);
      const mark = s.marks[i], starts = seg.start === i && seg.said !== 'first';
      const color = PERSON_COLORS[(seg.person - 1) % PERSON_COLORS.length];
      card.classList.toggle('unclear', !!mark); card.classList.toggle('cursor', s.cursor === i);
      card.style.borderColor = mark || s.cursor === i ? '' : color;
      card.style.boxShadow = starts ? `0 0 0 3px ${color || 'var(--line2)'}` : '';
      card._state.style.color = mark ? '' : color;
      card._state.textContent = mark ? STRIP_MARKS[mark][1] : !s.switches.size ? (i === 0 ? 'reference' : 'same') : starts ? `${personText(seg)} - starts here` : personText(seg);
      card.setAttribute('aria-pressed', String(s.switches.has(i)));
      const choices = starts && seg.options.length ? [...seg.options.map((person) => [person, `person ${person} again`]), ['new', 'not seen before'], [undefined, 'not said']] : [];
      setChildren(card._who, choices.map(([value, label]) => h('button', { type: 'button', class: `btn small ${s.who[i] === value ? 'active' : ''}`, 'aria-pressed': String(s.who[i] === value),
        onclick: (e) => { e.stopPropagation(); if (value === undefined) delete s.who[i]; else s.who[i] = value; refresh(); } }, label)));
    });
    modeBtns.forEach((b) => { const on = b.dataset.mode === s.mode; b.setAttribute('aria-pressed', String(on)); b.classList.toggle('active', on); });
    unsureBtn.setAttribute('aria-pressed', String(s.unsureAll)); unsureBtn.classList.toggle('active', s.unsureAll);
    const counts = {};
    Object.values(s.marks).forEach((kind) => { counts[kind] = (counts[kind] || 0) + 1; });
    const marked = Object.entries(counts).map(([kind, c]) => `${c} ${STRIP_MARKS[kind][1]}`).join(', ');
    const people = new Set(segs.map((g) => g.person)).size;
    const unsaid = segs.some((g) => g.options.length && g.said === 'switch');
    status.className = `strip-status ${s.unsureAll ? 'unsure' : s.switches.size ? 'break' : 'same'}`;
    status.textContent = (s.unsureAll ? 'Answer: unsure about this strip.'
      : !s.switches.size ? 'Answer: all crops show the same person.'
        : `Answer: ${s.switches.size} change${s.switches.size > 1 ? 's' : ''} of person, ${people} ${unsaid ? 'or fewer ' : ''}people - ` + segs.map((g) => `crop${g.stop - g.start > 1 ? `s ${g.start + 1}-${g.stop}` : ` ${g.start + 1}`}: ${personText(g)}`).join('; ') + '.'
          + (unsaid ? ' Where a later change is an earlier person again, say so on that crop (W); otherwise only the change itself is recorded.' : ''))
      + (marked ? ` Taken out: ${marked}.` : '');
  }
  const el = h('section', { class: 'strip-ui' },
    h('div', { class: 'strip-grid', style: { '--min': n <= 4 ? '260px' : n <= 8 ? '190px' : '140px' } }, cards),
    h('div', { class: 'strip-controls' }, status, resetBtn, unsureBtn),
    h('div', { class: 'strip-controls' }, modeRow));
  refresh();
  const indices = (kind) => Object.keys(s.marks).filter((i) => s.marks[i] === kind).map(Number).sort((a, b) => a - b);
  return {
    el,
    hints: [['2-9', 'a different person starts at that crop (several can)'], ['0', 'all same'], ['U', 'unsure'], ['←→', 'move'], ['B', 'change here'], ['W', 'who it is'], ['X N M', 'unclear / not a person / two people in one pose'], ['Shift+click', 'unclear']],
    help: [['2 ... 9', 'Mark (or unmark) that crop as the first of a different person; mark every change in the strip'], ['← → / B', 'Move the cursor / mark a change at the cursor'],
      ['W', 'For a change after the first: who starts there - an earlier person again, someone not seen before, or not said'],
      ['X / N / M', 'Take the cursor crop out: unclear / not a person / two people in one pose'], ['0', 'Reset: all the same person'], ['U', 'Unsure about the whole strip'], ['Shift+click', 'Toggle unclear on a crop']],
    onKey(e) {
      if (/^[0-9]$/.test(e.key) && !e.shiftKey) {
        if (e.key === '0') { s.switches.clear(); s.who = {}; s.unsureAll = false; refresh(); return true; }
        const i = Number(e.key) - 1;
        if (i < n) { toggleSwitch(i); s.cursor = i; refresh(); }
        return true;
      }
      const k = e.key.toLowerCase();
      if (e.key === 'ArrowRight') { s.cursor = Math.min(n - 1, s.cursor + 1); refresh(); return true; }
      if (e.key === 'ArrowLeft') { s.cursor = Math.max(0, s.cursor - 1); refresh(); return true; }
      if (k === 'b') { toggleSwitch(s.cursor); refresh(); return true; }
      if (k === 'w') { cycleWho(s.cursor); refresh(); return true; }
      const kind = { x: 'unclear', n: 'not_person', m: 'mixed' }[k];
      if (kind) { toggleMark(s.cursor, kind); refresh(); return true; }
      if (k === 'u') { toggleUnsure(); return true; }
      return false;
    },
    check() {
      // A server from before several changes and the new marks existed would refuse them or lose them.
      const needs = s.switches.size > 1 || Object.values(s.marks).some((kind) => kind !== 'unclear');
      if (needs && (ctx.answerModel || 1) < 5) return 'This web server runs an older version of the engine than this page and would refuse or lose parts of the answer. Restart it (data-engine serve), reload this page, then answer.';
      return null;
    },
    answer() {
      const switches = [...s.switches].sort((a, b) => a - b);
      const who = {};
      segments().forEach((g) => { if (g.said === 'back') who[g.start] = g.person; else if (g.said === 'new') who[g.start] = 'new'; });
      return { verdict: switches.length ? 'break' : s.unsureAll ? 'unclear' : 'continuous', switches, first_different: switches.length ? switches[0] : null, who,
        unclear: indices('unclear'), not_person: indices('not_person'), mixed: indices('mixed') };
    },
    restore(a) {
      const switches = a.verdict !== 'break' ? [] : a.switches || (a.first_different !== null && a.first_different !== undefined ? [a.first_different] : []);
      s.switches = new Set(switches); s.who = {};
      for (const [i, value] of Object.entries(a.who || {})) s.who[Number(i)] = value;
      s.unsureAll = a.verdict === 'unclear'; s.marks = {};
      for (const kind of Object.keys(STRIP_MARKS)) (a[kind] || []).forEach((i) => { s.marks[i] = kind; });
      refresh();
    },
  };
}

// ---------------------------------------------------------------- groups of images: reference + candidates
// Any image can be flagged. Four flags take the image out of its group - the answer then applies to the
// images that remain - and 'bad pose' leaves it in and reports the keypoints (taxonomy.IMAGE_FLAGS).
// 'Unsure' records nothing about the image: too little is visible to tell whether it belongs.
const IMAGE_FLAGS = [
  ['other_person', 'X', 'Someone else', 'not the person the rest of this group shows'],
  ['mixed', 'M', 'Two people in one pose', 'the skeleton takes joints from two people'],
  ['not_person', 'N', 'Not a person', 'reflection, poster, equipment'],
  ['unsure', 'Q', 'Unsure', 'too little visible to tell - nothing is recorded about it'],
];
const FLAG_KEYS = { x: 'other_person', m: 'mixed', n: 'not_person', q: 'unsure' };
const FLAG_TEXT = { other_person: 'someone else', mixed: 'two people in one pose', not_person: 'not a person', unsure: 'unsure' };
let closeFlagMenu = null;
document.addEventListener('click', () => { if (closeFlagMenu) closeFlagMenu(); });

function buildReference(ctx, task, sizes) {
  const p = task.payload;
  const candidates = p.candidates || [];
  const ref = p.reference || null;
  const refId = ref ? String(ref.id || 'A') : 'A';
  // What is asked: 'match' (is each candidate the reference's person), 'person' (is the reference a real
  // person), 'duplicate' (two poses: one person or two). Older payloads do not say; the task type does.
  const mode = p.mode || (task.type === 'detection_triage' || !candidates.length ? 'person' : task.type === 'pose_ownership' ? 'duplicate' : 'match');
  // A swap check always takes several: both candidates can show the reference's person (two tracks on one person).
  const isSwap = task.type === 'swap_check' || candidates.some((c) => c.continuation);
  const multiple = isSwap || (p.multiple !== undefined ? !!p.multiple : task.type === 'reidentify');
  const cannotBoth = p.cannot_both || [];
  const texts = p.texts || {};
  const keyOf = (panel) => (panel.highlight && panel.highlight[0]) || null;
  const groups = {};
  if (ref) groups[refId] = (ref.panels || []).map(keyOf).filter(Boolean);
  candidates.forEach((c) => { groups[String(c.id)] = (c.panels || []).map(keyOf).filter(Boolean); });
  // Images shown for comparison (a swap check's "other person before") are a track too: they can be flagged.
  const CONTEXT = 'context';
  const contextKeys = (p.context_panels || []).map(keyOf).filter(Boolean);
  if (contextKeys.length) groups[CONTEXT] = contextKeys;
  const ids = candidates.map((c) => String(c.id));
  const s = { images: {}, values: Object.fromEntries(ids.map((id) => [id, null])), verdict: null, better: null, otherBefore: null };
  const flagOf = (key) => (s.images[key] || {}).exclude || null;
  const kept = (gid) => (groups[gid] || []).filter((key) => !flagOf(key));
  // No image of the group is left to judge and at least one was left out as unsure: that group counts as "unsure".
  const unsureOnly = (gid) => !kept(gid).length && (groups[gid] || []).some((key) => flagOf(key) === 'unsure');
  // Swap check: the track that should continue the reference (stay), the other track (other), and the other
  // track just before the moment - asked about only when the reference is both candidates.
  const stay = isSwap ? candidates.find((c) => c.continuation) || null : null;
  const other = isSwap ? candidates.find((c) => !c.continuation) || null : null;
  const beforeKeys = ((p.other_before || {}).keys || (p.context_panels || []).map(keyOf).filter(Boolean));
  const otherBefore = isSwap && other && beforeKeys.length ? { track: (p.other_before || {}).track ?? other.track } : null;
  const bothSame = () => isSwap && ids.filter((id) => kept(id).length && kept(refId).length && s.values[id] === 'same').length > 1;
  // The follow-up about the other track before the moment needs an image of it left to judge.
  const beforeAlive = () => beforeKeys.some((key) => !flagOf(key));
  const refAlive = () => !ref || kept(refId).length > 0;
  const alive = (id) => refAlive() && kept(id).length > 0;
  const conflict = () => cannotBoth.find((pair) => pair.ids.every((id) => alive(String(id)) && s.values[String(id)] === 'same')) || null;

  // ---- images and their flags
  const tiles = [];
  const carried = {};  // flags pre-set from the person's answers to other tasks that show the same image
  const isCarried = (key) => { const c = carried[key], f = s.images[key] || {}; return !!c && (f.exclude || null) === c.exclude && !!f.bad_pose === c.bad_pose; };
  let menu = null, hovered = null;
  const closeMenu = () => { if (menu) { menu.remove(); menu = null; } };
  closeFlagMenu = closeMenu;
  const entry = (key) => (s.images[key] = s.images[key] || { exclude: null, bad_pose: false });
  function setFlag(key, flag) {
    const e = entry(key);
    e.exclude = e.exclude === flag ? null : flag;
    if (e.exclude === 'mixed' || e.exclude === 'not_person') e.bad_pose = false;
    refresh();
  }
  function toggleBad(key) {
    const e = entry(key);
    if (e.exclude === 'mixed' || e.exclude === 'not_person') return;
    e.bad_pose = !e.bad_pose;
    refresh();
  }
  function openMenu(key, gid, wrap) {
    const again = menu && menu.dataset.key === key;
    closeMenu();
    if (again) return;
    const cur = s.images[key] || {};
    const alone = (groups[gid] || []).length < 2;
    const item = (label, hint, on, act, k) => h('button', { type: 'button', class: `flag-item ${on ? 'on' : ''}`, role: 'menuitemcheckbox', 'aria-checked': String(!!on),
      onclick: (e) => { e.stopPropagation(); act(); closeMenu(); } }, h('span', { class: 'flag-label' }, label), h('span', { class: 'flag-hint' }, hint), h('kbd', {}, k));
    menu = h('div', { class: 'flag-menu', role: 'menu', dataset: { key }, onclick: (e) => e.stopPropagation() },
      h('div', { class: 'flag-head' }, 'Take this image out of its group'),
      IMAGE_FLAGS.filter(([flag]) => !(flag === 'other_person' && alone)).map(([flag, k, label, hint]) => item(label, hint, cur.exclude === flag, () => setFlag(key, flag), k)),
      h('div', { class: 'flag-head' }, 'Or keep it, and report'),
      item('Bad pose', 'right person, keypoints wrong', !!cur.bad_pose, () => toggleBad(key), 'P'));
    wrap.append(menu);
  }
  function tile(panel, gid) {
    const key = keyOf(panel);
    const fig = panelFigure(ctx, panel, { lo: sizes.lo, hi: sizes.hi });
    if (!key) return fig;
    const state = h('div', { class: 'img-state' });
    const flagBtn = h('button', { type: 'button', class: 'flag-btn', title: 'Flag this image: someone else (X), two people in one pose (M), not a person (N), unsure (Q), bad pose (P)', 'aria-label': 'Flag this image', 'aria-haspopup': 'menu',
      onclick: (e) => { e.stopPropagation(); openMenu(key, gid, wrap); } }, '⚑');
    const wrap = h('div', { class: 'img-tile', dataset: { key } }, fig, flagBtn, state);
    wrap.addEventListener('mouseenter', () => { hovered = { key, gid }; });
    wrap.addEventListener('mouseleave', () => { if (hovered && hovered.key === key) hovered = null; });
    tiles.push({ key, gid, wrap, state });
    return wrap;
  }
  const images = (panels, gid) => h('div', { class: 'panel-row' }, (panels || []).map((panel) => tile(panel, gid)));

  // ---- answers
  const LABEL = mode === 'duplicate' ? { same: 'One person', different: 'Two people', unsure: 'Unsure' } : { same: 'Same', different: 'Different', unsure: 'Unsure' };
  const CYCLE = [null, 'same', 'different', 'unsure'];
  const cardEls = candidates.map((c, idx) => {
    const id = String(c.id);
    const buttons = ['same', 'different', 'unsure'].map((v) => h('button', { type: 'button', class: `choice ${v}`, 'aria-pressed': 'false', dataset: { value: v },
      onclick: () => { s.values[id] = s.values[id] === v ? null : v; refresh(); } }, LABEL[v]));
    const note = h('span', { class: 'muted small' });
    const card = h('div', { class: 'cand', dataset: { cand: id } },
      h('div', { class: 'cand-head' }, h('span', { class: 'num-badge' }, idx + 1), h('strong', {}, `${id}`), h('span', { class: 'muted cap' }, c.caption || ''),
        c.track !== null && c.track !== undefined ? h('span', { class: 'chip small' }, `track ${c.track}`) : null),
      images(c.panels, id),
      h('div', { class: 'cand-controls' }, h('div', { class: 'seg', role: 'group', 'aria-label': `Is ${id} the same person as ${refId}?` }, buttons), note));
    card._buttons = buttons; card._note = note;
    return card;
  });
  const verdictBtns = mode !== 'person' ? [] : [['person', 'Real person', 'same'], ['not_person', 'Not a person', 'different'], ['unsure', 'Unsure', 'unsure']].map(([v, label, cls]) =>
    h('button', { type: 'button', class: `choice ${cls}`, 'aria-pressed': 'false', dataset: { value: v }, onclick: () => { s.verdict = s.verdict === v ? null : v; refresh(); } }, label));
  // Better, in order: the box holds the whole person and no one else; then, if both do, the better keypoints.
  const betterHint = '1. Whole person in the box, no one else · 2. If both, better keypoints · Keypoints still wrong? Also flag Bad pose (P)';
  const betterBtns = mode !== 'duplicate' || !ids.length ? [] : [[refId, `${refId} is better`], [ids[0], `${ids[0]} is better`], ['equal', 'No difference']].map(([v, label]) =>
    h('button', { type: 'button', class: 'choice better', 'aria-pressed': 'false', title: betterHint, dataset: { value: v }, onclick: () => { s.better = v; refresh(); } }, label));
  const betterRow = betterBtns.length ? h('div', { class: 'better-row' }, h('span', {}, 'They are one person. Which is better?'),
    h('div', { class: 'seg', role: 'group', 'aria-label': 'Which pose is better' }, betterBtns),
    h('div', { class: 'better-hint muted' }, betterHint)) : null;
  const beforeBtns = !otherBefore ? [] : [['same', 'Same person', 'same'], ['different', 'Someone else', 'different'], ['unsure', 'Unsure', 'unsure']].map(([v, label, cls]) =>
    h('button', { type: 'button', class: `choice ${cls}`, 'aria-pressed': 'false', dataset: { value: v }, onclick: () => { s.otherBefore = s.otherBefore === v ? null : v; refresh(); } }, label));
  const beforeRow = beforeBtns.length ? h('div', { class: 'better-row' }, h('span', {}, `${refId} is both candidates. Did track ${otherBefore.track} show this person before the moment too (these images)?`),
    h('div', { class: 'seg', role: 'group', 'aria-label': `Did track ${otherBefore.track} show this person before the moment` }, beforeBtns)) : null;
  const status = h('div', { class: 'cand-status', 'aria-live': 'polite' });

  // Swap check: what the two answers do together. Returns [text, what is still missing].
  function swapOutcome(live) {
    const a = stay ? stay.track : '?', b = other ? other.track : '?';
    const vStay = stay && live.includes(String(stay.id)) ? s.values[String(stay.id)] : null;
    const vOther = other && live.includes(String(other.id)) ? s.values[String(other.id)] : null;
    if (vStay === 'same' && vOther === 'same') {
      const head = `${refId} is both: no swap, track ${a} stays on this person, and track ${b} shows the same person after this moment.`;
      if (!otherBefore) return [`${head} Track ${b} is joined to this person.`, null];
      if (!beforeAlive()) return [`${head} Track ${b} is joined to this person; no image of it before this moment is left, so nothing is recorded about its earlier part.`, null];
      if (!s.otherBefore) return [null, `${refId} is both candidates: did track ${b} show this person before the moment too? Answer under the comparison images - Unsure is a valid answer.`];
      if (s.otherBefore === 'same') return [`${head} It did before as well: one person tracked twice, and the two tracks are joined.`, null];
      if (s.otherBefore === 'different') return [`${head} Before, track ${b} showed someone else: it is cut at this moment and only its later part is joined to this person.`, null];
      return [`${head} Nothing is recorded about track ${b} before this moment.`, null];
    }
    if (vStay === 'same') return [`No swap: track ${a} stays on this person.`, null];
    const cut = `track ${a} did not stay on this person and is cut at this moment`;
    if (vOther === 'same') return [vStay === 'different' ? `A swap: this person continues under the label ${b}; ${cut}.` : `${refId} continues under the label ${b} after this moment. Track ${a} is not cut, because its candidate was not judged different.`, null];
    if (vStay === 'different') return [vOther === 'different' || !other ? `${refId} is neither: ${cut}.` : `${cut[0].toUpperCase()}${cut.slice(1)}. Nothing is recorded about track ${b}.`, null];
    return ['The question stays open; every "different" is still recorded.', null];
  }

  const fill = (text, who, n) => String(text || '').replace('{ref}', refId).replace('{ids}', who || '').replace('{n}', n === undefined ? '' : n);
  // What this answer will do, in words; `warn` is what still has to be answered before it can be sent.
  function summary() {
    const said = [];
    let warn = null;
    for (const t of tiles) if (flagOf(t.key) === 'other_person' && !kept(t.gid).length) warn = `"Someone else" needs one image of ${t.gid} left as the person the others differ from. Leave one unflagged, or use "not a person" / "two people in one pose".`;
    if (!warn && !refAlive()) said.push(unsureOnly(refId) ? `No image of ${refId} is left to judge, and at least one is unsure: the question stays open.` : `No image of ${refId} is left: the flags are the whole answer.`);
    else if (!warn && mode === 'person') {
      if (s.verdict === 'person') said.push(texts.person || 'A real person.');
      else if (s.verdict === 'not_person') said.push(texts.not_person || 'Not a person.');
      else if (s.verdict === 'unsure') said.push('Unsure: nothing is recorded about the person, and the question stays open.');
      else warn = 'Real person, not a person, or unsure?';
    } else if (!warn) {
      const live = ids.filter((id) => kept(id).length);
      const undecided = live.filter((id) => !s.values[id]);
      const sames = live.filter((id) => s.values[id] === 'same');
      const clash = conflict();
      if (clash) warn = `${clash.ids.join(' and ')} cannot both be ${refId}: ${clash.why}.`;
      else if (!multiple && sames.length > 1) warn = `Only one of these can be ${refId} (now ${sames.join(', ')}).`;
      else if (undecided.length) warn = `${undecided.join(', ')} still need${undecided.length === 1 ? 's' : ''} an answer - Unsure is a valid one.`;
      else if (!live.length) said.push(ids.some(unsureOnly) ? 'No candidate has an image left to judge, and at least one is unsure: the question stays open.' : 'No candidate has an image left to compare: the flags are the whole answer.');
      else if (isSwap) {
        const [text, missing] = swapOutcome(live);
        if (missing) warn = missing; else said.push(text);
      } else if (mode === 'duplicate') {
        const v = s.values[live[0]];
        if (v === 'same') said.push(`${texts.same || 'One person with two poses.'}${s.better && s.better !== 'equal' ? ` Pose ${s.better} is the one used.` : ''}`);
        else said.push(v === 'different' ? (texts.different || 'Two people.') : 'Unsure: nothing is recorded, and the question stays open.');
      } else if (sames.length) {
        const special = sames.map((id) => candidates.find((c) => String(c.id) === id).if_same).filter(Boolean);
        said.push(special.length ? special.join(' ') : fill(texts.same || '{ref} is {ids}.', sames.join(' and ')));
      } else if (live.every((id) => s.values[id] === 'different') && ids.some(unsureOnly)) {
        said.push('A candidate whose images are all unsure keeps the question open; every "different" is still recorded.');
      } else if (live.every((id) => s.values[id] === 'different')) {
        said.push(p.more_candidates ? fill(texts.none_more || '{ref} is none of these.', '', p.more_candidates) : fill(texts.none_closed || '{ref} is none of these.'));
      } else said.push('With "unsure" among the answers the question stays open; every "different" is still recorded.');
    }
    const counts = {};
    let bad = 0;
    Object.values(s.images).forEach((f) => { if (f.exclude) counts[f.exclude] = (counts[f.exclude] || 0) + 1; if (f.bad_pose) bad++; });
    const out = Object.entries(counts).map(([flag, n]) => `${n} ${FLAG_TEXT[flag]}`);
    if (out.length) said.push(`Taken out: ${out.join(', ')}.`);
    if (bad) said.push(`${bad} bad pose${bad > 1 ? 's' : ''} reported.`);
    const again = Object.keys(carried).filter(isCarried).length;
    if (again) said.push(`${again} flag${again > 1 ? 's are' : ' is'} carried over from your earlier answers; undo any that does not hold here.`);
    return { warn, text: said.join(' ') };
  }

  function refresh() {
    tiles.forEach((t) => {
      const f = s.images[t.key] || {};
      t.wrap.classList.toggle('out', !!f.exclude);
      t.wrap.classList.toggle('bad', !!f.bad_pose);
      t.wrap.classList.toggle('flagged', !!(f.exclude || f.bad_pose));
      t.state.textContent = [f.exclude ? FLAG_TEXT[f.exclude] : null, f.bad_pose ? 'bad pose' : null].filter(Boolean).join(' + ')
        + (isCarried(t.key) ? ` - you said so in #${carried[t.key].task}` : '');
      t.wrap.classList.toggle('carried', isCarried(t.key));
    });
    candidates.forEach((c, i) => {
      const id = String(c.id), card = cardEls[i], live = alive(id), v = live ? s.values[id] : null;
      card.className = `cand ${v ? 'v-' + v : live ? 'undecided' : 'void'}`;
      card._buttons.forEach((b) => { const on = b.dataset.value === v; b.setAttribute('aria-pressed', String(on)); b.classList.toggle('on', on); b.disabled = !live; });
      card._note.textContent = live ? '' : kept(id).length ? `nothing left of ${refId} to compare with` : unsureOnly(id) ? 'every image unsure - counts as unsure' : 'no image left - no answer needed';
    });
    verdictBtns.forEach((b) => { const on = b.dataset.value === s.verdict; b.setAttribute('aria-pressed', String(on)); b.classList.toggle('on', on); b.disabled = !refAlive(); });
    if (betterRow) {
      betterRow.hidden = !(ids.length && alive(ids[0]) && s.values[ids[0]] === 'same');
      betterBtns.forEach((b) => { const on = b.dataset.value === (s.better || 'equal'); b.setAttribute('aria-pressed', String(on)); b.classList.toggle('on', on); });
    }
    if (beforeRow) {
      beforeRow.hidden = !(bothSame() && beforeAlive());
      beforeBtns.forEach((b) => { const on = b.dataset.value === s.otherBefore; b.setAttribute('aria-pressed', String(on)); b.classList.toggle('on', on); });
    }
    if (refCard) refCard.classList.toggle('bad', !refAlive());
    const { warn, text } = summary();
    status.className = `cand-status ${warn ? 'warn' : 'ok'}`;
    status.textContent = warn || text;
  }

  const refCard = ref ? h('div', { class: 'ref-card' },
    h('div', { class: 'cand-head' }, h('span', { class: 'ref-badge' }, refId), h('strong', {}, 'Reference'), h('span', { class: 'muted cap' }, ref.caption || '')),
    verdictBtns.length ? h('div', { class: 'cand-controls verdict' }, h('div', { class: 'seg', role: 'group', 'aria-label': 'Is this a real person?' }, verdictBtns)) : null,
    images(ref.panels, refId),
    (p.context_panels || []).length ? h('div', { class: 'context-block' }, h('div', { class: 'muted small' }, (beforeRow ? `For comparison - asked about only when ${refId} is both candidates.` : 'For comparison - not part of the answer.') + (contextKeys.length ? ' An image that does not belong can be flagged like any other.' : '')), contextKeys.length ? images(p.context_panels, CONTEXT) : panelRow(ctx, p.context_panels, { lo: sizes.lo, hi: sizes.hi }), beforeRow) : null) : null;
  if (betterRow && cardEls.length) cardEls[0].append(betterRow);  // right under the answer it belongs to
  const el = h('section', { class: `ref-ui ${ref ? '' : 'no-ref'} ${ids.length ? '' : 'no-cand'}`, style: { '--cols': Math.min(candidates.length, 3) } },
    refCard, ids.length ? h('div', { class: 'cand-grid' }, cardEls) : null);
  refresh();

  const cycle = (i) => { const id = ids[i]; if (!alive(id)) return; s.values[id] = CYCLE[(CYCLE.indexOf(s.values[id]) + 1) % CYCLE.length] || 'same'; refresh(); };
  const only = (i) => { // Shift+digit: this one is the person, every other undecided candidate is not
    const id = ids[i];
    if (!alive(id)) return;
    s.values[id] = 'same';
    for (const other of ids) if (other !== id && alive(other) && (!s.values[other] || (!multiple && s.values[other] === 'same'))) s.values[other] = 'different';
    refresh();
  };
  const flagHelp = [['⚑ or X / M / N / Q', 'on an image: take it out (someone else / two people in one pose / not a person / unsure)'], ['P', 'on an image: report a bad pose (the image stays)']];
  return {
    el,
    outcome: status,  // what the answer will do: shown in the footer, where it stays visible
    hints: mode === 'person' ? [['1', 'real person'], ['2', 'not a person'], ['U', 'unsure'], ['X M N Q P', 'flag the image under the pointer']]
      : [['1-9', multiple ? 'same / different / unsure (several can be same)' : 'same / different / unsure'], ['Shift+1-9', 'this one, the rest different'], ['U', 'unsure for the rest'], ['X M N Q P', 'flag the image under the pointer']],
    help: mode === 'person' ? [['1', 'A real person'], ['2', 'Not a person'], ['U', 'Unsure'], ...flagHelp]
      : [['1 ... 6', `Cycle that candidate: ${LABEL.same.toLowerCase()}, ${LABEL.different.toLowerCase()}, unsure`], ['Shift + 1 ... 6', 'That candidate is the person; undecided others become different'], ['U', 'Mark every undecided candidate "unsure"'], ...flagHelp],
    onKey(e) {
      const k = e.key.toLowerCase();
      if (hovered && 'xmnqp'.includes(k) && k.length === 1) {
        if (k === 'p') toggleBad(hovered.key);
        else {
          const flag = FLAG_KEYS[k];
          if (!(flag === 'other_person' && (groups[hovered.gid] || []).length < 2)) setFlag(hovered.key, flag);
        }
        closeMenu();
        return true;
      }
      const m = /^Digit([1-9])$/.exec(e.code);
      if (m && mode === 'person') { const v = { 1: 'person', 2: 'not_person' }[m[1]]; if (v && refAlive()) { s.verdict = v; refresh(); } return true; }
      if (m) { const i = Number(m[1]) - 1; if (i < ids.length) { e.shiftKey ? only(i) : cycle(i); } return true; }
      if (k === 'u') {
        if (mode === 'person') { if (refAlive()) s.verdict = 'unsure'; } else for (const id of ids) if (alive(id) && !s.values[id]) s.values[id] = 'unsure';
        refresh();
        return true;
      }
      return false;
    },
    check() {
      if ((ctx.answerModel || 1) < 4) return 'This web server runs an older version of the engine than this page and would refuse or lose parts of the answer. Restart it (data-engine serve), reload this page, then answer.';
      return summary().warn;
    },
    answer() {
      const flagged = {};
      for (const [key, f] of Object.entries(s.images)) if (f.exclude || f.bad_pose) flagged[key] = { exclude: f.exclude || null, bad_pose: !!f.bad_pose };
      const out = { images: flagged, candidates: Object.fromEntries(ids.map((id) => [id, alive(id) ? s.values[id] : null])) };
      if (mode === 'person') out.verdict = refAlive() ? s.verdict : null;
      if (mode === 'duplicate') out.better = ids.length && s.values[ids[0]] === 'same' ? (s.better || 'equal') : null;
      if (otherBefore) out.other_before = bothSame() && beforeAlive() ? s.otherBefore : null;
      return out;
    },
    prefill(flags) {
      for (const [key, f] of Object.entries(flags || {})) {
        if (!tiles.some((t) => t.key === key)) continue;
        carried[key] = { exclude: f.exclude || null, bad_pose: !!f.bad_pose, task: f.task };
        s.images[key] = { exclude: carried[key].exclude, bad_pose: carried[key].bad_pose };
      }
      refresh();
    },
    restore(a) {
      for (const [id, v] of Object.entries(a.candidates || {})) if (id in s.values) s.values[id] = v;
      if ('reference_ok' in a || 'none_of_these' in a || 'problems' in a) {
        // an answer given before image flags existed
        if (a.reference_ok === false) {
          const flag = (a.problems || {})[refId] === 'mixed_pose' ? 'mixed' : 'not_person';
          if (mode === 'person') s.verdict = 'not_person'; else (groups[refId] || []).forEach((key) => { entry(key).exclude = flag; });
        } else if (a.none_of_these) {
          if (mode === 'person') s.verdict = 'person'; else ids.forEach((id) => { if (!s.values[id]) s.values[id] = 'different'; });
        }
      } else {
        for (const [key, f] of Object.entries(a.images || {})) if (tiles.some((t) => t.key === key)) s.images[key] = { exclude: f.exclude || null, bad_pose: !!f.bad_pose };
        if (a.verdict) s.verdict = a.verdict;
        if (a.better) s.better = a.better;
        if (a.other_before) s.otherBefore = a.other_before;
      }
      refresh();
    },
  };
}

// ---------------------------------------------------------------- decision
function buildDecision(ctx, task, sizes) {
  const p = task.payload;
  const options = p.options || [];
  const s = { choice: null };
  const hotkeyOf = (o, i) => String(o.hotkey || i + 1).toLowerCase();
  const buttons = options.map((o, i) => h('button', { type: 'button', class: 'option', 'aria-pressed': 'false', dataset: { value: o.value }, onclick: () => pick(o.value) },
    kbd(hotkeyOf(o, i)), h('span', { class: 'opt-label' }, o.label), o.hint ? h('span', { class: 'opt-hint' }, o.hint) : null));
  const unsure = h('button', { type: 'button', class: 'option unsure', 'aria-pressed': 'false', dataset: { value: 'unsure' }, onclick: () => pick('unsure') },
    kbd('u'), h('span', { class: 'opt-label' }, 'Unsure'), h('span', { class: 'opt-hint' }, 'Cannot tell from these images'));
  function pick(v) { s.choice = s.choice === v ? null : v; refresh(); }
  function refresh() { [...buttons, unsure].forEach((b) => { const on = b.dataset.value === s.choice; b.setAttribute('aria-pressed', String(on)); b.classList.toggle('on', on); }); }
  const panels = p.panels || [];
  const el = h('section', { class: 'decision-ui' },
    h('div', { class: `decision-panels n${Math.min(panels.length, 4)}` }, panels.map((x) => panelFigure(ctx, x, { lo: sizes.lo, hi: sizes.hi }))),
    h('div', { class: 'options', role: 'group', 'aria-label': 'Answer options' }, buttons, unsure));
  return {
    el,
    hints: [...options.map((o, i) => [hotkeyOf(o, i), o.label]).slice(0, 5), ['U', 'unsure']],
    help: [...options.map((o, i) => [hotkeyOf(o, i), o.label + (o.hint ? ` - ${o.hint}` : '')]), ['U', 'Unsure']],
    onKey(e) {
      const k = e.key.toLowerCase();
      const i = options.findIndex((o, j) => hotkeyOf(o, j) === k);
      if (i >= 0) { pick(options[i].value); return true; }
      if (k === 'u') { pick('unsure'); return true; }
      return false;
    },
    check: () => (s.choice ? null : 'Pick an option (or Unsure) first.'),
    answer: () => ({ choice: s.choice }),
    restore(a) { s.choice = a.choice; refresh(); },
  };
}

// ---------------------------------------------------------------- frame picker
function buildPicker(ctx, task, sizes) {
  const p = task.payload;
  const candidates = p.candidates || [];
  const s = { cursor: Math.floor(candidates.length / 2), choice: null };
  const viewer = h('div', { class: 'picker-viewer' });
  const title = h('div', { class: 'picker-title' });
  const thumbs = candidates.map((c, i) => {
    const t = h('button', { type: 'button', class: 'thumb', 'aria-label': `Candidate ${i + 1}: ${c.caption || c.id}`, onclick: () => { s.cursor = i; refresh(); } },
      panelFigure(ctx, { ...(c.panels[0] || {}) , label: '' }, { lo: sizes.thumb, hi: 0, noButtons: true, noLens: true, clickZoom: false, cls: 'thumb-fig' }),
      h('span', { class: 'thumb-cap' }, h('span', { class: 'num-badge small' }, i + 1), ' ', c.caption || c.id));
    return t;
  });
  const pickBtn = h('button', { type: 'button', class: 'btn primary', onclick: () => pickCurrent() }, 'This is the same instant ', kbd('Space'));
  const noneBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', onclick: () => setChoice('none') }, 'None of these match ', kbd('N'));
  const unsureBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', onclick: () => setChoice('unsure') }, 'Unsure ', kbd('U'));
  const prevBtn = h('button', { type: 'button', class: 'btn', 'aria-label': 'Previous candidate', onclick: () => move(-1) }, '←');
  const nextBtn = h('button', { type: 'button', class: 'btn', 'aria-label': 'Next candidate', onclick: () => move(1) }, '→');
  const status = h('div', { class: 'cand-status', 'aria-live': 'polite' });
  const viewerFigs = new Map();

  function figsFor(i) {
    if (!viewerFigs.has(i)) viewerFigs.set(i, h('div', { class: 'panel-row' }, (candidates[i].panels || []).map((x) => panelFigure(ctx, x, { lo: sizes.lo, hi: sizes.hi }))));
    return viewerFigs.get(i);
  }
  function move(d) { if (!candidates.length) return; s.cursor = Math.max(0, Math.min(candidates.length - 1, s.cursor + d)); refresh(); }
  function setChoice(v) { s.choice = s.choice === v ? null : v; refresh(); }
  function pickCurrent() { if (candidates.length) setChoice(String(candidates[s.cursor].id)); }
  function refresh() {
    const c = candidates[s.cursor];
    if (c) {
      setChildren(title, h('strong', {}, `Candidate ${s.cursor + 1} / ${candidates.length}`), ' ', h('span', { class: 'muted' }, c.caption || c.id),
        s.choice === String(c.id) ? h('span', { class: 'chip ok' }, 'picked') : null);
      viewer.replaceChildren(figsFor(s.cursor));
      [s.cursor - 1, s.cursor + 1].forEach((j) => { if (candidates[j]) (candidates[j].panels || []).forEach((x) => { new Image().src = panelUrl(ctx.id, x, { w: sizes.lo }); }); });
    }
    thumbs.forEach((t, i) => { t.classList.toggle('cursor', i === s.cursor); t.classList.toggle('picked', s.choice === String(candidates[i].id)); t.setAttribute('aria-pressed', String(s.choice === String(candidates[i].id))); });
    noneBtn.setAttribute('aria-pressed', String(s.choice === 'none')); noneBtn.classList.toggle('active', s.choice === 'none');
    unsureBtn.setAttribute('aria-pressed', String(s.choice === 'unsure')); unsureBtn.classList.toggle('active', s.choice === 'unsure');
    pickBtn.classList.toggle('active', c && s.choice === String(c.id));
    prevBtn.disabled = s.cursor <= 0; nextBtn.disabled = s.cursor >= candidates.length - 1;
    const named = s.choice && !['none', 'unsure'].includes(s.choice) ? candidates.findIndex((x) => String(x.id) === s.choice) + 1 : 0;
    status.className = `cand-status ${s.choice ? 'ok' : 'warn'}`;
    status.textContent = !s.choice ? 'Flip through the candidates (← →) and pick the frame showing the same instant as the reference.'
      : named ? `Picked candidate ${named} (${candidates[named - 1].caption || s.choice}).` : s.choice === 'none' ? 'None of the candidates match.' : 'Unsure.';
  }
  const ref = p.reference || { panels: [] };
  const el = h('section', { class: 'picker-ui' },
    h('div', { class: 'picker-main' },
      h('div', { class: 'picker-ref' }, h('div', { class: 'cand-head' }, h('span', { class: 'ref-badge' }, 'REF'), h('strong', {}, 'Reference'), h('span', { class: 'muted cap' }, ref.caption || '')),
        panelRow(ctx, ref.panels, { lo: sizes.lo, hi: sizes.hi })),
      h('div', { class: 'picker-cand' }, h('div', { class: 'picker-head' }, title, h('div', { class: 'picker-nav' }, prevBtn, nextBtn)), viewer)),
    h('div', { class: 'picker-actions' }, pickBtn, noneBtn, unsureBtn), status,
    h('div', { class: 'thumbs', role: 'group', 'aria-label': 'All candidate frames' }, thumbs));
  refresh();
  return {
    el,
    hints: [['← →', 'flip'], ['Space', 'pick this frame'], ['1-9', 'jump + pick'], ['N', 'none match'], ['U', 'unsure']],
    help: [['← →', 'Flip through candidate frames'], ['Home / End', 'First / last candidate'], ['Space', 'Pick the shown frame as the same instant'], ['1 ... 9', 'Jump to and pick that candidate'], ['N', 'None of the candidates match'], ['U', 'Unsure']],
    onKey(e) {
      const m = /^Digit([1-9])$/.exec(e.code);
      if (m) { const i = Number(m[1]) - 1; if (i < candidates.length) { s.cursor = i; s.choice = String(candidates[i].id); refresh(); } return true; }
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { move(1); return true; }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { move(-1); return true; }
      if (e.key === 'Home') { s.cursor = 0; refresh(); return true; }
      if (e.key === 'End') { s.cursor = candidates.length - 1; refresh(); return true; }
      if (e.key === ' ') { pickCurrent(); return true; }
      const k = e.key.toLowerCase();
      if (k === 'n') { setChoice('none'); return true; }
      if (k === 'u') { setChoice('unsure'); return true; }
      return false;
    },
    check: () => (s.choice ? null : 'Pick a frame (Space), or None (N) / Unsure (U).'),
    answer: () => ({ choice: s.choice }),
    restore(a) { s.choice = a.choice === null || a.choice === undefined ? null : String(a.choice); const i = candidates.findIndex((c) => String(c.id) === s.choice); if (i >= 0) s.cursor = i; refresh(); },
  };
}

// ---------------------------------------------------------------- CVAT
function buildCvat(ctx, task) {
  const p = task.payload;
  const link = h('a', { class: 'btn primary big', href: p.cvat_url, target: '_blank', rel: 'noopener noreferrer' }, 'Open in CVAT');
  const el = h('section', { class: 'cvat-ui panel' },
    h('p', { class: 'instructions' }, p.instructions || 'Correct the pose in CVAT, then come back and confirm.'),
    h('p', {}, link), h('p', { class: 'muted' }, p.cvat_url));
  return {
    el, hints: [['Enter', 'done in CVAT']], help: [['Enter', 'Confirm the CVAT work is finished']],
    answer: () => ({ done: true }),
  };
}

function buildUnknown(ctx, task) {
  return {
    el: h('section', { class: 'panel' }, h('p', { class: 'warn-text' }, `This interface kind ("${task.ui}") is not supported by the web runner; skip it.`),
      h('pre', {}, JSON.stringify(task.payload, null, 2))),
    answer: () => ({}), check: () => 'Unsupported task kind - skip it.',
  };
}
