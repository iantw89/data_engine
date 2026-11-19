// Dataset list and dataset dashboard.
import {
  h, svg, api, toast, setCrumbs, fmtDuration, fmtTime, fmtAgo, fmtNumber, prettyKey, PHASE_LABELS, PHASE_ACTOR,
} from './util.js';

const KIND_COLORS = ['#5aa9ff', '#ff9f5a', '#6fd08c', '#e66fd2', '#f2d45c', '#4fd1c5', '#ff6b6b', '#a78bfa', '#9ccc65', '#ffb3c6', '#8d9db6'];
const ACTOR_LABEL = { human: 'Human', machine: 'Machine', operator: 'Operator' };

// ------------------------------------------------------------------ dataset list
export async function renderIndex(root) {
  setCrumbs([{ text: 'Datasets' }]);
  root.replaceChildren(h('p', { class: 'muted pad' }, 'Loading datasets...'));
  const render = (rows) => {
    if (!rows.length) {
      return root.replaceChildren(h('div', { class: 'panel pad' }, h('h2', {}, 'No datasets yet'),
        h('p', { class: 'muted' }, 'A dataset appears here once its workspace has an engine database.')));
    }
    root.replaceChildren(h('div', { class: 'page' },
      h('h1', {}, 'Datasets'),
      h('div', { class: 'cards' }, rows.map((r) => h('article', { class: `dataset-card phase-${r.phase}` },
        h('div', { class: 'card-top' }, h('a', { class: 'dataset-name', href: `/d/${encodeURIComponent(r.id)}` }, r.id), phasePill(r.phase)),
        h('p', { class: 'muted' }, r.phase_detail),
        h('dl', { class: 'facts' },
          fact('Round', r.round), fact('Open unknowns', r.open_unknowns), fact('Open tasks', r.open_tasks),
          fact('Cameras', r.cameras)),
        h('div', { class: 'card-actions' },
          h('a', { class: 'btn', href: `/d/${encodeURIComponent(r.id)}` }, 'Dashboard'),
          r.open_tasks ? h('a', { class: 'btn primary', href: `/d/${encodeURIComponent(r.id)}/tasks` }, `Review ${r.open_tasks} tasks`) : null))))));
  };
  const refresh = async () => { try { render(await api('/api/datasets')); } catch (e) { toast(e.message, 'error'); } };
  await refresh();
  setInterval(refresh, 8000);
}

const fact = (label, value) => h('div', {}, h('dt', {}, label), h('dd', {}, value === null || value === undefined ? '-' : String(value)));
const phasePill = (phase) => h('span', { class: `pill phase-${phase}` }, PHASE_LABELS[phase] || phase);

// ------------------------------------------------------------------ dataset dashboard
export async function renderDataset(root, id) {
  setCrumbs([{ text: 'Datasets', href: '/' }, { text: id }]);
  const q = encodeURIComponent(id);
  const sections = {};
  const layout = h('div', { class: 'page dashboard' });
  const order = ['banner', 'timeline', 'history', 'pipeline', 'chart', 'unknowns', 'tasks', 'coverage', 'cameras', 'events'];
  for (const name of order) layout.append((sections[name] = h('section', { class: 'dash-section', dataset: { section: name } })));
  root.replaceChildren(layout);

  const seen = {};
  const patch = (name, key, build) => {
    const signature = JSON.stringify(key);
    if (seen[name] === signature) return;
    seen[name] = signature;
    sections[name].replaceChildren(build());
  };
  const state = { logOpen: false, run: null, status: null };

  async function startRun() {
    const s = state.status;
    const pending = s ? s.tasks.answered_pending : 0;
    const msg = `Run the next round for ${id}?\n\n` +
      (pending ? `${pending} answered tasks will be applied as constraints, then the machine stages re-run.\n` : 'No new human answers are waiting; only stale or pending stages will run.\n') +
      'This starts a background process (it can take a long time and uses the GPU).';
    if (!confirm(msg)) return;
    try {
      await api(`/api/datasets/${q}/advance`, { method: 'POST' });
      toast('Run started', 'ok');
      state.logOpen = true;
    } catch (e) { toast(e.message, 'error'); }
    await tick(true);
  }

  async function tick(force) {
    let s;
    try { s = await api(`/api/datasets/${q}/status`); } catch (e) {
      sections.banner.replaceChildren(h('div', { class: 'panel error-box', role: 'alert' }, `Cannot load status: ${e.message}`));
      seen.banner = null;
      return schedule(8000);
    }
    state.status = s;
    const running = s.run.running || s.phase === 'running';
    if (running || state.logOpen) {
      try { state.run = await api(`/api/datasets/${q}/run`); } catch (_) { /* log is optional */ }
    }
    const second = running ? Math.floor(Date.now() / 1000) : 0;
    patch('banner', [s.phase, s.phase_detail, s.next_actions, s.run.running, s.run.reason, state.logOpen, state.run && state.run.tail, s.round,
      running ? s.events.slice(0, 14) : null],
      () => banner(s, id, startRun, state));
    patch('timeline', [s.timeline, s.run.running, second], () => timeline(s));
    // its own section: it changes only when a round ends, so an opened list stays open while a run ticks
    patch('history', [(s.timeline || {}).history, (s.timeline || {}).round], () => roundHistory(s));
    patch('pipeline', [s.stages, s.tasks.open_total, s.tasks.answered_pending, second], () => pipeline(s));
    patch('chart', [s.rounds, s.phase === 'running'], () => roundsChart(s));
    patch('unknowns', s.unknowns, () => unknownsTable(s, id));
    patch('tasks', s.tasks, () => tasksTable(s, id));
    patch('coverage', s.coverage, () => coverage(s));
    patch('cameras', s.cameras, () => cameras(s));
    patch('events', s.events, () => events(s));
    schedule(running ? 1500 : 5000);
  }

  let timer = null;
  function schedule(ms) { clearTimeout(timer); timer = setTimeout(tick, ms); }
  document.addEventListener('visibilitychange', () => { if (!document.hidden) tick(); });
  await tick();
}

// ------------------------------------------------------------------ banner
function banner(s, id, startRun, state) {
  const actor = PHASE_ACTOR[s.phase] || 'machine';
  const running = s.run.running || s.phase === 'running';
  const groups = { human: [], machine: [], operator: [] };
  for (const a of s.next_actions) (groups[a.kind] || groups.operator).push(a);
  const actions = h('div', { class: 'next-actions' }, ['human', 'machine', 'operator'].map((kind) => groups[kind].length ? h('div', { class: `action-group ${kind}` },
    h('span', { class: `actor ${kind}` }, ACTOR_LABEL[kind]),
    h('ul', {}, groups[kind].map((a) => h('li', {},
      a.task_type ? h('a', { href: `/d/${encodeURIComponent(id)}/tasks?type=${encodeURIComponent(a.task_type)}` }, a.text) : a.text,
      a.command ? h('code', {}, a.command) : null)))) : null));
  const runButton = h('button', {
    class: 'btn primary', type: 'button', disabled: running, onclick: startRun,
    title: running ? 'A run is already in progress' : 'Apply answered tasks and run the machine stages',
  }, running ? 'Running...' : 'Run next round');
  const reviewButton = s.tasks.open_total
    ? h('a', { class: 'btn', href: `/d/${encodeURIComponent(id)}/tasks` }, `Review ${s.tasks.open_total} open tasks`) : null;
  const recent = () => s.events.slice(0, 14).reverse().map((e) => `${fmtTime(e.time)}  ${e.stage ? '[' + e.stage + '] ' : ''}${e.message}`).join(String.fromCharCode(10));
  const logBox = (state.logOpen || running) && state.run ? h('details', { class: 'runlog', open: true },
    h('summary', {}, state.run.tail ? `Run log ${state.run.log ? '(' + state.run.log + ')' : ''}` : 'Latest engine messages (this run was not started from this page)'),
    h('pre', {}, state.run.tail || recent() || '(no output yet)')) : (state.run && state.run.log ? h('details', { class: 'runlog' },
    h('summary', {}, `Last run log (${state.run.log})`), h('pre', {}, state.run.tail || '')) : null);
  return h('div', { class: `banner phase-${s.phase}` },
    h('div', { class: 'banner-main' },
      h('div', { class: 'banner-head' }, phasePill(s.phase), h('span', { class: `actor ${actor}` }, actor === 'none' ? 'done' : `next: ${ACTOR_LABEL[actor] || actor}`),
        h('span', { class: 'muted' }, `round ${s.round < 0 ? '-' : s.round}`)),
      h('p', { class: 'banner-detail' }, s.phase_detail),
      s.run.reason === 'stage' && !s.run.pid ? h('p', { class: 'warn-text' }, `A stage (${s.run.stages_running.join(', ')}) is marked running but no web-started process exists; it may be a CLI run, or a crashed one.`) : null,
      actions),
    h('div', { class: 'banner-buttons' }, runButton, reviewButton),
    logBox);
}

// ------------------------------------------------------------------ this round, step by step
const STEP_MARK = { done: '\u2713', running: '\u25cf', expected: '\u25cb', failed: '\u2715', blocked: '\u2715', skipped: '\u2013' };
const lapName = (lap) => (lap ? `Lap ${lap}` : 'First pass');
// What a lap is for, from the stages it runs: with Autofit it tracks again, without it only identity is decided again.
const lapKind = (steps) => (steps.some((x) => x.stage === 'autofit') ? 'Autofit lap: tracking again with the recovered detections'
  : steps.some((x) => x.stage === 'solve') ? 'identity lap: linking tracks with the recovered positions' : 'repeat within the round');

function stepChip(step) {
  const status = step.status || 'expected';
  const elapsed = status === 'running' && step.started ? Date.now() / 1000 - step.started : null;
  const time = status === 'running' ? `${fmtDuration(elapsed)} so far`
    : status === 'expected' ? `queued: ${step.reason}`
      : step.duration_s !== null && step.duration_s !== undefined ? fmtDuration(step.duration_s) : '';
  return h('div', { class: `step sp-${status}`, title: step.message || step.reason || '' },
    h('div', { class: 'step-head' }, h('span', { class: 'step-mark', 'aria-hidden': 'true' }, STEP_MARK[status] || '?'),
      h('span', { class: 'step-title' }, step.title), h('span', { class: 'sr-only' }, ` (${status})`)),
    h('div', { class: 'step-time' }, time, step.started && status !== 'expected' ? ` - ${fmtTime(step.started)}` : ''),
    status === 'running' && step.progress ? progressBar({ title: step.title, progress: step.progress }, elapsed) : null,
    step.message && status !== 'running' ? h('div', { class: 'step-msg' }, step.message) : null);
}

function timeline(s) {
  const header = (text, extra) => h('h2', {}, text, extra ? h('span', { class: 'count' }, extra) : null);
  if (!s.timeline) {
    return h('div', {}, header('This round'), h('p', { class: 'warn-text' },
      'This web server runs an older version of the engine and cannot report the order of the round. Restart it (data-engine serve) to see what is running when.'));
  }
  const t = s.timeline;
  const terms = h('p', { class: 'muted small terms' },
    h('strong', {}, 'Round'), ': one run of the engine - it applies new answers, runs every stage whose inputs changed, in order, and builds the human tasks. ',
    h('strong', {}, 'Lap'), ': a repeat inside a round, asked for by the coverage recovery before the tasks are built. ',
    h('strong', {}, 'Identity lap'), ' (short): the recovered positions let more tracks be linked - identity is decided again on the same association, and the recovery looks only around the joined tracks. ',
    h('strong', {}, 'Autofit lap'), ' (long): the recovery produced enough detections Autofit has not seen - Autofit, the whole solve and the recovery run again.');
  if (t.round === null || t.round === undefined) return h('div', {}, header('This round'), terms, h('p', { class: 'muted' }, 'No round has run yet.'));
  const active = t.finished === null || t.finished === undefined;
  const runningStep = t.steps.find((x) => x.status === 'running');
  const processAlive = s.run.running || !!runningStep;
  const all = [...t.steps, ...(active ? t.expected.map((x) => ({ ...x, status: 'expected' })) : [])];
  const inRound = all.filter((x) => !x.after_round);
  const afterRound = all.filter((x) => x.after_round);
  const laps = [...new Set(inRound.map((x) => x.lap))].sort((a, b) => a - b);
  const state = !active ? `finished ${fmtTime(t.finished)}, took ${fmtDuration(t.finished - t.started)}`
    : processAlive ? `running for ${fmtDuration(Date.now() / 1000 - t.started)}`
      : 'not finished and nothing is running (interrupted) - run the round again to continue';
  const now = runningStep
    ? (runningStep.after_round ? `Now: ${runningStep.title}, started on its own after the round finished - not a new round.`
      : `Now: ${runningStep.title}${runningStep.lap ? ` (lap ${runningStep.lap})` : ''}.`)
    : active ? '' : 'Nothing is running now.';
  const next = active && t.expected.length ? ` Then: ${t.expected.map((x) => x.title).join(', ')}.` : '';
  const mayLap = active && all.some((x) => x.stage === 'recover' && x.status !== 'done')
    ? ' The coverage recovery decides at its end whether another lap is worth it, so the list can grow.' : '';
  const rows = laps.map((lap) => h('div', { class: 'lap-row' },
    h('div', { class: 'lap-label' }, lapName(lap), lap ? h('div', { class: 'muted small' }, lapKind(inRound.filter((x) => x.lap === lap))) : null),
    h('div', { class: 'lap-steps' }, inRound.filter((x) => x.lap === lap).flatMap((x, i) => [i ? h('span', { class: 'arrow', 'aria-hidden': 'true' }) : null, stepChip(x)]))));
  if (afterRound.length) {
    rows.push(h('div', { class: 'lap-row' },
      h('div', { class: 'lap-label' }, 'After the round', h('div', { class: 'muted small' }, 'run on its own')),
      h('div', { class: 'lap-steps' }, afterRound.flatMap((x, i) => [i ? h('span', { class: 'arrow', 'aria-hidden': 'true' }) : null, stepChip(x)]))));
  }
  return h('div', {},
    header(`${active ? 'This round' : 'Last round'}: round ${t.round}`, `${t.trigger} - started ${fmtTime(t.started)} - ${state}`),
    terms,
    h('p', { class: `now-line ${runningStep ? 'is-running' : ''}` }, now, next, mayLap),
    all.length ? h('div', { class: 'laps' }, rows) : h('p', { class: 'muted' }, 'No stage needed to run in this round.'));
}

function roundHistory(s) {
  const t = s.timeline || { round: null, history: [] };
  const history = (t.history || []).filter((r) => r.round !== t.round).reverse();
  const sequence = (r) => {
    const parts = [];
    let lap = 0;
    let after = false;
    r.steps.forEach((x, i) => {
      if (x.after_round && !after) { after = true; parts.push(h('span', { class: 'lap-sep' }, ' | after the round: ')); } else if (!x.after_round && x.lap !== lap) {
        const kind = r.steps.some((y) => y.lap === x.lap && y.stage === 'autofit') ? 'Autofit' : 'identity'; lap = x.lap; parts.push(h('span', { class: 'lap-sep' }, ` | lap ${lap} (${kind}): `)); } else if (i) parts.push(' \u2192 ');
      parts.push(h('span', { class: `seq sp-${x.status}` }, `${x.title}${x.duration_s !== null && x.duration_s !== undefined ? ` ${fmtDuration(x.duration_s)}` : ''}`));
    });
    return parts.length ? parts : h('span', { class: 'muted' }, 'nothing needed to run');
  };
  return h('div', {},
    history.length ? h('details', { class: 'round-history' }, h('summary', {}, `Earlier rounds (${history.length})`),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, ['Round', 'Started', 'Took', 'What ran, in order', 'Open unknowns after'].map((x) => h('th', { scope: 'col' }, x)))),
        h('tbody', {}, history.map((r) => h('tr', {},
          h('td', {}, h('strong', {}, r.round), h('div', { class: 'muted small' }, r.trigger)),
          h('td', {}, fmtTime(r.started)),
          h('td', {}, r.duration_s !== null && r.duration_s !== undefined ? fmtDuration(r.duration_s) : 'interrupted'),
          h('td', { class: 'small' }, sequence(r), r.stopped ? h('div', { class: 'warn-text' }, `stopped: ${r.stopped}`) : null),
          h('td', { class: 'num' }, r.open_total ?? '-'))))))) : null);
}

// ------------------------------------------------------------------ pipeline
function stageCard(stage) {
  const metrics = Object.entries(stage.metrics || {}).filter(([, v]) => ['number', 'string', 'boolean'].includes(typeof v)).slice(0, 4);
  const elapsed = stage.status === 'running' && stage.started ? Date.now() / 1000 - stage.started : null;
  return h('div', { class: `stage st-${stage.status}`, title: stage.description, dataset: { stage: stage.name } },
    h('div', { class: 'stage-head' }, h('span', { class: 'stage-title' }, stage.title), h('span', { class: `badge st-${stage.status}` }, stage.status)),
    h('div', { class: 'stage-sub muted' },
      stage.round !== null && stage.round !== undefined ? `last run: round ${stage.round}` : 'not run',
      ' - ', stage.status === 'running' ? `${fmtDuration(elapsed)} so far` : stage.duration_s !== null && stage.duration_s !== undefined ? fmtDuration(stage.duration_s) : '-',
      stage.runs_this_round > 1 ? ` - ran ${stage.runs_this_round}x this round` : ''),
    stage.note ? h('div', { class: 'stage-note' }, stage.note) : null,
    stage.progress ? progressBar(stage, elapsed) : null,
    metrics.length ? h('dl', { class: 'metrics' }, metrics.map(([k, v]) => h('div', {}, h('dt', {}, prettyKey(k)), h('dd', {}, fmtNumber(v, k))))) : null,
    stage.message ? h('p', { class: 'stage-msg', title: stage.message }, stage.message) : null);
}

function progressBar(stage, elapsed) {
  const p = stage.progress || {};
  const done = p.done ?? p.current ?? p.processed ?? p.frames_done;
  const total = p.total ?? p.n_total ?? p.frames_total;
  let fraction = p.fraction ?? (total ? done / total : null);
  if (fraction !== null && fraction !== undefined) fraction = Math.max(0, Math.min(1, fraction));
  let eta = p.eta_seconds ?? p.eta_s ?? p.eta;
  if ((eta === undefined || eta === null) && fraction && elapsed && fraction > 0.01) eta = elapsed * (1 - fraction) / fraction;
  const stale = p.age_seconds > 120;
  return h('div', { class: `progress ${stale ? 'stale' : ''}`, role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100,
    'aria-valuenow': fraction === null || fraction === undefined ? null : Math.round(fraction * 100), 'aria-label': `${stage.title} progress` },
    h('div', { class: `bar ${fraction === null || fraction === undefined ? 'indeterminate' : ''}`, style: { width: fraction === null || fraction === undefined ? '' : `${fraction * 100}%` } }),
    h('span', { class: 'progress-text' },
      fraction === null || fraction === undefined ? (p.message || 'working...') : `${(fraction * 100).toFixed(0)}%${total ? ` (${fmtNumber(done)}/${fmtNumber(total)})` : ''}`,
      eta !== undefined && eta !== null ? ` - ETA ${fmtDuration(eta)}` : '', stale ? ` - no update for ${fmtDuration(p.age_seconds)}` : ''));
}

function pipeline(s) {
  const counts = {};
  for (const step of (s.timeline && s.timeline.steps) || []) counts[step.stage] = (counts[step.stage] || 0) + 1;
  const stages = s.stages.map((x) => ({ ...x, runs_this_round: counts[x.name] || 0,
    note: x.name === 'autofit' ? 'again only in a lap' : null }));
  const pre = stages.filter((x) => !x.every_round && x.name !== 'publish' && x.name !== 'autofit');
  const loop = stages.filter((x) => x.every_round || x.name === 'autofit');
  const post = stages.filter((x) => x.name === 'publish');
  const arrow = () => h('span', { class: 'arrow', 'aria-hidden': 'true' });
  const humanNode = h('div', { class: `stage human-node ${s.tasks.open_total ? 'st-waiting' : 'st-done'}` },
    h('div', { class: 'stage-head' }, h('span', { class: 'stage-title' }, 'Human answers'), h('span', { class: 'badge st-waiting' }, s.tasks.open_total ? 'waiting' : 'none open')),
    h('dl', { class: 'metrics' },
      h('div', {}, h('dt', {}, 'open tasks'), h('dd', {}, fmtNumber(s.tasks.open_total))),
      h('div', {}, h('dt', {}, 'answered, not applied'), h('dd', {}, fmtNumber(s.tasks.answered_pending)))));
  return h('div', {},
    h('h2', {}, 'Pipeline ', h('span', { class: 'count' }, 'the latest run of every stage; the order of this round is above')),
    h('div', { class: 'pipeline' },
      h('div', { class: 'once' }, h('div', { class: 'lane-label' }, 'Prepared once (re-run only when inputs change)'),
        h('div', { class: 'lane' }, pre.flatMap((x, i) => [i ? arrow() : null, stageCard(x)]))),
      arrow(),
      h('div', { class: 'loop' }, h('div', { class: 'lane-label' }, 'Every round: solve, recover, tasks - lap: Autofit, solve and recover again when the recovery adds detections'),
        h('div', { class: 'lane' }, [...loop.flatMap((x, i) => [i ? arrow() : null, stageCard(x)]), arrow(), humanNode]),
        h('div', { class: 'loop-return' }, h('span', {}, 'answers become constraints; the next round re-solves'))),
      arrow(),
      h('div', { class: 'once' }, h('div', { class: 'lane-label' }, 'On request'), h('div', { class: 'lane' }, post.map(stageCard)))));
}

// ------------------------------------------------------------------ rounds chart
function roundsChart(s) {
  const kinds = s.unknowns.by_kind.map((k) => k.kind);
  const color = (kind) => KIND_COLORS[kinds.indexOf(kind) % KIND_COLORS.length];
  const rounds = s.rounds;
  const titleOf = Object.fromEntries(s.unknowns.by_kind.map((k) => [k.kind, k.title]));
  const header = h('h2', {}, 'Convergence: open unknowns per round');
  if (!rounds.length) return h('div', {}, header, h('p', { class: 'muted' }, 'No rounds yet.'));
  const slot = 78, left = 46, top = 28, plotH = 190, bottom = 44;
  const totals = rounds.map((r) => Object.values(r.summary.open_by_kind || {}).reduce((a, b) => a + b, 0));
  const maxTotal = Math.max(10, ...totals);
  const nice = niceMax(maxTotal);
  const width = left + slot * rounds.length + 20, height = top + plotH + bottom;
  const y = (v) => top + plotH - (v / nice) * plotH;
  const chart = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, class: 'chart', role: 'img',
    'aria-label': 'Stacked bars of open unknowns by kind for each round' });
  for (let i = 0; i <= 4; i++) {
    const v = (nice / 4) * i;
    chart.append(svg('line', { x1: left, x2: width - 10, y1: y(v), y2: y(v), class: 'gridline' }), svg('text', { x: left - 6, y: y(v) + 4, class: 'axis', 'text-anchor': 'end' }, Math.round(v)));
  }
  rounds.forEach((r, i) => {
    const x = left + i * slot + 14, bw = slot - 28;
    let cursor = 0;
    const group = svg('g', { class: 'bar-group' });
    const detail = Object.entries(r.summary.open_by_kind || {}).map(([k, v]) => `${titleOf[k] || k}: ${v}`).join('\n');
    group.append(svg('title', {}, `Round ${r.round} (${r.trigger})\nopen unknowns: ${totals[i]}\n${detail}${r.finished ? '' : '\nnot finished'}`));
    for (const kind of kinds) {
      const v = (r.summary.open_by_kind || {})[kind] || 0;
      if (!v) continue;
      group.append(svg('rect', { x, width: bw, y: y(cursor + v), height: Math.max(1, y(cursor) - y(cursor + v)), fill: color(kind), class: 'seg' }));
      cursor += v;
    }
    if (!r.finished) group.append(svg('rect', { x, width: bw, y: top, height: plotH, class: 'inflight' }));
    group.append(svg('text', { x: x + bw / 2, y: y(cursor) - 6, class: 'total', 'text-anchor': 'middle' }, r.finished ? totals[i] : '...'));
    if (r.summary.shrinking === false) {
      group.append(svg('g', { transform: `translate(${x + bw / 2},${top - 14})`, class: 'warn-mark' },
        svg('title', {}, `Round ${r.round} did not shrink the unknown set`),
        svg('path', { d: 'M0,-9 L10,8 L-10,8 Z' }), svg('text', { y: 6, 'text-anchor': 'middle' }, '!')));
    }
    chart.append(group,
      svg('text', { x: x + bw / 2, y: top + plotH + 16, class: 'axis', 'text-anchor': 'middle' }, `round ${r.round}`),
      svg('text', { x: x + bw / 2, y: top + plotH + 31, class: 'axis dim', 'text-anchor': 'middle' }, r.trigger));
  });
  const legend = h('ul', { class: 'legend' }, kinds.filter((k) => rounds.some((r) => (r.summary.open_by_kind || {})[k])).map((k) =>
    h('li', {}, h('span', { class: 'swatch', style: { background: color(k) } }), titleOf[k] || k)));
  const anyWarn = rounds.some((r) => r.summary.shrinking === false);
  return h('div', {}, header,
    h('p', { class: 'muted' }, 'Every round should shrink this set; the recursion is only justified while it converges.',
      anyWarn ? h('strong', { class: 'warn-text' }, ' A warning marker flags a round that did not shrink it.') : null),
    h('div', { class: 'chart-scroll' }, chart), legend);
}

function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v)); const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

// ------------------------------------------------------------------ tables
function unknownsTable(s, id) {
  const rows = s.unknowns.by_kind;
  const isHidden = (k) => !k.open && !k.resolved && !k.undeterminable;
  const taskTitle = Object.fromEntries(s.tasks.by_type.map((t) => [t.type, t.title]));
  return h('div', {}, h('h2', {}, `Unknowns `, h('span', { class: 'count' }, `${s.unknowns.open_total} open`)),
    h('p', { class: 'muted' }, 'Everything the engine cannot decide. Done means none open: each is resolved or explicitly undeterminable.'),
    h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, ['Kind', 'Open', 'Resolved', 'Undeterminable', 'Machine steps tried first', 'Human task'].map((t) => h('th', { scope: 'col' }, t)))),
      h('tbody', {}, rows.map((k) => h('tr', { class: isHidden(k) ? 'dim' : '' },
        h('td', {}, h('strong', {}, k.title), h('div', { class: 'muted small' }, k.meaning)),
        h('td', { class: `num ${k.open ? 'hot' : ''}` }, k.open),
        h('td', { class: 'num' }, k.resolved), h('td', { class: 'num' }, k.undeterminable),
        h('td', {}, h('ul', { class: 'steps' }, k.machine_steps.map((x) => h('li', {}, x)))),
        h('td', {}, k.task_type ? h('a', { href: `/d/${encodeURIComponent(id)}/tasks?type=${encodeURIComponent(k.task_type)}` }, taskTitle[k.task_type] || k.task_type) : h('span', { class: 'muted' }, 'operator'))))))));
}

function tasksTable(s, id) {
  return h('div', {}, h('h2', {}, 'Human tasks ', h('span', { class: 'count' }, `${s.tasks.open_total} open`)),
    h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, ['Task type', 'Interface', 'Open', 'Answered', 'Applied', 'Skipped', ''].map((t) => h('th', { scope: 'col' }, t)))),
      h('tbody', {}, s.tasks.by_type.map((t) => h('tr', { class: t.open + t.answered + t.applied + t.skipped ? '' : 'dim' },
        h('td', {}, h('strong', {}, t.title), h('div', { class: 'muted small' }, t.question)),
        h('td', {}, h('code', {}, t.ui)),
        h('td', { class: `num ${t.open ? 'hot' : ''}` }, t.open), h('td', { class: 'num' }, t.answered),
        h('td', { class: 'num' }, t.applied), h('td', { class: 'num' }, t.skipped),
        h('td', {}, t.open ? h('a', { class: 'btn small primary', href: `/d/${encodeURIComponent(id)}/tasks?type=${encodeURIComponent(t.type)}` }, 'start') : null)))))));
}

function coverage(s) {
  const entries = Object.entries(s.coverage || {});
  return h('div', {}, h('h2', {}, 'Coverage'),
    entries.length ? h('div', { class: 'tiles' }, entries.map(([k, v]) => h('div', { class: 'tile' },
      h('div', { class: 'tile-value' }, typeof v === 'object' && v !== null ? Object.entries(v).map(([a, b]) => `${a}: ${fmtNumber(b, k)}`).join(', ') : fmtNumber(v, k)),
      h('div', { class: 'tile-label' }, prettyKey(k))))) : h('p', { class: 'muted' }, 'No coverage numbers yet (reported by the solve stage).'));
}

function cameras(s) {
  return h('div', {}, h('h2', {}, 'Cameras'),
    s.cameras.length ? h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, ['Camera', 'Frames', 'fps', 'Resolution', 'Calibration', 'Issues', 'Notes'].map((t) => h('th', { scope: 'col' }, t)))),
      h('tbody', {}, s.cameras.map((c) => h('tr', {},
        h('td', {}, h('strong', {}, c.camera_id)), h('td', { class: 'num' }, fmtNumber(c.frames)), h('td', { class: 'num' }, c.fps),
        h('td', {}, `${c.resolution[0]}x${c.resolution[1]}`), h('td', {}, c.calibration_id ?? '-'),
        h('td', { class: c.issues.length ? 'warn-text' : 'muted' }, c.issues.length ? c.issues.join('; ') : 'none'),
        h('td', { class: 'small muted' }, c.notes.join('; ')))))))
      : h('p', { class: 'muted' }, 'dataset.json has not been written yet (ingest has not run).'));
}

function events(s) {
  return h('div', {}, h('h2', {}, 'Recent events'),
    h('div', { class: 'events', role: 'log' }, s.events.map((e) => h('div', { class: `event ${e.level}` },
      h('span', { class: 'when', title: new Date(e.time * 1000).toLocaleString() }, fmtTime(e.time)),
      h('span', { class: 'lvl' }, e.level), h('span', { class: 'stg' }, e.stage || ''),
      h('span', { class: 'msg' }, e.message)))));
}
