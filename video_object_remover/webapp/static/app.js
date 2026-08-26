'use strict';

const $ = (id) => document.getElementById(id);
const state = { sid: null, info: null, frame: 0, points: [], job: null, timer: null };

/* ---------------------------------------------------------------- env */

async function checkEnv() {
  try {
    const st = await (await fetch('/api/status')).json();
    const el = $('env');
    if (st.ready) {
      el.className = 'env ok';
      $('env-text').textContent = 'ProPainter + SAM 2 ready';
    } else {
      el.className = 'env bad';
      $('env-text').textContent = st.hint || 'models missing';
    }
  } catch { $('env-text').textContent = 'server unreachable'; }
}

/* --------------------------------------------------------------- open */

async function openSession(promise) {
  $('open-err').textContent = '';
  try {
    const res = await promise;
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    state.info = await res.json();
    state.sid = state.info.id;
    state.frame = 0;
    state.points = [];
    $('frame').max = Math.max(0, state.info.nframes - 1);
    $('frame').value = 0;
    $('meta').textContent =
      `${state.info.width}x${state.info.height} · ${state.info.fps} fps · ` +
      `${state.info.nframes} frames · ${state.info.has_audio ? 'audio' : 'no audio'}`;
    $('out').value = state.info.suggested_output;
    $('step-select').classList.remove('hidden');
    $('step-run').classList.add('hidden');
    await refresh();
    $('step-select').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    $('open-err').textContent = String(e.message || e);
  }
}

$('open-btn').onclick = () => {
  const body = new FormData();
  body.append('path', $('path').value);
  openSession(fetch('/api/open', { method: 'POST', body }));
};
$('path').onkeydown = (e) => { if (e.key === 'Enter') $('open-btn').click(); };

function uploadFile(file) {
  const body = new FormData();
  body.append('file', file);
  openSession(fetch('/api/upload', { method: 'POST', body }));
}
$('file').onchange = (e) => e.target.files[0] && uploadFile(e.target.files[0]);
const drop = $('drop');
['dragenter', 'dragover'].forEach((k) => drop.addEventListener(k, (e) => {
  e.preventDefault(); drop.classList.add('over');
}));
['dragleave', 'drop'].forEach((k) => drop.addEventListener(k, (e) => {
  e.preventDefault(); drop.classList.remove('over');
}));
drop.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) uploadFile(f);
});

/* ------------------------------------------------------------- select */

const view = $('view');

/** Map a mouse event to native video pixels. The served frame is full-res, so
 *  naturalWidth is the source width and the ratio is just display scaling. */
function toVideo(e) {
  const r = view.getBoundingClientRect();
  return {
    x: Math.round((e.clientX - r.left) / r.width * view.naturalWidth),
    y: Math.round((e.clientY - r.top) / r.height * view.naturalHeight),
  };
}

function addPoint(e, label) {
  if (!state.sid || !view.naturalWidth) return;
  const { x, y } = toVideo(e);
  if (x < 0 || y < 0 || x >= view.naturalWidth || y >= view.naturalHeight) return;
  state.points.push({ x, y, label });
  refresh();
}

view.addEventListener('click', (e) => addPoint(e, 1));
view.addEventListener('contextmenu', (e) => { e.preventDefault(); addPoint(e, 0); });

$('undo').onclick = () => { state.points.pop(); refresh(); };
$('clear').onclick = () => { state.points = []; refresh(); };

$('frame').oninput = (e) => {
  state.frame = Number(e.target.value);
  $('frame-n').textContent = state.frame;
};
$('frame').onchange = () => {
  // Points belong to the frame they were clicked on; moving frames drops them.
  if (state.points.length) state.points = [];
  refresh();
};

let pending = null;
async function refresh() {
  if (!state.sid) return;
  const nIn = state.points.filter((p) => p.label === 1).length;
  const nOut = state.points.length - nIn;
  $('pts').textContent = state.points.length
    ? `${nIn} include · ${nOut} exclude` : 'no points yet';
  $('run').disabled = nIn === 0;
  $('frame-n').textContent = state.frame;

  if (pending) pending.abort();
  const ctrl = new AbortController();
  pending = ctrl;
  $('busy').classList.remove('hidden');
  try {
    let blob;
    if (state.points.length === 0) {
      const r = await fetch(`/api/session/${state.sid}/frame?n=${state.frame}`,
        { signal: ctrl.signal });
      blob = await r.blob();
    } else {
      const r = await fetch(`/api/session/${state.sid}/preview`, {
        method: 'POST', signal: ctrl.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: state.frame, points: state.points }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const cov = Number(r.headers.get('X-Mask-Coverage') || 0);
      $('pts').textContent += ` · mask ${(cov * 100).toFixed(1)}% of frame`;
      blob = await r.blob();
    }
    const url = URL.createObjectURL(blob);
    const old = view.src;
    view.src = url;
    if (old.startsWith('blob:')) URL.revokeObjectURL(old);
  } catch (e) {
    if (e.name !== 'AbortError') $('pts').textContent = String(e.message || e);
  } finally {
    if (pending === ctrl) { pending = null; $('busy').classList.add('hidden'); }
  }
}

$('settings-btn').onclick = () => $('settings').classList.toggle('hidden');

/* ---------------------------------------------------------------- run */

$('run').onclick = async () => {
  const body = {
    frame: state.frame, points: state.points, output: $('out').value,
    proc_scale: Number($('proc-scale').value), soften: Number($('soften').value),
    raft_iter: Number($('raft').value), crf: Number($('crf').value),
    preset: $('preset').value, pad: Number($('pad').value),
  };
  const r = await fetch(`/api/session/${state.sid}/run`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) { alert(data.detail || 'run failed'); return; }
  state.job = data.job;
  $('step-run').classList.remove('hidden');
  $('result').classList.add('hidden');
  $('reveal').classList.add('hidden');
  $('bar-fill').style.width = '0%';
  $('cancel').disabled = false;
  $('step-run').scrollIntoView({ behavior: 'smooth', block: 'start' });
  poll();
};

$('cancel').onclick = async () => {
  if (state.job) await fetch(`/api/job/${state.job}/cancel`, { method: 'POST' });
};
$('again').onclick = () => {
  $('step-run').classList.add('hidden');
  $('step-select').scrollIntoView({ behavior: 'smooth', block: 'start' });
};

function showReveal(lines) {
  if (!lines.length) return;
  const el = $('reveal');
  const text = lines.join(' ');
  el.textContent = text;
  el.className = 'notice ' +
    (/GOOD/.test(text) ? 'good' : /POOR/.test(text) ? 'poor' : '');
  el.classList.remove('hidden');
}

async function poll() {
  clearTimeout(state.timer);
  const r = await fetch(`/api/job/${state.job}`);
  if (!r.ok) return;
  const j = await r.json();

  $('bar-fill').style.width = `${j.percent}%`;
  $('stage-txt').textContent = `${j.stage} — ${j.percent.toFixed(0)}%`;
  $('elapsed').textContent = `${j.elapsed.toFixed(0)}s`;
  $('log').textContent = j.tail.join('\n');
  showReveal(j.reveal);

  if (j.state === 'running') {
    state.timer = setTimeout(poll, 1000);
    return;
  }
  $('cancel').disabled = true;
  if (j.state === 'done') {
    $('stage-txt').textContent = `done in ${j.elapsed.toFixed(0)}s`;
    $('result-video').src = `/api/job/${state.job}/result`;
    $('download').href = `/api/job/${state.job}/result`;
    $('out-path').textContent = j.output;
    $('result').classList.remove('hidden');
  } else {
    $('stage-txt').textContent = j.state === 'cancelled' ? 'cancelled' : 'failed — see log';
    $('log-wrap').open = true;
  }
}

checkEnv();
