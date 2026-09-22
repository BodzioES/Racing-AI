(() => {
'use strict';
const $ = s => document.querySelector(s);

/* ---------- i18n ---------- */
const I18N = {
  pl: {
    title: 'AI Racers - auta uczą się jeździć',
    sub: 'auta uczą się jeździć od zera',
    gen: 'Pokolenie', alive: 'Na torze', best: 'Najlepszy wynik', lap: 'Rekord okrążenia',
    speed: 'Prędkość symulacji', width: 'Szerokość toru',
    newTrack: 'Nowy tor', randTrack: 'Losowy tor', pause: 'Pauza', resume: 'Wznów', how: 'Jak to działa',
    hintTitle: 'Narysuj tor',
    hintSub: 'Przeciągnij palcem lub myszką po ekranie. Zamknij pętlę, łącząc koniec toru z jego początkiem.',
    toastClose: 'Dociągnij koniec toru do pierścienia na starcie, żeby zamknąć pętlę.',
    toastShort: 'Ta pętla jest za mała. Narysuj większy tor.',
    toastConnErr: 'Brak połączenia z serwerem. Odśwież stronę.',
    learning: 'Uczą się…', lapDone: 'Pierwsze okrążenie zaliczone', mastered: 'Tor opanowany',
    brain: 'Mózg lidera (sieć 6-8-2)', graph: 'Postęp w kolejnych pokoleniach',
    steer: 'skręt', gas: 'gaz', laps: 'okr.', oneLap: '1 okr.', start: 'Start',
    connecting: 'Łączenie z serwerem…', connected: 'Połączono', disconnected: 'Rozłączono - odśwież stronę',
    infoTitle: 'Co tu się dzieje?', close: 'Zamknij',
    info: [
      'Każde auto ma własną sieć neuronową (6-8-2). Na starcie wagi są losowe, więc auta jeżdżą jak popadnie.',
      'Auto widzi tor przez 5 promieni (sensorów) i zna swoją prędkość. Na tej podstawie sieć decyduje o skręcie i gazie.',
      'Po każdym pokoleniu auta, które dojechały najdalej, są krzyżowane i losowo mutowane, a ich potomstwo jedzie w następnym pokoleniu.',
      'Nie ma tu danych treningowych ani wpisanych reguł jazdy. Działa wyłącznie selekcja naturalna, czyli algorytm genetyczny (neuroewolucja).',
      'Cała symulacja i uczenie liczone są w Pythonie (NumPy) na serwerze; ta strona tylko rysuje to, co przyjdzie przez WebSocket.'
    ],
    miniLine: (g, a, p) => `Pokolenie ${g}, na torze ${a} z ${p}`
  },
  en: {
    title: 'AI Racers - cars learning to drive',
    sub: 'cars learning to drive from scratch',
    gen: 'Generation', alive: 'On track', best: 'Best distance', lap: 'Lap record',
    speed: 'Simulation speed', width: 'Track width',
    newTrack: 'New track', randTrack: 'Random track', pause: 'Pause', resume: 'Resume', how: 'How it works',
    hintTitle: 'Draw a track',
    hintSub: 'Drag a finger or mouse anywhere on the screen. Close the loop by joining the end of your track to its start.',
    toastClose: 'Drag the end of the track back to the ring at the start to close the loop.',
    toastShort: 'That loop is too small. Draw a bigger track.',
    toastConnErr: 'Lost connection to the server. Refresh the page.',
    learning: 'Learning…', lapDone: 'First lap completed', mastered: 'Track mastered',
    brain: "Leader's brain (6-8-2 network)", graph: 'Progress by generation',
    steer: 'steer', gas: 'gas', laps: 'laps', oneLap: '1 lap', start: 'Start',
    connecting: 'Connecting to server…', connected: 'Connected', disconnected: 'Disconnected - refresh the page',
    infoTitle: 'What is going on here?', close: 'Close',
    info: [
      "Each car has its own neural network (6-8-2). The weights start out random, so the cars drive aimlessly.",
      'A car sees the track through 5 rays (sensors) and knows its own speed. From that the network decides how much to steer and how much gas to give.',
      'After every generation the cars that got farthest are crossed and randomly mutated, and their offspring drive the next generation.',
      'There is no training data and no hand-written driving rules. Only natural selection is at work: a genetic algorithm (neuroevolution).',
      'All simulation and learning runs in Python (NumPy) on the server; this page only draws whatever arrives over the WebSocket.'
    ],
    miniLine: (g, a, p) => `Generation ${g}, ${a} of ${p} on track`
  }
};
let lang = (navigator.language || 'en').toLowerCase().startsWith('pl') ? 'pl' : 'en';
const t = k => I18N[lang][k];

/* ---------- canvas layers ---------- */
const cv = $('#cv'), ctx = cv.getContext('2d');
const staticLayer = document.createElement('canvas');
const bc = $('#brain'), bctx = bc.getContext('2d');
const gc = $('#graph'), gctx = gc.getContext('2d');
const BW = 226, BH = 118, GW = 226, GH = 64;
const FONT = '"Instrument Sans", system-ui, sans-serif';
const C = {};
let cssW = 0, cssH = 0, dpr = 1;

function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ['bg', 'grid', 'vignette', 'track', 'edge', 'line', 'car', 'lead', 'ok', 'bad', 'accent', 'text', 'muted'])
    C[k] = cs.getPropertyValue('--' + k).trim();
}

/* ---------- app / connection state ---------- */
const RAY_ANG = [-1.2, -0.6, 0, 0.6, 1.2];
let state = 'draw';           // 'draw' | 'connecting' | 'run'
let trackW = 70, widthTouched = false;
let raw = [], drawing = false, pathLen = 0, maxFromStart = 0, pending = null, shortWarned = false;
let centerline = null, carLen = 14;
let latest = null;            // last "frame" message from the server
let history = [];             // best/avg per generation, for the graph
let speedMult = 3, paused = false;
let ws = null, connStatus = 'connecting';

function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws`;
}
function connect() {
  connStatus = 'connecting'; updateConn();
  ws = new WebSocket(wsUrl());
  ws.onopen = () => { connStatus = 'connected'; updateConn(); };
  ws.onclose = () => { connStatus = 'disconnected'; updateConn(); toast(t('toastConnErr'), 6000); };
  ws.onerror = () => { connStatus = 'disconnected'; updateConn(); };
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'track_ready') {
      centerline = msg.centerline; trackW = msg.width; carLen = Math.min(20, Math.max(11, trackW * 0.26));
      state = 'run'; renderStatic(); updateHud();
    } else if (msg.type === 'track_too_short') {
      toast(t('toastShort')); resetToDraw();
    } else if (msg.type === 'frame') {
      latest = msg;
      if (msg.history && msg.history.length) { history.push(msg.history[0]); if (history.length > 150) history.shift(); drawGraph(); }
      updateHud(); drawBrain();
    }
  };
}

/* ---------- geometry helpers (drawing UI only; server owns the real track) ---------- */
function getCloseR() { return Math.max(30, trackW * 0.6); }
function getMinLen() { return Math.max(320, trackW * 5); }
function isArmed() { return raw.length > 1 && pathLen >= getMinLen() && maxFromStart > getCloseR() * 2.5; }

function tracePath(g, pts) {
  g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
  g.closePath();
}
function drawTrack(g) {
  g.lineJoin = 'round'; g.lineCap = 'round';
  tracePath(g, centerline); g.strokeStyle = C.edge; g.lineWidth = trackW + 5; g.stroke();
  tracePath(g, centerline); g.strokeStyle = C.track; g.lineWidth = trackW; g.stroke();
  tracePath(g, centerline); g.strokeStyle = C.line; g.lineWidth = 1.5; g.setLineDash([9, 13]); g.stroke(); g.setLineDash([]);
  const [sx, sy] = centerline[0], [nx, ny] = centerline[1];
  const startA = Math.atan2(ny - sy, nx - sx);
  g.save(); g.translate(sx, sy); g.rotate(startA);
  const sq = 6, cols = Math.max(2, Math.round(trackW / sq)), total = cols * sq;
  for (let r = 0; r < 2; r++) for (let c = 0; c < cols; c++) {
    g.fillStyle = (r + c) % 2 ? C.text : C.track;
    g.fillRect(-sq + r * sq, -total / 2 + c * sq, sq, sq);
  }
  g.restore();
}
function renderStatic() {
  const d = dpr;
  staticLayer.width = Math.floor(cssW * d); staticLayer.height = Math.floor(cssH * d);
  const g = staticLayer.getContext('2d');
  g.setTransform(d, 0, 0, d, 0, 0);
  g.fillStyle = C.bg; g.fillRect(0, 0, cssW, cssH);
  g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
  for (let x = 0.5; x < cssW; x += 40) { g.moveTo(x, 0); g.lineTo(x, cssH); }
  for (let y = 0.5; y < cssH; y += 40) { g.moveTo(0, y); g.lineTo(cssW, y); }
  g.stroke();
  const grd = g.createRadialGradient(cssW / 2, cssH / 2, Math.min(cssW, cssH) * 0.3, cssW / 2, cssH / 2, Math.max(cssW, cssH) * 0.75);
  grd.addColorStop(0, 'rgba(0,0,0,0)'); grd.addColorStop(1, C.vignette);
  g.fillStyle = grd; g.fillRect(0, 0, cssW, cssH);
  if (centerline) drawTrack(g);
}
function fitSmall(c, w, h) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.width = w + 'px'; c.style.height = h + 'px'; }
function defaultWidth() { return Math.round(Math.min(90, Math.max(46, Math.min(cssW, cssH) * 0.085))); }
function resize() {
  const r = cv.getBoundingClientRect();
  const w = Math.max(1, Math.floor(r.width)), h = Math.max(1, Math.floor(r.height));
  const d = Math.min(window.devicePixelRatio || 1, 2);
  cssW = w; cssH = h; dpr = d;
  cv.width = Math.floor(w * d); cv.height = Math.floor(h * d);
  fitSmall(bc, BW, BH); fitSmall(gc, GW, GH);
  if (!widthTouched) { trackW = defaultWidth(); $('#rWidth').value = trackW; $('#oWidth').textContent = trackW + ' px'; }
  renderStatic(); drawGraph(); drawBrain();
}

/* ---------- track lifecycle ---------- */
function submitTrack(points) {
  const cssPts = points;
  send({ type: 'track', points: cssPts, width: trackW, w: cssW, h: cssH });
  state = 'connecting'; drawing = false; raw = points;
}
function resetToDraw() {
  state = 'draw'; centerline = null; latest = null; history = [];
  raw = []; drawing = false; pathLen = 0; maxFromStart = 0; pending = null; shortWarned = false;
  send({ type: 'reset' });
  renderStatic(); drawGraph(); drawBrain(); updateHud();
}
function randomTrack() {
  const K = [2, 3, 4, 5], rr = (a, b) => a + Math.random() * (b - a);
  const amp = [rr(.08, .22), rr(.05, .16), rr(.03, .10), rr(.02, .07)];
  const ph = K.map(() => Math.random() * Math.PI * 2);
  const n = 180, pts = [];
  let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
  for (let i = 0; i < n; i++) {
    const th = i / n * Math.PI * 2;
    let r = 1;
    for (let k = 0; k < K.length; k++) r += amp[k] * Math.sin(K[k] * th + ph[k]);
    const x = Math.cos(th) * r, y = Math.sin(th) * r;
    pts.push([x, y]);
    if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y;
  }
  const m = Math.max(trackW * 0.9, 50) + 10;
  const sx = (cssW - 2 * m) / (maxx - minx), sy = (cssH - 2 * m) / (maxy - miny);
  let out = pts.map(p => [m + (p[0] - minx) * sx, m + (p[1] - miny) * sy]);
  if (Math.random() < 0.5) out.reverse();
  const shift = Math.floor(Math.random() * n);
  out = out.slice(shift).concat(out.slice(0, shift));
  submitTrack(out);
}

/* ---------- drawing input ---------- */
function pos(ev) { const r = cv.getBoundingClientRect(); return { x: ev.clientX - r.left, y: ev.clientY - r.top }; }
function beginSketch(p) {
  if (state !== 'draw') resetToDraw();
  raw = [[p.x, p.y]]; drawing = true; pathLen = 0; maxFromStart = 0; shortWarned = false;
}
function handleMove(p) {
  if (pending) {
    if (Math.hypot(p.x - pending.x, p.y - pending.y) > 30) { const s = pending; pending = null; beginSketch(s); }
    else return;
  }
  if (!drawing) return;
  const last = raw[raw.length - 1], d = Math.hypot(p.x - last[0], p.y - last[1]);
  if (d < 4) return;
  raw.push([p.x, p.y]); pathLen += d;
  const ds = Math.hypot(p.x - raw[0][0], p.y - raw[0][1]);
  if (ds > maxFromStart) maxFromStart = ds;
  if (ds < getCloseR() && maxFromStart > getCloseR() * 2.5) {
    if (pathLen >= getMinLen()) { drawing = false; submitTrack(raw); }
    else if (!shortWarned) { shortWarned = true; toast(t('toastShort')); }
  }
}
cv.addEventListener('pointerdown', e => {
  cv.setPointerCapture(e.pointerId);
  const p = pos(e);
  if (state !== 'draw') { pending = p; return; }
  if (raw.length > 1 && !drawing) {
    const end = raw[raw.length - 1];
    if (Math.hypot(p.x - end[0], p.y - end[1]) < 60) { drawing = true; return; }
  }
  beginSketch(p);
});
cv.addEventListener('pointermove', e => {
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
  if (evs.length) for (const ev of evs) handleMove(pos(ev)); else handleMove(pos(e));
});
function endStroke() {
  pending = null;
  if (drawing) { drawing = false; if (raw.length > 8 && state === 'draw') toast(t('toastClose')); }
}
cv.addEventListener('pointerup', endStroke);
cv.addEventListener('pointercancel', endStroke);
cv.addEventListener('contextmenu', e => e.preventDefault());

/* ---------- rendering ---------- */
function dart(x, y, a, color, alpha, glow) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(a);
  ctx.globalAlpha = alpha; ctx.fillStyle = color;
  if (glow) { ctx.shadowColor = color; ctx.shadowBlur = 16; }
  const L = carLen, W = carLen * 0.6;
  ctx.beginPath();
  ctx.moveTo(L * 0.6, 0); ctx.lineTo(-L * 0.5, -W / 2); ctx.lineTo(-L * 0.25, 0); ctx.lineTo(-L * 0.5, W / 2);
  ctx.closePath(); ctx.fill(); ctx.restore();
}
function drawSim() {
  const f = latest;
  if (!f) return;
  if (f.deaths) {
    ctx.fillStyle = C.bad; ctx.globalAlpha = 0.35;
    for (const [x, y] of f.deaths) ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    ctx.globalAlpha = 1;
  }
  for (const [x, y, a] of f.cars) dart(x, y, a, C.car, 0.62, false);
  if (f.leaderPos) {
    const [lx, ly, la] = f.leaderPos;
    if (f.leaderRays) {
      const rayLen = Math.max(110, trackW * 1.9);
      ctx.lineWidth = 1.5;
      for (let r = 0; r < 5; r++) {
        const a = la + RAY_ANG[r], frac = f.leaderRays[r], d = frac * rayLen;
        const ex = lx + Math.cos(a) * d, ey = ly + Math.sin(a) * d;
        ctx.strokeStyle = frac < 0.5 ? C.bad : C.ok; ctx.globalAlpha = 0.75;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(ex, ey, 2.5, 0, 7); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    dart(lx, ly, la, C.lead, 1, true);
    ctx.shadowBlur = 0;
  }
  ctx.globalAlpha = 1;
}
function drawSketch(now) {
  const closeR = getCloseR(), armed = isArmed(), p = 0.5 + 0.5 * Math.sin(now / 260);
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  if (raw.length > 1) {
    ctx.beginPath(); ctx.moveTo(raw[0][0], raw[0][1]);
    for (let i = 1; i < raw.length; i++) ctx.lineTo(raw[i][0], raw[i][1]);
    ctx.globalAlpha = 0.14; ctx.strokeStyle = C.edge; ctx.lineWidth = trackW; ctx.stroke();
    ctx.globalAlpha = 1; ctx.strokeStyle = C.text; ctx.lineWidth = 2.5; ctx.stroke();
  }
  if (raw.length > 0) {
    const s = raw[0], col = armed ? C.ok : C.accent;
    ctx.beginPath(); ctx.arc(s[0], s[1], closeR + (armed ? p * 5 : 0), 0, Math.PI * 2);
    ctx.globalAlpha = armed ? 0.16 : 0.07; ctx.fillStyle = col; ctx.fill(); ctx.globalAlpha = 1;
    ctx.setLineDash(armed ? [] : [6, 6]); ctx.lineWidth = armed ? 3 : 2; ctx.strokeStyle = col; ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(s[0], s[1], 5, 0, 7); ctx.fillStyle = C.text; ctx.fill();
    if (raw.length > 1) {
      ctx.font = '600 13px ' + FONT; ctx.textAlign = 'center'; ctx.fillStyle = C.text;
      ctx.fillText(t('start'), s[0], s[1] - closeR - 10);
    }
  }
  if (!drawing && raw.length > 1) {
    const e = raw[raw.length - 1];
    ctx.beginPath(); ctx.arc(e[0], e[1], 7 + p * 3, 0, 7); ctx.fillStyle = C.car; ctx.fill();
  }
}
function render(now) {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.globalAlpha = 1;
  ctx.drawImage(staticLayer, 0, 0, cssW, cssH);
  if (state === 'run') drawSim(); else drawSketch(now);
}

function drawBrain() {
  const g = bctx;
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, BW, BH);
  const f = latest;
  if (!f || !f.leaderBrain) return;
  const { w1, b1, w2, b2 } = f.leaderBrain;
  const NIN = w1.length, NH = w1[0].length, NOUT = w2[0].length;
  const inp = f.leaderInput, hidden = f.leaderHidden, out = f.leaderOutput;
  const ix = 12, hx = 100, ox = 158;
  const iy = i => 8 + i * ((BH - 16) / (NIN - 1)), hy = j => 8 + j * ((BH - 16) / (NH - 1)), oy = k => BH / 2 - 24 + k * 48;
  g.lineWidth = 1;
  for (let i = 0; i < NIN; i++) for (let j = 0; j < NH; j++) {
    const v = w1[i][j];
    g.strokeStyle = v >= 0 ? C.accent : C.car; g.globalAlpha = Math.min(0.5, Math.abs(v) * 0.3 + 0.03);
    g.beginPath(); g.moveTo(ix, iy(i)); g.lineTo(hx, hy(j)); g.stroke();
  }
  for (let j = 0; j < NH; j++) for (let k = 0; k < NOUT; k++) {
    const v = w2[j][k];
    g.strokeStyle = v >= 0 ? C.accent : C.car; g.globalAlpha = Math.min(0.6, Math.abs(v) * 0.3 + 0.03);
    g.beginPath(); g.moveTo(hx, hy(j)); g.lineTo(ox, oy(k)); g.stroke();
  }
  const node = (x, y, v) => {
    g.globalAlpha = 1; g.beginPath(); g.arc(x, y, 5, 0, 7);
    g.fillStyle = v >= 0 ? C.accent : C.car; g.globalAlpha = 0.15 + 0.85 * Math.min(1, Math.abs(v)); g.fill();
    g.globalAlpha = 1; g.strokeStyle = C.muted; g.lineWidth = 1; g.stroke();
  };
  for (let i = 0; i < NIN; i++) node(ix, iy(i), inp[i]);
  for (let j = 0; j < NH; j++) node(hx, hy(j), hidden[j]);
  for (let k = 0; k < NOUT; k++) node(ox, oy(k), out[k]);
  g.globalAlpha = 1; g.fillStyle = C.muted; g.font = '12px ' + FONT; g.textAlign = 'left';
  g.fillText(t('steer'), ox + 12, oy(0) + 4); g.fillText(t('gas'), ox + 12, oy(1) + 4);
}
function drawGraph() {
  const g = gctx;
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, GW, GH);
  const pad = 4;
  let mx = 1.2;
  for (const p of history) mx = Math.max(mx, p.best * 1.08);
  const X = i => pad + (history.length < 2 ? 0 : i * (GW - 2 * pad) / (history.length - 1));
  const Y = v => GH - pad - (v / mx) * (GH - 2 * pad);
  g.setLineDash([3, 4]); g.lineWidth = 1; g.strokeStyle = C.muted; g.globalAlpha = 0.7;
  g.beginPath(); g.moveTo(pad, Y(1)); g.lineTo(GW - pad, Y(1)); g.stroke(); g.setLineDash([]);
  g.globalAlpha = 1; g.fillStyle = C.muted; g.font = '11px ' + FONT; g.textAlign = 'left';
  g.fillText(t('oneLap'), pad, Y(1) - 3);
  if (history.length > 1) {
    g.lineJoin = 'round';
    g.strokeStyle = C.muted; g.lineWidth = 1.5; g.beginPath();
    history.forEach((p, i) => i ? g.lineTo(X(i), Y(p.avg)) : g.moveTo(X(i), Y(p.avg))); g.stroke();
    g.strokeStyle = C.accent; g.lineWidth = 2.2; g.beginPath();
    history.forEach((p, i) => i ? g.lineTo(X(i), Y(p.best)) : g.moveTo(X(i), Y(p.best))); g.stroke();
  }
}

/* ---------- HUD ---------- */
function updateConn() {
  $('#connLine').textContent = connStatus === 'connecting' ? t('connecting')
    : connStatus === 'connected' ? '' : t('disconnected');
}
function updateHud() {
  const run = state === 'run', f = latest;
  $('#vGen').textContent = f ? f.gen : 1;
  $('#vAlive').textContent = run && f ? `${f.alive} / ${f.pop}` : '-';
  $('#vBest').textContent = run && f ? f.bestLaps.toFixed(2) + ' ' + t('laps') : '-';
  $('#vLap').textContent = run && f && f.bestLapSec != null ? f.bestLapSec.toFixed(1) + ' s' : '-';
  $('#miniLine').textContent = run && f ? t('miniLine')(f.gen, f.alive, f.pop) : '';
  const st = $('#status');
  st.hidden = !(run && f);
  if (run && f) {
    const mastered = f.lapFraction >= 0.3, lapped = f.bestLapSec != null;
    st.textContent = mastered ? t('mastered') : lapped ? t('lapDone') : t('learning');
    st.className = 'status' + (mastered ? ' done' : lapped ? ' lap' : '');
  }
  $('#hint').classList.toggle('gone', state !== 'draw' || raw.length > 0);
}
function applyLang() {
  document.documentElement.lang = lang;
  document.title = t('title');
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  $('#langBtn').textContent = lang === 'pl' ? 'EN' : 'PL';
  $('#bPause').textContent = paused ? t('resume') : t('pause');
  const ul = $('#infoList'); ul.textContent = '';
  for (const s of t('info')) { const li = document.createElement('li'); li.textContent = s; ul.appendChild(li); }
  updateConn(); updateHud(); drawGraph(); drawBrain();
}
let toastT = 0;
function toast(msg, ms = 2800) {
  const el = $('#toast'); el.textContent = msg; el.classList.add('show');
  clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove('show'), ms);
}

/* ---------- controls ---------- */
$('#bNew').addEventListener('click', resetToDraw);
$('#bRand').addEventListener('click', () => { if (state !== 'draw') resetToDraw(); randomTrack(); });
$('#bPause').addEventListener('click', () => { paused = !paused; send({ type: 'pause', value: paused }); $('#bPause').textContent = paused ? t('resume') : t('pause'); });
$('#bInfo').addEventListener('click', () => $('#info').classList.add('open'));
$('#infoClose').addEventListener('click', () => $('#info').classList.remove('open'));
$('#info').addEventListener('click', e => { if (e.target.id === 'info') $('#info').classList.remove('open'); });
window.addEventListener('keydown', e => { if (e.key === 'Escape') $('#info').classList.remove('open'); });
$('#langBtn').addEventListener('click', () => { lang = lang === 'pl' ? 'en' : 'pl'; applyLang(); });
$('#colBtn').addEventListener('click', () => {
  const h = $('#hud'); h.classList.toggle('collapsed');
  $('#colBtn').innerHTML = h.classList.contains('collapsed') ? '+' : '&minus;';
});
$('#rSpeed').addEventListener('input', e => { speedMult = +e.target.value; $('#oSpeed').textContent = speedMult + 'x'; send({ type: 'speed', value: speedMult }); });
let widthTimer = 0;
$('#rWidth').addEventListener('input', e => {
  widthTouched = true; trackW = +e.target.value; $('#oWidth').textContent = trackW + ' px';
  if (state === 'run') {
    clearTimeout(widthTimer);
    widthTimer = setTimeout(() => submitTrack(raw), 250);
  }
});
if (!document.fullscreenEnabled) $('#bFull').hidden = true;
$('#bFull').addEventListener('click', () => {
  try {
    const p = document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    if (p && p.catch) p.catch(() => {});
  } catch (err) { /* fullscreen not available here */ }
});

/* ---------- theme + resize ---------- */
function themeChanged() { readColors(); renderStatic(); drawGraph(); drawBrain(); }
const mq = window.matchMedia('(prefers-color-scheme: dark)');
if (mq.addEventListener) mq.addEventListener('change', themeChanged);
new MutationObserver(themeChanged).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
let resizeTimer = 0;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(resize, 120); });

/* ---------- main loop (rendering only - the server drives the simulation) ---------- */
function loop(now) {
  render(now);
  requestAnimationFrame(loop);
}

readColors();
if (innerWidth < 700) { $('#hud').classList.add('collapsed'); $('#colBtn').textContent = '+'; }
resize();
$('#oWidth').textContent = trackW + ' px';
applyLang();
connect();
requestAnimationFrame(loop);
})();
