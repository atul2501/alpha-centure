"use strict";
/* Alpha Centure dashboard. Polls the JSON API and patches only the parts that changed (no page reloads):
   workflow every 2 s, ledger and market every 10 s, the candle chart every 30 s. Clocks tick every second. */

const $ = id => document.getElementById(id);
const POLL = { workflow: 2000, ledger: 10000, market: 10000 };
const LABEL = { ok: "OK", warn: "CHECK", fail: "PROBLEM", idle: "WAITING" };
const KINDS = ["decision", "order", "fill", "funding", "connection", "error"];
const S = { page: null, wf: null, lg: null, mk: null, okAt: 0, err: null, busy: false, kinds: new Set(KINDS),
  stage: null, tab: "decisions", candleKey: "", candleAt: 0 };

/* ---------- helpers ---------- */
function setHTML(el, html) { if (el && el._h !== html) { el.innerHTML = html; el._h = html; } }
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 2) => v == null || !isFinite(v) ? "–" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const usd = (v, d = 2) => v == null || !isFinite(v) ? "–" : (v < 0 ? "−$" : "$") + num(Math.abs(v), d);
const sUsd = (v, d = 2) => v == null || !isFinite(v) ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + "$" + num(Math.abs(v), d);
const pct = (v, d = 2) => v == null || !isFinite(v) ? "–" : (v * 100).toFixed(d) + "%";
const sPct = (v, d = 2) => v == null || !isFinite(v) ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toFixed(d) + "%";
const cls = v => v > 0 ? "pos" : v < 0 ? "neg" : "mut";
const qty = v => v == null ? "–" : Math.abs(v) >= 100 ? num(v, 1) : Math.abs(v) >= 1 ? num(v, 3) : num(v, 5);
const px = v => v == null ? "–" : v >= 1000 ? num(v, 1) : v >= 1 ? num(v, 3) : num(v, 5);
function ago(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return sec + "s";
  if (sec < 3600) return Math.floor(sec / 60) + "m " + String(sec % 60).padStart(2, "0") + "s";
  if (sec < 86400) return Math.floor(sec / 3600) + "h " + String(Math.floor(sec % 3600 / 60)).padStart(2, "0") + "m";
  return Math.floor(sec / 86400) + "d " + Math.floor(sec % 86400 / 3600) + "h";
}
function countdown(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  const d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
  return (d ? d + "d " : "") + [h, m, s].map(x => String(x).padStart(2, "0")).join(":");
}
const toDate = t => new Date(String(t).replace(" ", "T"));  /* Postgres text timestamps use a space; Safari needs the T */
const tsMs = t => t ? toDate(t).getTime() : NaN;
const utc = (t, withDate = false) => { if (!t) return "–"; const d = toDate(t); if (isNaN(d)) return "–"; const iso = d.toISOString();
  return (withDate ? iso.slice(5, 10) + " " : "") + iso.slice(11, 19); };
function since(t) { return (Date.now() - tsMs(t)) / 1000; }

function table(cols, rows, empty = "Nothing yet") {
  if (!rows || !rows.length) return `<div class="empty">${esc(empty)}</div>`;
  const head = cols.map(c => `<th class="${c.l ? "l" : ""}">${esc(c.h)}</th>`).join("");
  const body = rows.map(r => "<tr>" + cols.map(c => {
    const v = c.f ? c.f(r[c.k], r) : esc(r[c.k] ?? "–");
    const k = typeof c.c === "function" ? c.c(r[c.k], r) : (c.c || "");
    return `<td class="${k}${c.l ? " l" : ""}">${v}</td>`;
  }).join("") + "</tr>").join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

/* tooltip */
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight, x = e.clientX + 14, y = e.clientY + 14;
  tip.style.left = Math.min(x, innerWidth - w - 8) + "px"; tip.style.top = Math.min(y, innerHeight - h - 8) + "px"; }
const hideTip = () => { tip.hidden = true; };

/* ---------- svg charts ---------- */
const NS = "http://www.w3.org/2000/svg";
function svg(w, h, label) { const s = document.createElementNS(NS, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`); s.setAttribute("role", "img"); s.setAttribute("aria-label", label); return s; }
function el(s, tag, a) { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); s.appendChild(e); return e; }
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || Math.abs(hi) || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
}

function lineChart(box, pts, { h = 240, fmt = v => num(v), base = null, color = "var(--accent)", area = true, label = "chart" } = {}) {
  box.innerHTML = "";
  if (!pts.length) { box.innerHTML = `<div class="empty">No data yet</div>`; return; }
  const W = 900, m = { l: 64, r: 14, t: 10, b: 24 }, iw = W - m.l - m.r, ih = h - m.t - m.b;
  const vs = pts.map(p => p.v).concat(base != null ? [base] : []);
  let lo = Math.min(...vs), hi = Math.max(...vs); if (lo === hi) { lo -= Math.abs(lo) * 0.01 || 1; hi += Math.abs(hi) * 0.01 || 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const t0 = tsMs(pts[0].t), t1 = tsMs(pts[pts.length - 1].t) || t0 + 1;
  const x = t => m.l + ((tsMs(t) - t0) / ((t1 - t0) || 1)) * iw, y = v => m.t + (1 - (v - lo) / (hi - lo)) * ih;
  const s = svg(W, h, label);
  niceTicks(lo, hi, 4).forEach(v => { el(s, "line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--line)" });
    el(s, "text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }).textContent = fmt(v); });
  [0, 0.25, 0.5, 0.75, 1].forEach(f => { const t = t0 + f * (t1 - t0), d = new Date(t);
    el(s, "text", { x: m.l + f * iw, y: h - 6, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle" }).textContent =
      (t1 - t0) > 3 * 86400e3 ? d.toISOString().slice(5, 10) : d.toISOString().slice(5, 16).replace("T", " "); });
  if (base != null) el(s, "line", { x1: m.l, x2: W - m.r, y1: y(base), y2: y(base), stroke: "var(--muted)", "stroke-dasharray": "4 4" });
  const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join("");
  if (area) {
    const gid = "g" + Math.random().toString(36).slice(2, 8), defs = el(s, "defs", {}), g = el(defs, "linearGradient", { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 });
    el(g, "stop", { offset: "0%", "stop-color": color, "stop-opacity": 0.28 }); el(g, "stop", { offset: "100%", "stop-color": color, "stop-opacity": 0 });
    el(s, "path", { d: d + `L${x(pts[pts.length - 1].t)},${m.t + ih}L${x(pts[0].t)},${m.t + ih}Z`, fill: `url(#${gid})` });
  }
  el(s, "path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round" });
  const last = pts[pts.length - 1]; el(s, "circle", { cx: x(last.t), cy: y(last.v), r: 3.5, fill: color });
  const cross = el(s, "line", { y1: m.t, y2: m.t + ih, stroke: "var(--ink-2)", opacity: 0 }), dot = el(s, "circle", { r: 4, fill: color, opacity: 0 });
  const hit = el(s, "rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  hit.addEventListener("pointermove", e => { const r = s.getBoundingClientRect(), xv = (e.clientX - r.left) / r.width * W;
    let i = 0, best = Infinity; pts.forEach((p, j) => { const dd = Math.abs(x(p.t) - xv); if (dd < best) { best = dd; i = j; } });
    const p = pts[i]; cross.setAttribute("x1", x(p.t)); cross.setAttribute("x2", x(p.t)); cross.setAttribute("opacity", .4);
    dot.setAttribute("cx", x(p.t)); dot.setAttribute("cy", y(p.v)); dot.setAttribute("opacity", 1);
    showTip(e, `<div class="h">${utc(p.t, true)} UTC</div><div class="r"><span>value</span><b>${fmt(p.v)}</b></div>`); });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); hideTip(); });
  box.appendChild(s);
}

function waterfallChart(box, w) {
  box.innerHTML = "";
  const steps = [["Start", w.start, "total"], ["Trading", w.trading, "delta"], ["Fees", -w.fees, "delta"], ["Funding", -w.funding, "delta"], ["Equity", w.equity, "total"]];
  const W = 520, H = 250, m = { l: 64, r: 10, t: 16, b: 28 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  let run = 0; const bars = steps.map(([n, v, k]) => { const from = k === "total" ? 0 : run, to = k === "total" ? v : run + v; run = to; return { n, v, k, from, to }; });
  const vals = bars.flatMap(b => [b.from, b.to]); let lo = Math.min(...vals.filter(v => v !== 0), w.start), hi = Math.max(...vals);
  const span = hi - lo || hi * 0.02 || 1; lo = Math.max(0, lo - span * 0.6); hi += span * 0.25;
  const y = v => m.t + (1 - (v - lo) / (hi - lo)) * ih, bw = iw / bars.length * 0.56, s = svg(W, H, "P&L waterfall");
  niceTicks(lo, hi, 4).forEach(v => { el(s, "line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--line)" });
    el(s, "text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }).textContent = "$" + num(v, 0); });
  bars.forEach((b, i) => {
    const cx = m.l + (i + 0.5) * iw / bars.length, top = y(Math.max(b.from, b.to)), bot = y(Math.max(lo, Math.min(b.from, b.to)));
    const fill = b.k === "total" ? "var(--blue)" : b.v >= 0 ? "var(--up)" : "var(--down)";
    const r = el(s, "rect", { x: cx - bw / 2, y: top, width: bw, height: Math.max(2, bot - top), rx: 3, fill, opacity: b.k === "total" ? .85 : 1 });
    el(s, "text", { x: cx, y: top - 5, "text-anchor": "middle" }).textContent = b.k === "total" ? usd(b.v, 0) : sUsd(b.v, b.v && Math.abs(b.v) < 10 ? 2 : 0);
    el(s, "text", { x: cx, y: H - 8, "text-anchor": "middle" }).textContent = b.n;
    r.addEventListener("pointermove", e => showTip(e, `<div class="h">${b.n}</div><div class="r"><span>${b.k === "total" ? "value" : "change"}</span><b>${b.k === "total" ? usd(b.v) : sUsd(b.v)}</b></div>`));
    r.addEventListener("pointerleave", hideTip);
  });
  box.appendChild(s);
}

function candleChart(box, bars) {
  box.innerHTML = "";
  if (!bars.length) { box.innerHTML = `<div class="empty">No candles</div>`; return; }
  const W = 1100, H = 380, m = { l: 70, r: 12, t: 10, b: 24 }, volH = 70, ih = H - m.t - m.b - volH - 8, iw = W - m.l - m.r;
  const lo = Math.min(...bars.map(b => b.low)), hi = Math.max(...bars.map(b => b.high)), pad = (hi - lo) * 0.05 || hi * 0.01;
  const y = v => m.t + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * ih, vmax = Math.max(...bars.map(b => b.volume)) || 1;
  const step = iw / bars.length, bw = Math.max(1, step * 0.65), x = i => m.l + (i + 0.5) * step, s = svg(W, H, "Candlestick chart");
  niceTicks(lo - pad, hi + pad, 5).forEach(v => { el(s, "line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--line)" });
    el(s, "text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }).textContent = px(v); });
  const vy0 = m.t + ih + 8 + volH;
  bars.forEach((b, i) => {
    const up = b.close >= b.open, c = up ? "var(--up)" : "var(--down)";
    el(s, "line", { x1: x(i), x2: x(i), y1: y(b.high), y2: y(b.low), stroke: c, "stroke-width": 1 });
    el(s, "rect", { x: x(i) - bw / 2, y: y(Math.max(b.open, b.close)), width: bw, height: Math.max(1, Math.abs(y(b.open) - y(b.close))), fill: c });
    el(s, "rect", { x: x(i) - bw / 2, y: vy0 - b.volume / vmax * volH, width: bw, height: b.volume / vmax * volH, fill: c, opacity: .35 });
  });
  [0, 0.5, 1].forEach(f => { const i = Math.min(bars.length - 1, Math.round(f * (bars.length - 1)));
    el(s, "text", { x: x(i), y: H - 6, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle" }).textContent = utc(bars[i].open_time, true).slice(0, 11); });
  const cross = el(s, "line", { y1: m.t, y2: vy0, stroke: "var(--ink-2)", opacity: 0 });
  const hit = el(s, "rect", { x: m.l, y: m.t, width: iw, height: vy0 - m.t, fill: "transparent" });
  hit.addEventListener("pointermove", e => { const r = s.getBoundingClientRect(), xv = (e.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(bars.length - 1, Math.floor((xv - m.l) / step))), b = bars[i];
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", .4);
    const ch = (b.close - b.open) / b.open;
    showTip(e, `<div class="h">${utc(b.open_time, true)} UTC</div>` + [["open", px(b.open)], ["high", px(b.high)], ["low", px(b.low)], ["close", px(b.close)],
      ["change", `<span class="${cls(ch)}">${sPct(ch)}</span>`], ["volume", num(b.volume, 0)]].map(([k, v]) => `<div class="r"><span>${k}</span><b>${v}</b></div>`).join("")); });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
  box.appendChild(s);
}

/* ---------- workflow ---------- */
function renderWorkflow(d) {
  const b = d.banner, n = b.issues.length;
  setHTML($("banner"), `<span class="icon" style="background:var(--${b.status === "ok" ? "ok" : b.status})">${b.status === "ok" ? "✓" : "!"}</span>
    <span class="title">${b.status === "ok" ? "Everything is running normally" : `${n} item${n > 1 ? "s" : ""} need${n > 1 ? "" : "s"} attention`}</span>
    <span class="issues">${b.issues.map(i => `<span class="issue">${esc(i.stage.replace(/^\d+ · /, ""))} · ${esc(i.check)} = ${esc(i.value)}</span>`).join("")}</span>
    <span class="when">checked ${utc(d.checked)} UTC</span>`);
  $("banner").className = "banner " + (b.status === "ok" ? "ok" : b.status);
  const e = d.engine || {};
  $("model").textContent = e.model || "–";
  $("model-sub").textContent = e.model_train_end ? `ridge model trained to ${e.model_train_end}` : "model not trained yet";
  $("lat").textContent = e.latency_ms != null ? Math.round(e.latency_ms) + " ms" : "–";
  const ld = d.last_decision;
  $("ld-sub").textContent = ld ? ld.reason : "no decision yet";
  $("ld-sub").title = ld ? ld.reason : "";
  if (!S.stage || !d.stages.find(s => s.key === S.stage)) {
    const worst = d.stages.find(s => s.status === "fail") || d.stages.find(s => s.status === "warn") || d.stages.find(s => s.key === "strategy") || d.stages[0];
    S.stage = worst.key;
  }
  setHTML($("pipeline"), d.stages.map((s, i) => {
    const live = s.checks.filter(c => c.status !== "idle"), show = (live.length ? live : s.checks).slice(0, 3);
    return `<button type="button" class="stage ${s.status}${s.key === S.stage ? " sel" : ""}" data-k="${s.key}" title="${esc(s.what)}">
      <span class="n">${String(i + 1).padStart(2, "0")}</span><span class="t">${esc(s.title.replace(/^\d+ · /, ""))}</span>
      <span class="pill ${s.status}">${LABEL[s.status]}</span>
      <span class="kv">${show.map(c => `<div><span>${esc(c.name)}</span><span>${esc(c.value)}</span></div>`).join("")}</span></button>`;
  }).join(""));
  renderDetail();
  renderFeed();
}
function renderDetail() {
  const st = S.wf && S.wf.stages.find(s => s.key === S.stage); if (!st) return;
  $("detail-title").textContent = st.title.replace(/^\d+ · /, "") + " · details";
  setHTML($("detail"), `<p class="hint" style="margin:0 0 6px">${esc(st.what)}</p><div class="checks">` + st.checks.map(c =>
    `<div class="row"><span class="name"><span class="dot ${c.status}"></span>${esc(c.name)}</span><span class="val">${esc(c.value)}</span>${c.detail ? `<span class="det">${esc(c.detail)}</span>` : ""}</div>`).join("") + "</div>");
}
function renderFeed() {
  const rows = (S.wf?.activity || []).filter(a => S.kinds.has(a.kind)).slice(0, 60);
  const today = new Date().toISOString().slice(0, 10);
  setHTML($("feed"), rows.length ? rows.map(a => `<div class="ev"><span class="time">${(a.time || "").slice(0, 10) === today ? utc(a.time) : utc(a.time, true).slice(0, 11)}</span>
    <span class="kind k-${a.kind}">${a.kind}</span><span><span class="what">${esc(a.what)}</span><span class="det" title="${esc(a.detail)}">${esc(a.detail)}</span></span></div>`).join("")
    : `<div class="empty">No events for the selected kinds</div>`);
}
$("pipeline").addEventListener("click", e => { const b = e.target.closest(".stage"); if (!b) return; S.stage = b.dataset.k;
  document.querySelectorAll(".stage").forEach(x => x.classList.toggle("sel", x.dataset.k === S.stage)); $("pipeline")._h = null; renderDetail(); });
setHTML($("kinds"), KINDS.map(k => `<button type="button" class="chip" data-k="${k}" aria-pressed="true">${k}</button>`).join(""));
$("kinds").addEventListener("click", e => { const b = e.target.closest(".chip"); if (!b) return; const k = b.dataset.k;
  S.kinds.has(k) ? S.kinds.delete(k) : S.kinds.add(k); b.setAttribute("aria-pressed", S.kinds.has(k)); renderFeed(); });

/* ---------- ledger ---------- */
function kpi(label, value, sub = "", klass = "") {
  return `<div class="metric"><span class="label">${label}</span><span class="value ${klass}">${value}</span><span class="sub">${sub}</span></div>`;
}
function renderLedger(d) {
  if (d.empty) {
    setHTML($("kpis"), kpi("Paper account", "Not started", "the engine writes its first equity mark when it starts"));
    ["equity-chart", "waterfall", "positions", "bycoin", "ledger-table"].forEach(id => setHTML($(id), `<div class="empty">No paper trading data yet</div>`));
    return;
  }
  const k = d.kpi, w = d.waterfall, started = d.account.started_at ? utc(d.account.started_at, true).slice(0, 5) : "";
  setHTML($("kpis"), [
    kpi("Equity", usd(k.equity), `started ${usd(k.start, 0)}${started ? " on " + started : ""}`),
    kpi("Total P&L", `<span class="${cls(w.total)}">${sUsd(w.total)}</span>`, `<span class="${cls(w.total)}">${sPct(w.total / k.start)}</span> of start`),
    kpi("Drawdown", pct(k.drawdown), `peak ${usd(k.peak)} · kill switch at −30%`, k.drawdown > 0.15 ? "neg" : ""),
    kpi("Gross leverage", num(k.gross_lev, 2) + "×", `net ${num(k.net_lev, 2)}× · cap 3×`),
    kpi("Unrealized", `<span class="${cls(k.unrealized)}">${sUsd(k.unrealized)}</span>`, `cash ${usd(k.cash)}`),
    kpi("Costs so far", usd(w.fees + Math.max(0, w.funding)), `fees ${usd(w.fees)} · funding ${w.funding >= 0 ? "paid " + usd(w.funding) : "received " + usd(-w.funding)}`),
  ].join(""));
  $("eq-hint").textContent = `marked every minute · as of ${utc(d.as_of)} UTC`;
  const key = JSON.stringify([d.curve.length, d.curve.at(-1)]);
  if ($("equity-chart")._k !== key) { $("equity-chart")._k = key;
    lineChart($("equity-chart"), d.curve.map(p => ({ t: p.t, v: p.equity })), { base: k.start, fmt: v => "$" + num(v, 0), label: "Equity curve" }); }
  const wkey = JSON.stringify(w);
  if ($("waterfall")._k !== wkey) { $("waterfall")._k = wkey; waterfallChart($("waterfall"), w); }
  const f = d.fills_summary, fu = d.funding_summary;
  setHTML($("costs"), `<div>Fees paid<b>${usd(f.fees)}</b>${f.count} fills · maker share ${f.maker_share != null ? pct(f.maker_share, 0) : "–"} (model 60%)</div>
    <div>Funding<b class="${cls(-(fu.paid - fu.received))}">${sUsd(fu.received - fu.paid)}</b>paid ${usd(fu.paid)} · received ${usd(fu.received)} · ${fu.count} payments</div>`);
  $("pos-hint").textContent = d.positions.length ? `${d.positions.length} coins · marks from the live mark price` : "";
  const maxW = Math.max(0.0001, ...d.positions.map(p => Math.abs(p.weight)));
  setHTML($("positions"), table([
    { k: "symbol", h: "Coin", l: 1, f: v => `<b>${esc(v.replace("USDT", ""))}</b>` },
    { k: "qty", h: "Side", f: v => v > 0 ? `<span class="pos">LONG</span>` : `<span class="neg">SHORT</span>` },
    { k: "qty", h: "Quantity", f: v => qty(Math.abs(v)), c: "m" }, { k: "entry", h: "Entry", f: px, c: "m" }, { k: "mark", h: "Mark", f: px, c: "m" },
    { k: "notional", h: "Notional", f: v => usd(Math.abs(v)), c: "m" },
    { k: "unrealized", h: "Unrealized", f: v => sUsd(v), c: v => "m " + cls(v) },
    { k: "weight", h: "Weight", f: v => `<div style="display:flex;align-items:center;gap:8px;justify-content:flex-end"><span class="m">${sPct(v, 1)}</span><span class="bar"><i style="${v >= 0 ? "left:50%" : "right:50%"};width:${Math.abs(v) / maxW * 50}%;background:var(--${v >= 0 ? "up" : "down"})"></i></span></div>` },
  ], d.positions, "No open positions (waiting for the next 72-hour rebalance)"));
  const maxN = Math.max(0.01, ...d.by_coin.map(r => Math.abs(r.net)));
  setHTML($("bycoin"), table([
    { k: "symbol", h: "Coin", l: 1, f: v => `<b>${esc(v.replace("USDT", ""))}</b>` },
    { k: "realized", h: "Realized", f: sUsd, c: v => "m " + cls(v) }, { k: "unrealized", h: "Unrealized", f: sUsd, c: v => "m " + cls(v) },
    { k: "fees", h: "Fees", f: v => v ? "−" + usd(v) : "–", c: "m neg" }, { k: "funding", h: "Funding", f: v => v ? sUsd(-v) : "–", c: v => "m " + cls(-v) },
    { k: "net", h: "Net", f: v => `<b>${sUsd(v)}</b>`, c: v => "m " + cls(v) },
    { k: "net", h: "", f: v => `<span class="bar"><i style="${v >= 0 ? "left:50%" : "right:50%"};width:${Math.abs(v) / maxN * 50}%;background:var(--${v >= 0 ? "up" : "down"})"></i></span>` },
  ], d.by_coin.slice().reverse(), "No trades yet"));
  renderLedgerTab();
}
function renderLedgerTab() {
  const d = S.lg; if (!d || d.empty) return;
  const side = v => v > 0 ? `<span class="pos">BUY</span>` : `<span class="neg">SELL</span>`;
  const T = {
    decisions: [d.decisions, [{ k: "ts", h: "Time (UTC)", l: 1, f: v => utc(v, true), c: "m" }, { k: "action", h: "Action", l: 1, f: v => `<b>${esc(v)}</b>` },
      { k: "reason", h: "Reason", l: 1, c: "wrap" }, { k: "equity", h: "Equity", f: v => usd(v), c: "m" }], "Every hourly check: HOLD between rebalances, REBALANCE every 72h, NO_TRADE when a safety check blocks it"],
    orders: [d.orders, [{ k: "ts", h: "Time (UTC)", l: 1, f: v => utc(v, true), c: "m" }, { k: "symbol", h: "Coin", l: 1 }, { k: "kind", h: "Type", l: 1 },
      { k: "side", h: "Side", f: side }, { k: "qty", h: "Qty", f: qty, c: "m" }, { k: "price", h: "Limit", f: v => v == null ? "market" : px(v), c: "m" },
      { k: "filled", h: "Filled", f: qty, c: "m" }, { k: "status", h: "Status", l: 1 }], "Maker orders rest 20 min at the best price; leftovers go taker"],
    fills: [d.fills, [{ k: "ts", h: "Time (UTC)", l: 1, f: v => utc(v, true), c: "m" }, { k: "symbol", h: "Coin", l: 1 }, { k: "side", h: "Side", f: side },
      { k: "qty", h: "Qty", f: qty, c: "m" }, { k: "price", h: "Price", f: px, c: "m" }, { k: "liquidity", h: "Liquidity", l: 1 },
      { k: "fee", h: "Fee", f: v => usd(v, 4), c: "m" }, { k: "slippage_bps", h: "Slippage", f: v => v == null ? "–" : num(v, 2) + " bps", c: "m" },
      { k: "latency_ms", h: "Latency", f: v => v == null ? "–" : Math.round(v) + " ms", c: "m" }], `Fees: 0.02% maker · 0.05% taker. Average slippage ${d.fills_summary.avg_slip_bps != null ? num(d.fills_summary.avg_slip_bps, 2) + " bps" : "–"}`],
    funding: [d.funding, [{ k: "ts", h: "Settlement (UTC)", l: 1, f: v => utc(v, true), c: "m" }, { k: "symbol", h: "Coin", l: 1 },
      { k: "qty", h: "Position", f: v => qty(v), c: "m" }, { k: "mark", h: "Mark", f: px, c: "m" }, { k: "rate", h: "Rate", f: v => pct(v, 4), c: "m" },
      { k: "amount", h: "Paid / received", f: v => v >= 0 ? `<span class="neg">paid ${usd(v, 4)}</span>` : `<span class="pos">received ${usd(-v, 4)}</span>`, c: "m" }],
      "Funding settles every 8 hours: longs pay when the rate is positive, shorts receive"],
  };
  const [rows, cols, hint] = T[S.tab];
  $("tab-hint").textContent = hint;
  setHTML($("ledger-table"), table(cols, rows));
}
$("ledger-tabs").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.tab = b.dataset.t;
  document.querySelectorAll("#ledger-tabs button").forEach(x => x.setAttribute("aria-selected", x === b)); renderLedgerTab(); });

/* ---------- market ---------- */
function renderMarket(d) {
  const h = d.heavy || {};
  setHTML($("mkpis"), [
    kpi("Streams live", `${d.streams_live}/${d.streams_expected}`, "closed candles arriving on time", d.streams_live < d.streams_expected ? "neg" : ""),
    kpi("Missing candles · 7d", num(h.missing_7d ?? 0, 0), "gaps the collector could not repair"),
    kpi("OHLC violations · 24h", num((h.ohlc_violations || []).length, 0), "high below low and similar"),
    kpi("1m → 5m mismatches · 24h", num((h.resample_mismatch || []).length, 0), "1m candles that don't add up to 5m"),
    kpi("Candles stored", num(d.total_candles, 0), h.computed ? `audit computed ${utc(h.computed)} UTC · every 5 min` : ""),
  ].join(""));
  const sSel = $("c-symbol"), iSel = $("c-interval");
  if (!sSel.options.length) {
    sSel.innerHTML = d.symbols.map(s => `<option${s === "BTCUSDT.P" ? " selected" : ""}>${s}</option>`).join("");
    iSel.innerHTML = d.intervals.map(i => `<option${i === "1h" ? " selected" : ""}>${i}</option>`).join("");
    loadCandles(true);
  }
  const health = d.health.slice().sort((a, b) => (b.stale - a.stale) || a.symbol.localeCompare(b.symbol));
  const stale = health.filter(r => r.stale).length;
  $("streams-hint").textContent = stale ? `${stale} overdue` : "all on time";
  setHTML($("streams"), table([
    { k: "stale", h: "", f: v => `<span class="dot ${v ? "fail" : "ok"}"></span>` }, { k: "symbol", h: "Symbol", l: 1 }, { k: "interval", h: "Interval", l: 1 },
    { k: "last_close", h: "Last close", f: v => ago(since(v)) + " ago", c: "m" }, { k: "bars", h: "Bars", f: v => num(v, 0), c: "m" },
    { k: "first_open", h: "Since", f: v => (v || "").slice(0, 10), c: "m" }], health));
  const feeds = h.feeds || [];
  $("feeds-hint").textContent = h.computed ? "row counts refresh every 5 min" : "";
  setHTML($("feeds"), table([{ k: "feed", h: "Feed", l: 1 }, { k: "symbol", h: "Symbol", l: 1 },
    { k: "last_ts", h: "Latest", f: v => ago(since(v)) + " ago", c: "m" }, { k: "rows", h: "Rows", f: v => num(v, 0), c: "m" }], feeds, "Computing… (runs at most every 5 minutes)"));
  setHTML($("wsev"), table([{ k: "stream", h: "Stream", l: 1 }, { k: "status", h: "Status", l: 1 }, { k: "events", h: "Events", f: v => num(v, 0), c: "m" },
    { k: "last_event", h: "Last", f: v => utc(v, true), c: "m" }], d.ws_events, "No connection events in 24 hours"));
  setHTML($("fetches"), table([{ k: "fetched_at", h: "Time (UTC)", l: 1, f: v => utc(v), c: "m" }, { k: "kind", h: "Kind", l: 1 },
    { k: "symbol", h: "Symbol", l: 1 }, { k: "interval", h: "Int", l: 1 }, { k: "rows", h: "Rows", c: "m" },
    { k: "status", h: "Status", l: 1, f: v => v === "error" ? `<span class="neg">error</span>` : esc(v) }], d.fetches));
}
async function loadCandles(force) {
  const sym = $("c-symbol").value, iv = $("c-interval").value, bars = $("c-bars").value; if (!sym) return;
  const key = `${sym}|${iv}|${bars}`;
  if (!force && key === S.candleKey && Date.now() - S.candleAt < 30000) return;
  S.candleKey = key; S.candleAt = Date.now();
  try {
    const d = await get(`/api/candles?symbol=${encodeURIComponent(sym)}&interval=${encodeURIComponent(iv)}&bars=${bars}`);
    candleChart($("candles"), d.bars);
    lineChart($("oi"), d.oi.map(p => ({ t: p.ts, v: p.v })), { h: 140, fmt: v => "$" + (v / 1e6).toFixed(0) + "M", label: "Open interest", color: "var(--blue)" });
    lineChart($("fr"), d.funding.map(p => ({ t: p.ts, v: p.v })), { h: 140, fmt: v => (v * 100).toFixed(4) + "%", label: "Funding rate", color: "var(--warn)", area: false });
  } catch (e) { $("candles").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
}
["c-symbol", "c-interval", "c-bars"].forEach(id => $(id).addEventListener("change", () => loadCandles(true)));

/* ---------- polling, routing, clocks ---------- */
async function get(url) { const r = await fetch(url, { cache: "no-store" }); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); }
const LOAD = {
  workflow: async () => { S.wf = await get("/api/workflow"); renderWorkflow(S.wf); },
  ledger: async () => { S.lg = await get("/api/ledger"); renderLedger(S.lg); },
  market: async () => { S.mk = await get("/api/market"); renderMarket(S.mk); loadCandles(false); },
};
async function tick() {
  if (document.hidden || S.busy) return;
  S.busy = true;
  try { await LOAD[S.page](); S.okAt = Date.now(); S.err = null; } catch (e) { S.err = e.message; }
  finally { S.busy = false; clock(); }
}
let timer = null;
function route() {
  const page = (location.hash.slice(1) || "workflow");
  S.page = POLL[page] ? page : "workflow";
  document.querySelectorAll(".page").forEach(p => p.hidden = p.id !== "page-" + S.page);
  document.querySelectorAll(".tabs a").forEach(a => a.classList.toggle("active", a.dataset.page === S.page));
  clearInterval(timer); tick(); timer = setInterval(tick, POLL[S.page]);
}
function clock() {
  const live = $("live");
  if (S.err) { live.className = "live off"; $("live-text").textContent = "offline · retrying"; live.title = S.err; }
  else if (S.okAt) { live.className = "live on"; $("live-text").textContent = `live · ${ago((Date.now() - S.okAt) / 1000)} ago · every ${POLL[S.page] / 1000}s`; live.title = ""; }
  const e = S.wf?.engine; if (!e) return;
  const hb = since(e.heartbeat), hm = $("hb").closest(".metric");
  $("hb").textContent = e.heartbeat ? ago(hb) + " ago" : "never";
  hm.className = "metric " + (!e.heartbeat ? "fail" : hb < 150 ? "ok" : hb < 600 ? "warn" : "fail");
  $("hb-sub").textContent = e.heartbeat ? `last at ${utc(e.heartbeat)} UTC · every minute` : "engine has not started";
  const nr = e.next_rebalance_decision ? (tsMs(e.next_rebalance_decision) - Date.now()) / 1000 : null;
  $("nr").textContent = nr == null ? "–" : nr > 0 ? countdown(nr) : "due now";
  $("nr-sub").textContent = e.next_rebalance_decision ? `${utc(e.next_rebalance_decision, true)} UTC · every ${e.rebalance_hours || 72}h` : "every 72 hours";
  const ld = S.wf.last_decision;
  $("ld").innerHTML = ld ? `${esc(ld.action)} <span class="mut" style="font-size:13px">${ago(since(ld.ts))} ago</span>` : "–";
}
addEventListener("hashchange", route);
document.addEventListener("visibilitychange", () => { if (!document.hidden) tick(); });
setInterval(clock, 1000);

/* theme toggle (remembered per browser) */
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }
$("theme").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = cur === "light" ? "dark" : "light"; document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked */ }
  ["equity-chart", "waterfall"].forEach(id => $(id)._k = null); if (S.page === "ledger" && S.lg) renderLedger(S.lg);
});
route();
