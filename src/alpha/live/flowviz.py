"""Animated SVG views for the dashboard: the live A-to-Z pipeline and a strategy decision (any StrategyTrace).

Motion is CSS only (dashes marching along each path: st.html strips SVG <animate> tags). Speed follows the live rate on
each edge; a broken edge turns red and stops, an idle one is grey and dashed. Honors prefers-reduced-motion.
"""

import html
import math
from dataclasses import dataclass

import pandas as pd

from alpha.live.workflow import FAIL, IDLE, OK, WARN, Stage
from alpha.strategy.trace import BLEND, BOOK, FINAL, SIGNAL, StrategyTrace, gross, scale_factor

COLOR = {OK: "#1a9e77", WARN: "#d99a00", FAIL: "#d6455d", IDLE: "#8a8f98"}
LONG, SHORT = "#1a9e77", "#d6455d"

CSS = """
<style>
.fv{width:100%;height:auto;font-family:inherit;color:inherit}
.fv text{fill:currentColor}
.fv .node{fill:rgba(128,128,128,.08);stroke-width:2}
.fv .ttl{font-size:13px;font-weight:700}
.fv .big{font-size:16px;font-weight:700}
.fv .sub{font-size:10.5px;opacity:.7}
.fv .lbl{font-size:10.5px;opacity:.8}
.fv .base{fill:none;stroke-opacity:.18}
.fv .flow{fill:none;stroke-linecap:round;animation:fv-march linear infinite}
.fv .still{fill:none;stroke-dasharray:4 6;stroke-opacity:.6}
@keyframes fv-march{to{stroke-dashoffset:-20}}
.fv .pulse{animation:fv-pulse 2.4s ease-in-out infinite}
@keyframes fv-pulse{50%{stroke-opacity:.35}}
.fv .step{opacity:.08;animation:fv-show .7s ease forwards}
@keyframes fv-show{to{opacity:1}}
@media (prefers-reduced-motion: reduce){.fv .flow,.fv .pulse{animation:none}.fv .step{animation:none;opacity:1}}
</style>
"""


def esc(x) -> str:
    return html.escape(str(x))


# ---------------- pure helpers ----------------

@dataclass
class EdgeStyle:
    color: str
    moving: bool
    dur_s: float          # seconds for one dash cycle: smaller = faster
    width: float


def edge_style(rate: float | None, status: str, width: float = 3.0) -> EdgeStyle:
    """Broken -> red and still; nothing flowing -> grey and still; otherwise faster the higher the rate (log scale)."""
    if status == FAIL:
        return EdgeStyle(COLOR[FAIL], False, 0.0, width)
    if not rate or rate <= 0:
        return EdgeStyle(COLOR[IDLE], False, 0.0, width)
    lvl = math.log10(rate + 1)
    return EdgeStyle(COLOR.get(status, COLOR[OK]), True, round(min(2.5, max(0.25, 2.5 / (1 + lvl))), 2), width)


def cycle_progress(next_decision: str | None, hours: int, now: pd.Timestamp | None = None) -> float | None:
    """Share of the rebalance cycle already elapsed (0 just after a rebalance, 1 at the next one)."""
    if not next_decision or not hours:
        return None
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    left_h = (pd.Timestamp(next_decision) - now).total_seconds() / 3600
    return min(1.0, max(0.0, 1 - left_h / hours))


def fmt_rate(x: float | None, unit: str = "/min") -> str:
    if x is None:
        return "–"
    if x >= 1e6:
        return f"{x / 1e6:.1f}M{unit}"
    if x >= 1e3:
        return f"{x / 1e3:.1f}k{unit}"
    return f"{x:.0f}{unit}"


def check_value(stage: Stage | None, name: str, default: str = "–") -> str:
    if stage is None:
        return default
    return next((c.value for c in stage.checks if c.name == name), default)


# ---------------- svg pieces ----------------

def _edge(d: str, st: EdgeStyle, cls: str = "") -> str:
    base = f'<path class="base" d="{d}" stroke="{st.color}" stroke-width="{st.width + 3:.1f}"/>'
    if not st.moving:
        return f'<g class="{cls}">{base}<path class="still" d="{d}" stroke="{st.color}" stroke-width="{st.width:.1f}"/></g>'
    return (f'<g class="{cls}">{base}<path class="flow" d="{d}" stroke="{st.color}" stroke-width="{st.width:.1f}" '
            f'stroke-dasharray="1 19" style="animation-duration:{st.dur_s}s"/></g>')


def _ring(cx: float, cy: float, r: float, frac: float | None, color: str) -> str:
    if frac is None:
        return ""
    c = 2 * math.pi * r
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="currentColor" stroke-opacity=".15" stroke-width="5"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="5" stroke-linecap="round" '
            f'stroke-dasharray="{frac * c:.1f} {c:.1f}" transform="rotate(-90 {cx} {cy})"/>')


# ---------------- live pipeline ----------------

PIPE = ["binance", "collector", "database", "strategy", "risk", "execution", "ledger"]


def pipeline_svg(stages: list[Stage], r: dict) -> str:
    """The 7 stages in a row with live flows between them. r = workflow.rates(conn)."""
    by = {s.key: s for s in stages}
    W, NW, NH, Y = 1240, 132, 104, 92
    gap = (W - 20 - len(PIPE) * NW) / (len(PIPE) - 1)
    x = {k: 10 + i * (NW + gap) for i, k in enumerate(PIPE)}
    status = {k: by[k].status if k in by else IDLE for k in PIPE}
    frac = cycle_progress(r.get("next_decision"), r.get("rebalance_hours") or 0)
    left_h = None
    if r.get("next_decision"):
        left_h = (pd.Timestamp(r["next_decision"]) - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600
    text = {
        "binance": (fmt_rate(r.get("msgs_min")), "book · trades · mark"),
        "collector": (fmt_rate(r.get("rows_min"), " rows/min"), f"{check_value(by.get('collector'), 'Tokens updating')} coins live"),
        "database": (check_value(by.get("database"), "Size"), f"{check_value(by.get('database'), 'Candles (approx.)')} candles"),
        "strategy": (f"{left_h:.1f}h" if left_h is not None else "–", f"to rebalance · {r.get('last_action') or 'no decision'}"),
        "risk": (check_value(by.get("risk"), "Drawdown from peak"), f"kill switch {check_value(by.get('risk'), 'Kill switch')}"),
        "execution": (f"{r.get('open_orders', 0)} open", f"{r.get('fills_30m', 0)} fills · 30 min"),
        "ledger": (check_value(by.get("ledger"), "Equity"), f"marked {check_value(by.get('ledger'), 'Equity marks')}"),
    }
    titles = {"binance": "Binance", "collector": "Collector", "database": "Database", "strategy": r.get("model") or "Strategy",
              "risk": "Risk", "execution": "Paper broker", "ledger": "Ledger & P&L"}

    out = []
    # side branch: the paper broker reads the live order book straight from Binance
    sx, ex = x["binance"] + NW / 2, x["execution"] + NW / 2
    d = f"M{sx},{Y} C{sx},{Y - 70} {ex},{Y - 70} {ex},{Y}"
    out.append(_edge(d, edge_style(r.get("book_min"), status["execution"] if status["strategy"] != FAIL else FAIL, 2.0)))
    out.append(f'<text class="lbl" x="{(sx + ex) / 2}" y="{Y - 56}" text-anchor="middle">live order book → paper fills '
               f'({fmt_rate(r.get("book_min"))})</text>')
    flows = [("binance", "collector", r.get("msgs_min"), "msgs"),
             ("collector", "database", r.get("rows_min"), "rows"),
             ("database", "strategy", 60 if (r.get("decision_age_s") or 1e9) < 120 else 1, "hourly"),
             ("strategy", "risk", r.get("orders_30m"), "targets"),
             ("risk", "execution", r.get("orders_30m"), "orders"),
             ("execution", "ledger", (r.get("fills_30m") or 0) + 1, "fills")]
    for a, b, rate, lab in flows:
        x1, x2, yc = x[a] + NW, x[b], Y + NH / 2
        worst = FAIL if FAIL in (status[a], status[b]) and status[b] != IDLE else status[a]
        out.append(_edge(f"M{x1 + 2},{yc} L{x2 - 2},{yc}", edge_style(rate, worst)))
        out.append(f'<text class="lbl" x="{(x1 + x2) / 2}" y="{yc - 9}" text-anchor="middle">{esc(lab)}</text>')
    for k in PIPE:
        c = COLOR[status[k]]
        tip = "; ".join(f"{ch.name}: {ch.value}" for ch in by[k].checks) if k in by else ""
        big, sub = text[k]
        pulse = ' pulse' if status[k] in (WARN, FAIL) else ""
        out.append(f'<g><title>{esc(tip)}</title>'
                   f'<rect class="node{pulse}" x="{x[k]}" y="{Y}" width="{NW}" height="{NH}" rx="12" stroke="{c}"/>'
                   f'<rect x="{x[k]}" y="{Y}" width="{NW}" height="6" rx="3" fill="{c}"/>'
                   f'<text class="ttl" x="{x[k] + 10}" y="{Y + 26}">{esc(titles[k])}</text>'
                   f'<text class="big" x="{x[k] + 10}" y="{Y + 56}">{esc(big)}</text>'
                   f'<text class="sub" x="{x[k] + 10}" y="{Y + 80}">{esc(sub)[:26]}</text></g>')
        if k == "strategy":
            out.append(_ring(x[k] + NW - 22, Y + 26, 12, frac, c))
    return (CSS + f'<svg class="fv" viewBox="0 0 {W} {Y + NH + 14}" role="img" aria-label="Live pipeline">'
            + "".join(out) + "</svg>")


# ---------------- strategy decision ----------------

def _fmt_w(v: float) -> str:
    return f"{v:+.2f}x"


def strategy_svg(t: StrategyTrace, max_coins: int = 14, nonce: int = 0) -> str:
    """Signals → books → blend → final → coins, revealed step by step, then flowing. Works for any trace."""
    cols: list[list[tuple[str, str, str, str]]] = []   # per column: (key, title, line, color)
    for kind in (SIGNAL, BOOK, BLEND, FINAL):
        col = []
        for x in t.of(kind):
            if kind == SIGNAL:
                top = max(x.values.items(), key=lambda kv: abs(kv[1]), default=None)
                line = f"strongest {top[0].replace('USDT', '')} {top[1]:+.2f}" if top else "no data"
            elif kind == BOOK:
                line = f"gross {gross(x.values):.2f}x" + (f" · mix {x.blend_weight:.0%}" if x.blend_weight is not None else "")
            elif kind == BLEND:
                line = f"gross {gross(x.values):.2f}x"
            else:
                sf = scale_factor(t)
                line = f"gross {gross(t.final):.2f}x" + (f" · scaled ×{sf:.1f}" if sf else "")
            col.append((x.key, x.title, line, COLOR[OK] if kind != SIGNAL else "#5b6ee1"))
        if col:
            cols.append(col)
    coins = [s for s in t.symbols() if abs(t.final.get(s, 0.0)) > 1e-6][:max_coins]
    cols.append([(f"coin:{s}", s.replace("USDT", ""), _fmt_w(t.final[s]), LONG if t.final[s] > 0 else SHORT)
                 for s in coins] or [("coin:none", "No positions", "flat book", COLOR[IDLE])])

    W, NW, NH, VG = 1180, 168, 40, 10
    HEAD = 24
    H = max(len(c) for c in cols) * (NH + VG) + HEAD + 6
    step = (W - 20 - NW) / max(1, len(cols) - 1)
    pos = {}
    for i, col in enumerate(cols):
        top = HEAD + (H - HEAD - len(col) * (NH + VG) + VG) / 2
        for j, (k, *_rest) in enumerate(col):
            pos[k] = (10 + i * step, top + j * (NH + VG), i)

    out = []
    delay = 0.75
    edges = []
    for x in t.layers:
        for k in t.feeds(x):
            w = 2.0
            if x.kind == BLEND:
                b = next(y for y in t.layers if y.key == k)
                w = 1.5 + 4 * min(1.0, gross(b.values) * (b.blend_weight or 1) / max(gross(x.values), 1e-9))
            edges.append((k, x.key, w, COLOR[OK]))
    last = t.of(FINAL) or t.of(BLEND) or t.of(BOOK)
    if last:
        m = max((abs(t.final.get(s, 0)) for s in coins), default=1) or 1
        for s in coins or ["none"]:
            v = t.final.get(s, 0.0)
            edges.append((last[-1].key, f"coin:{s}", 1.2 + 4 * abs(v) / m, LONG if v > 0 else SHORT if v < 0 else COLOR[IDLE]))
    for a, b, w, c in edges:
        if a not in pos or b not in pos:
            continue
        (x1, y1, i), (x2, y2, _) = pos[a], pos[b]
        x1, y1, y2 = x1 + NW, y1 + NH / 2, y2 + NH / 2
        mx = (x1 + x2) / 2
        d = f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}"
        out.append(f'<g class="step" style="animation-delay:{(i + 0.5) * delay:.2f}s">'
                   + _edge(d, EdgeStyle(c, True, 1.2, w)) + "</g>")
    for col in cols:
        for k, title, line, c in col:
            x, y, i = pos[k]
            tip = next((f"{l.title}: {l.description}" for l in t.layers if l.key == k), title)
            out.append(f'<g class="step" style="animation-delay:{i * delay:.2f}s"><title>{esc(tip)}</title>'
                       f'<rect class="node" x="{x}" y="{y}" width="{NW}" height="{NH}" rx="8" stroke="{c}"/>'
                       f'<rect x="{x}" y="{y}" width="5" height="{NH}" rx="2" fill="{c}"/>'
                       f'<text class="ttl" x="{x + 12}" y="{y + 17}" style="font-size:12px">{esc(title)[:24]}</text>'
                       f'<text class="sub" x="{x + 12}" y="{y + 32}">{esc(line)[:30]}</text></g>')
    heads = {SIGNAL: "1 · Signals", BOOK: "2 · Sub-strategies", BLEND: "3 · Blend", FINAL: "4 · Risk scaling"}
    kinds = [k for k in (SIGNAL, BOOK, BLEND, FINAL) if t.of(k)] + ["coins"]
    for i, k in enumerate(kinds):
        out.append(f'<text class="lbl" x="{10 + i * step}" y="12" style="font-weight:700">'
                   f'{esc(heads.get(k, f"{len(kinds)} · Positions"))}</text>')
    return (CSS + f'<!-- {nonce} --><svg class="fv" viewBox="0 0 {W} {H}" role="img" aria-label="Strategy decision flow">'
            + "".join(out) + "</svg>")
