"""Live workflow: the whole pipeline A to Z with a health light per stage and a live activity feed (refresh 5s)."""

import html
from datetime import datetime, timezone

import psycopg
import streamlit as st

from alpha.config import get_settings
from alpha.live.workflow import FAIL, IDLE, OK, WARN, activity, stages

st.set_page_config(page_title="Alpha Centure · Live Workflow", layout="wide")
settings = get_settings()

COLOR = {OK: "#1a9e77", WARN: "#d99a00", FAIL: "#d6455d", IDLE: "#8a8f98"}
LABEL = {OK: "OK", WARN: "CHECK", FAIL: "PROBLEM", IDLE: "WAITING"}
KIND_COLOR = {"decision": "#5b6ee1", "order": "#d99a00", "fill": "#1a9e77", "funding": "#9b59b6",
              "connection": "#8a8f98", "error": "#d6455d"}

CSS = """
<style>
.wf-banner{border-radius:10px;padding:14px 18px;margin:4px 0 14px;font-size:1.05rem;font-weight:600;
  border:1px solid rgba(128,128,128,.25)}
.wf-flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:6px;margin-bottom:8px}
.wf-card{flex:1 1 150px;min-width:150px;border:1px solid rgba(128,128,128,.25);border-top:5px solid;
  border-radius:10px;padding:10px 12px;background:rgba(128,128,128,.06)}
.wf-arrow{align-self:center;font-size:1.3rem;opacity:.45;padding:0 2px}
.wf-title{font-weight:700;font-size:.95rem;margin-bottom:2px}
.wf-what{font-size:.75rem;opacity:.7;line-height:1.25;min-height:2.5em;margin-bottom:6px}
.wf-pill{display:inline-block;font-size:.7rem;font-weight:700;color:#fff;border-radius:999px;padding:1px 8px;
  margin-bottom:6px}
.wf-kv{font-size:.78rem;line-height:1.45}
.wf-kv b{font-weight:600}
.wf-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.wf-feed{border-left:2px solid rgba(128,128,128,.3);margin-left:6px;padding-left:12px}
.wf-ev{margin:0 0 9px 0;font-size:.82rem;line-height:1.35}
.wf-ev .t{opacity:.6;font-size:.72rem}
.wf-ev .k{display:inline-block;color:#fff;font-size:.66rem;font-weight:700;border-radius:4px;padding:0 6px;
  margin-right:6px;text-transform:uppercase}
.wf-ev .d{opacity:.75}
.wf-checks td{padding:3px 10px 3px 0;font-size:.82rem;vertical-align:top}
</style>
"""


@st.cache_resource
def connection() -> psycopg.Connection:
    return psycopg.connect(settings.database_url, autocommit=True)


def conn() -> psycopg.Connection:
    c = connection()
    if c.closed or c.broken:
        connection.clear()
        c = connection()
    return c


def esc(x) -> str:
    return html.escape(str(x))


def card(s) -> str:
    color = COLOR[s.status]
    key = [c for c in s.checks if c.status != IDLE][:3] or s.checks[:2]
    rows = "".join(f'<div class="wf-kv"><span class="wf-dot" style="background:{COLOR[c.status]}"></span>'
                   f'{esc(c.name)}: <b>{esc(c.value)}</b></div>' for c in key)
    return (f'<div class="wf-card" style="border-top-color:{color}">'
            f'<div class="wf-title">{esc(s.title)}</div>'
            f'<span class="wf-pill" style="background:{color}">{LABEL[s.status]}</span>'
            f'<div class="wf-what">{esc(s.what)}</div>{rows}</div>')


st.title("Live workflow · A to Z")
st.caption("Binance → Collector → Database → Strategy R1 → Risk → Execution (paper) → Ledger. "
           "Every light is computed from live data; the page refreshes every 5 seconds. No real orders are ever sent.")


@st.fragment(run_every="5s")
def live() -> None:
    c = conn()
    try:
        ss = stages(c, settings.symbols)
        feed = activity(c, 60)
    except Exception as e:  # never blank the page: show what failed
        st.error(f"Could not read the database: {e}")
        return
    problems = [(s, ch) for s in ss for ch in s.checks if ch.status in (WARN, FAIL)]
    worst = FAIL if any(ch.status == FAIL for _, ch in problems) else WARN if problems else OK
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    if worst == OK:
        msg = f"✅ Everything is running normally · checked {now}"
    else:
        items = "; ".join(f"{s.title.split('·')[1].strip()}: {ch.name} = {ch.value}" for s, ch in problems[:4])
        msg = f"{'⛔' if worst == FAIL else '⚠️'} {len(problems)} item(s) need attention · {items} · checked {now}"
    st.html(CSS + f'<div class="wf-banner" style="background:{COLOR[worst]}1f;border-color:{COLOR[worst]}66">'
            f'{esc(msg)}</div>')
    st.html(CSS + '<div class="wf-flow">' + '<div class="wf-arrow">→</div>'.join(card(s) for s in ss) + "</div>")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Stage details")
        for s in ss:
            with st.expander(f"{s.title} — {LABEL[s.status]}", expanded=s.status in (WARN, FAIL)):
                rows = "".join(
                    f'<tr><td><span class="wf-dot" style="background:{COLOR[ch.status]}"></span>{esc(ch.name)}</td>'
                    f'<td><b>{esc(ch.value)}</b></td><td style="opacity:.65">{esc(ch.detail)}</td></tr>'
                    for ch in s.checks)
                st.html(CSS + f'<div style="font-size:.8rem;opacity:.7;margin-bottom:6px">{esc(s.what)}</div>'
                        f'<table class="wf-checks">{rows}</table>')
    with right:
        st.subheader("Live activity")
        kinds = st.multiselect("Show", list(KIND_COLOR), default=list(KIND_COLOR), label_visibility="collapsed",
                               key="kinds")
        f = feed[feed["kind"].isin(kinds)] if len(feed) else feed
        if f.empty:
            st.info("Nothing yet.")
        else:
            evs = "".join(
                f'<div class="wf-ev"><div class="t">{r.time:%Y-%m-%d %H:%M:%S}</div>'
                f'<span class="k" style="background:{KIND_COLOR.get(r.kind, "#888")}">{esc(r.kind)}</span>'
                f'<b>{esc(r.what)}</b><div class="d">{esc(r.detail)[:160]}</div></div>'
                for r in f.head(40).itertuples())
            st.html(CSS + f'<div class="wf-feed">{evs}</div>')


live()

with st.expander("How the workflow works, step by step", expanded=False):
    st.markdown(f"""
1. **Binance mainnet** — two websocket connections: order book for {len(settings.symbols)} perps on `/public`
   (every 100 ms), trades and mark price/funding on `/market`. Latency is measured on every message.
2. **Collector** (`alpha.main`, always on) — saves closed candles (1m…1d), 1-minute buy/sell flow, the order book
   every second, open interest every minute, funding/premium/long-short every 5 minutes; repairs gaps after any
   disconnect.
3. **Database** (PostgreSQL) — history since 2020 plus everything live, and the paper ledger.
4. **Strategy R1** (`alpha.live.engine`) — every hour at HH:01:30 UTC it checks the schedule. Every **72 hours** it
   builds signals (momentum, trend, funding, basis, open interest, positioning, flow), runs the **ridge forecast** (retrained monthly) and the **momentum rules** and averages the two
   books (each at a 20% volatility target): max 3x gross, max 0.5x per coin, 1% no-trade band.
5. **Risk** — before any order: caps, a **−3% day** turns the engine reduce-only, a **−30% drawdown** flattens
   everything and halts until a manual restart.
6. **Execution (paper)** — each coin gets a maker limit order at the best price for 20 minutes; it fills only when
   real trades pass through that price (after the queue ahead of it). Whatever is left goes **taker**, walking the
   real order book after the measured latency. Fees: 0.02% maker / 0.05% taker.
7. **Ledger & P&L** — every decision, order, fill, fee, funding payment and a mark-to-market equity value every
   minute are stored; the account is rebuilt from them on restart.

**Reading the lights:** green = normal · amber = slower or weaker than normal · red = broken or stopped ·
grey = nothing expected yet (for example, no fills before the first rebalance).
""")
