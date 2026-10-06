"""Live A-to-Z workflow health for the dashboard: one status per pipeline stage plus a unified activity feed.

    Binance -> Collector -> Database -> Strategy (P6) -> Risk -> Execution (paper broker) -> Ledger

Every check is a cheap query (indexed recent rows or per-feed index lookups) so the page can refresh every few
seconds. Status rules are pure functions (unit tested): OK (green), WARN (amber), FAIL (red), IDLE (grey: nothing
expected yet, e.g. no trades before the first rebalance).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import psycopg

OK, WARN, FAIL, IDLE = "ok", "warn", "fail", "idle"
RANK = {OK: 0, IDLE: 0, WARN: 1, FAIL: 2}


@dataclass
class Check:
    name: str
    status: str
    value: str
    detail: str = ""


@dataclass
class Stage:
    key: str
    title: str
    what: str                                  # one line: what this stage does
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> str:
        live = [c for c in self.checks if c.status != IDLE]
        return max(live, key=lambda c: RANK[c.status]).status if live else IDLE


# ---------------- pure rules ----------------

def by_age(age_s: float | None, ok_s: float, warn_s: float) -> str:
    if age_s is None:
        return FAIL
    return OK if age_s <= ok_s else WARN if age_s <= warn_s else FAIL


def by_level(x: float | None, warn: float, fail: float) -> str:
    """Higher is worse (drawdown, latency, leverage...)."""
    if x is None:
        return IDLE
    return OK if x < warn else WARN if x < fail else FAIL


def disconnect_status(last_15m: int, last_hour: int, data_fresh: bool, last_age_s: float | None) -> tuple[str, str]:
    """Red only while it is still happening (many drops in 15 min and data not flowing); a finished outage is amber
    with the time since it ended, and clears after an hour."""
    if last_15m >= 3 and not data_fresh:
        return FAIL, f"{last_15m} in 15 min, data stopped"
    if last_hour and data_fresh:
        return WARN, f"{last_hour} in 1h · recovered, last {fmt_age(last_age_s)}"
    if last_hour:
        return WARN, f"{last_hour} in 1h"
    return OK, "0"


def fmt_age(age_s: float | None) -> str:
    if age_s is None:
        return "never"
    if age_s < 120:
        return f"{age_s:.0f}s ago"
    if age_s < 7200:
        return f"{age_s / 60:.0f}m ago"
    if age_s < 172800:
        return f"{age_s / 3600:.1f}h ago"
    return f"{age_s / 86400:.1f}d ago"


# ---------------- queries ----------------

def _one(conn, sql: str, params=None):
    r = conn.execute(sql, params or ()).fetchone()
    return r if r else (None,)


def _age(conn, sql: str, params=None) -> float | None:
    v = _one(conn, f"SELECT extract(epoch FROM now() - ({sql}))", params)[0]
    return float(v) if v is not None else None


def _state(conn, key: str) -> dict:
    return _one(conn, "SELECT value FROM paper_state WHERE key = %s", (key,))[0] or {}


def stages(conn: psycopg.Connection, symbols: list[str]) -> list[Stage]:
    n = len(symbols)
    perp = [s + ".P" for s in symbols]
    out = []

    # 1. Binance -> websockets
    lat = conn.execute("""SELECT stream, max(p95_ms), extract(epoch FROM now() - max(minute)) FROM ws_latency
                          WHERE minute > now() - interval '10 minutes' GROUP BY 1""").fetchall()
    lat = {r[0]: (float(r[1]), float(r[2])) for r in lat}
    tick_age = _age(conn, "SELECT max(ts) FROM book_tick")
    disc = _one(conn, """SELECT count(*), count(*) FILTER (WHERE fetched_at > now() - interval '15 minutes'),
                                extract(epoch FROM now() - max(fetched_at))
                         FROM fetch_log WHERE kind = 'ws' AND status = 'disconnected'
                         AND fetched_at > now() - interval '1 hour'""")
    st = Stage("binance", "1 · Binance mainnet", "Live websockets: order book (/public), trades + mark price (/market)")
    st.checks.append(Check("Order book stream", by_age(tick_age, 15, 60), fmt_age(tick_age), "1s top-of-book sample"))
    for k, label in (("depth20", "Book latency p95"), ("aggtrade", "Trade latency p95"), ("markprice", "Mark latency p95")):
        v = lat.get(k)
        st.checks.append(Check(label, by_level(v[0], 500, 2000) if v else FAIL,
                               f"{v[0]:.0f} ms" if v else "no data", "exchange event → received"))
    d_status, d_text = disconnect_status(int(disc[1] or 0), int(disc[0] or 0), by_age(tick_age, 15, 60) == OK,
                                         float(disc[2]) if disc[2] is not None else None)
    st.checks.append(Check("Disconnects", d_status, d_text, "red only while an outage is ongoing"))
    out.append(st)

    # 2. Collector -> database
    st = Stage("collector", "2 · Collector", f"Writes candles, flow, book, funding, OI for {n} perps (alpha.main)")
    c1 = _age(conn, "SELECT max(close_time) FROM candles WHERE symbol = 'BTCUSDT.P' AND interval = '1m'")
    c1h = _age(conn, "SELECT max(close_time) FROM candles WHERE symbol = 'BTCUSDT.P' AND interval = '1h'")
    fl = _age(conn, "SELECT max(minute) FROM flow_1m WHERE symbol = 'BTCUSDT.P'")
    oi = _age(conn, "SELECT max(ts) FROM open_interest_live")
    fr = _age(conn, "SELECT max(ts) FROM premium_snap")
    fm = _age(conn, "SELECT max(ts) FROM futures_metrics WHERE symbol = 'BTCUSDT'")
    fresh = _one(conn, """SELECT count(DISTINCT symbol) FROM candles WHERE interval = '1m'
                          AND symbol = ANY(%s) AND close_time > now() - interval '3 minutes'""", (perp,))[0]
    errs = _one(conn, """SELECT count(*) FROM fetch_log WHERE status = 'error' AND fetched_at > now() - interval '1 hour'""")[0]
    st.checks += [
        Check("1m candles", by_age(c1, 150, 600), fmt_age(c1)),
        Check("Tokens updating", OK if fresh == n else WARN if fresh >= n - 2 else FAIL, f"{fresh}/{n}"),
        Check("1h candles", by_age(c1h, 3900, 7500), fmt_age(c1h), "strategy input"),
        Check("Order flow (1m)", by_age(fl, 150, 600), fmt_age(fl)),
        Check("Open interest (live)", by_age(oi, 150, 600), fmt_age(oi)),
        Check("Mark / funding", by_age(fr, 150, 600), fmt_age(fr)),
        Check("OI / long-short (5m)", by_age(fm, 900, 1800), fmt_age(fm), "strategy input"),
        Check("Write errors (1h)", OK if errs == 0 else WARN if errs < 10 else FAIL, str(errs)),
    ]
    out.append(st)

    # 3. Database
    st = Stage("database", "3 · Database", "PostgreSQL: market history + live data + paper ledger")
    size = _one(conn, "SELECT pg_database_size(current_database())")[0]
    rows = dict(conn.execute("""SELECT relname, reltuples::bigint FROM pg_class
                                WHERE relname IN ('candles', 'book_tick', 'flow_1m', 'paper_fills')""").fetchall())
    st.checks += [Check("Size", OK, f"{size / 1e9:.1f} GB"),
                  Check("Candles (approx.)", OK, f"{max(rows.get('candles', 0), 0):,}"),
                  Check("Book ticks (approx.)", OK, f"{max(rows.get('book_tick', 0), 0):,}")]
    out.append(st)

    # 4. Strategy
    eng = _state(conn, "engine")
    hb = None
    if eng.get("heartbeat"):
        hb = (datetime.now(timezone.utc) - datetime.fromisoformat(eng["heartbeat"])).total_seconds()
    dec = _one(conn, "SELECT action, reason, extract(epoch FROM now() - ts), bar_time FROM paper_decisions ORDER BY id DESC LIMIT 1")
    nxt = eng.get("next_rebalance_decision")
    to_next = (pd.Timestamp(nxt) - pd.Timestamp.now(tz="UTC")).total_seconds() if nxt else None
    model_age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(eng["model_train_end"], tz="UTC")).days if eng.get("model_train_end") else None
    st = Stage("strategy", "4 · Strategy P6", "Every hour: check schedule. Every 72h: signals → ridge forecast → targets")
    st.checks += [
        Check("Engine heartbeat", by_age(hb, 150, 600), fmt_age(hb), "alpha.live.engine"),
        Check("Last decision", by_age(dec[2], 3900, 7500) if dec[0] else IDLE,
              f"{dec[0]} · {fmt_age(dec[2])}" if dec[0] else "none yet", (dec[1] or "")[:90] if dec[0] else ""),
        Check("Next rebalance", OK if to_next is not None else IDLE,
              f"in {to_next / 3600:.1f}h" if to_next is not None else "unknown", str(nxt or "")[:16] + " UTC"),
        Check("Model trained up to", OK if model_age is not None and model_age < 40 else WARN,
              str(eng.get("model_train_end") or "unknown"), "retrains monthly"),
    ]
    out.append(st)

    # 5. Risk
    risk = _state(conn, "risk")
    acct = _state(conn, "account")
    eq = _one(conn, "SELECT equity, gross_lev, margin_ratio FROM paper_equity ORDER BY ts DESC LIMIT 1")
    st = Stage("risk", "5 · Risk", "Caps 3x gross / 0.5x per coin · −3% day: reduce only · −30%: flatten + halt")
    if eq[0] is not None:
        peak = float(risk.get("peak_equity") or eq[0])
        dd = 1 - float(eq[0]) / peak if peak else 0.0
        dstart = float(risk.get("day_start_equity") or eq[0])
        dloss = max(0.0, 1 - float(eq[0]) / dstart) if dstart else 0.0
        st.checks += [
            Check("Kill switch", FAIL if risk.get("halted") else OK, "HALTED" if risk.get("halted") else "armed",
                  risk.get("reason", "") or "flatten at −30% from peak"),
            Check("Drawdown from peak", by_level(dd, 0.15, 0.25), f"{dd:.2%}", "kill at 30%"),
            Check("Loss today", by_level(dloss, 0.02, 0.03), f"{dloss:.2%}", "reduce-only at 3%"),
            Check("Gross leverage", by_level(float(eq[1]), 2.5, 3.01), f"{float(eq[1]):.2f}x", "cap 3x"),
            Check("Margin ratio", by_level(float(eq[2]), 0.3, 0.5), f"{float(eq[2]):.1%}", "liquidation at 100%"),
        ]
    out.append(st)

    # 6. Execution
    st = Stage("execution", "6 · Execution (paper)", "Maker limit 20 min at best price → leftover taker on the real book")
    stuck = _one(conn, "SELECT count(*) FROM paper_orders WHERE status = 'open' AND ts < now() - interval '25 minutes'")[0]
    f = _one(conn, """SELECT count(*), sum(qty * price) FILTER (WHERE liquidity = 'maker') / nullif(sum(qty * price), 0),
                             avg(slippage_bps), avg(latency_ms), count(*) FILTER (WHERE beyond_book)
                      FROM paper_fills""")
    if f[0]:
        st.checks += [
            Check("Fills", OK, f"{f[0]:,}"),
            Check("Maker share", OK if f[1] is not None and 0.48 <= float(f[1]) <= 0.72 else WARN,
                  f"{float(f[1] or 0):.0%}", "model assumes 60%"),
            Check("Avg slippage vs mid", by_level(float(f[2] or 0), 3, 8), f"{float(f[2] or 0):.2f} bps"),
            Check("Taker latency used", OK, f"{float(f[3] or 0):.0f} ms"),
            Check("Beyond-book fills", OK if not f[4] else FAIL, str(f[4])),
        ]
    else:
        st.checks.append(Check("Fills", IDLE, "none yet", "first trades at the first rebalance"))
    st.checks.append(Check("Stuck orders", OK if not stuck else FAIL, str(stuck), "open > 25 min"))
    out.append(st)

    # 7. Ledger
    st = Stage("ledger", "7 · Ledger & P&L", "Every order, fill, fee, funding payment and equity mark is recorded")
    if eq[0] is not None and acct:
        start = float(acct.get("start_equity", 0) or 0)
        tot = _one(conn, "SELECT coalesce(sum(fee), 0) FROM paper_fills")[0]
        fund = _one(conn, "SELECT coalesce(sum(amount), 0) FROM paper_funding")[0]
        mark_age = _age(conn, "SELECT max(ts) FROM paper_equity")
        st.checks += [
            Check("Equity", OK, f"${float(eq[0]):,.2f}", f"start ${start:,.0f} · {float(eq[0]) / start - 1:+.2%}" if start else ""),
            Check("Equity marks", by_age(mark_age, 150, 600), fmt_age(mark_age), "every minute"),
            Check("Fees paid", OK, f"${float(tot):,.2f}"),
            Check("Funding paid", OK, f"${float(fund):,.2f}", "negative = received"),
        ]
    out.append(st)
    return out


def activity(conn: psycopg.Connection, limit: int = 60) -> pd.DataFrame:
    """Everything that happened, newest first: decisions, orders, fills, funding, connections, errors."""
    cur = conn.execute("""
        (SELECT ts, 'decision' AS kind, action AS what,
                reason || coalesce(' · equity $' || round(equity::numeric, 2), '') AS detail FROM paper_decisions
         ORDER BY id DESC LIMIT %(n)s)
        UNION ALL
        (SELECT ts, 'order', kind || ' ' || CASE WHEN side > 0 THEN 'BUY ' ELSE 'SELL ' END || symbol,
                qty || coalesce(' @ ' || price, ' market') || ' · ' || status FROM paper_orders ORDER BY id DESC LIMIT %(n)s)
        UNION ALL
        (SELECT ts, 'fill', liquidity || ' ' || CASE WHEN side > 0 THEN 'BUY ' ELSE 'SELL ' END || symbol,
                qty || ' @ ' || price || ' · fee $' || round(fee::numeric, 4) || ' · slip ' || round(slippage_bps::numeric, 2) || ' bps'
         FROM paper_fills ORDER BY id DESC LIMIT %(n)s)
        UNION ALL
        (SELECT ts, 'funding', symbol, 'rate ' || rate || ' · paid $' || round(amount::numeric, 4) FROM paper_funding
         ORDER BY id DESC LIMIT %(n)s)
        UNION ALL
        (SELECT fetched_at, 'connection', coalesce(interval, kind) || ' ' || status, coalesce(error, '') FROM fetch_log
         WHERE kind = 'ws' AND fetched_at > now() - interval '7 days' ORDER BY fetched_at DESC LIMIT %(n)s)
        UNION ALL
        (SELECT fetched_at, 'error', kind || ' ' || coalesce(symbol, ''), coalesce(error, '') FROM fetch_log
         WHERE status = 'error' AND fetched_at > now() - interval '7 days' ORDER BY fetched_at DESC LIMIT %(n)s)
        ORDER BY 1 DESC LIMIT %(n)s
    """, {"n": limit})
    return pd.DataFrame(cur.fetchall(), columns=["time", "kind", "what", "detail"])
