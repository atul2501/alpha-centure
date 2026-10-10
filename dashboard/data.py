"""Read-only queries behind the dashboard API. Everything returns plain JSON-ready dicts / lists.

Reuses the workflow checks (alpha.live.workflow), the data audit (alpha.audit) and the paper account model
(alpha.exec.account) unchanged; nothing here writes to the database.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import psycopg

from alpha import audit
from alpha.config import PERP_SUFFIX, Settings
from alpha.exec.account import Account
from alpha.live.workflow import FAIL, RANK, WARN, activity, stages

MAX_POINTS = 1500  # equity curve points sent to the browser


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON records (timestamps as ISO strings, NaN as null)."""
    return json.loads(df.to_json(orient="records", date_format="iso")) if len(df) else []


def _rows(conn, sql: str, params=None) -> list[dict]:
    cur = conn.execute(sql, params or ())
    cols = [d.name for d in cur.description]
    return records(pd.DataFrame(cur.fetchall(), columns=cols))


def _state(conn, key: str) -> dict:
    r = conn.execute("SELECT value FROM paper_state WHERE key = %s", (key,)).fetchone()
    return (r[0] or {}) if r else {}


# ---------------------------------------------------------------------------------------------------------------
# workflow (live)

def workflow(conn: psycopg.Connection, settings: Settings) -> dict:
    st = stages(conn, settings.symbols)
    issues = [(s.title, c) for s in st for c in s.checks if c.status in (WARN, FAIL)]
    worst = max((RANK[c.status] for _, c in issues), default=0)
    dec = conn.execute("SELECT ts, action, reason, bar_time FROM paper_decisions ORDER BY id DESC LIMIT 1").fetchone()
    return {
        "checked": datetime.now(timezone.utc).isoformat(),
        "banner": {"status": FAIL if worst >= RANK[FAIL] else WARN if issues else "ok",
                   "issues": [{"stage": t, "check": c.name, "value": c.value} for t, c in issues]},
        "stages": [{"key": s.key, "title": s.title, "what": s.what, "status": s.status,
                    "checks": [{"name": c.name, "status": c.status, "value": c.value, "detail": c.detail} for c in s.checks]}
                   for s in st],
        "engine": _state(conn, "engine"),
        "last_decision": {"ts": dec[0].isoformat(), "action": dec[1], "reason": dec[2], "bar_time": dec[3].isoformat()} if dec else None,
        "activity": records(activity(conn, 60)),
    }


# ---------------------------------------------------------------------------------------------------------------
# ledger and P&L

def pnl_by_coin(fills: list[dict], funding: dict[str, float], positions: dict[str, dict], start_equity: float) -> list[dict]:
    """Per coin: realized (fills replayed through the paper Account), unrealized (latest marks), fees, funding, net.
    fills: dicts with symbol, side, qty, price, liquidity in ledger order. funding: symbol -> amount paid (>0)."""
    acct = Account(start_equity)
    realized: dict[str, float] = {}
    fees: dict[str, float] = {}
    for f in fills:
        before = acct.realized
        fees[f["symbol"]] = fees.get(f["symbol"], 0.0) + acct.apply_fill(f["symbol"], f["side"], f["qty"], f["price"], f["liquidity"])
        realized[f["symbol"]] = realized.get(f["symbol"], 0.0) + acct.realized - before
    out = []
    for s in sorted(set(realized) | set(funding) | set(positions)):
        p = positions.get(s) or {}
        unreal = (p.get("qty") or 0.0) * ((p.get("mark") or 0.0) - (p.get("entry") or 0.0))
        r, fe, fu = realized.get(s, 0.0), fees.get(s, 0.0), funding.get(s, 0.0)
        out.append({"symbol": s, "realized": r, "unrealized": unreal, "fees": fe, "funding": fu, "net": r + unreal - fe - fu})
    return sorted(out, key=lambda x: x["net"])


def waterfall(start_equity: float, equity: float, fees: float, funding: float) -> dict:
    """equity = start + trading P&L - fees - funding paid, so trading P&L = equity - start + fees + funding."""
    return {"start": start_equity, "trading": equity - start_equity + fees + funding, "fees": fees,
            "funding": funding, "equity": equity, "total": equity - start_equity}


def ledger(conn: psycopg.Connection) -> dict:
    acct = _state(conn, "account")
    last = conn.execute("""SELECT ts, equity, cash, unrealized, gross_lev, net_lev, margin_ratio, fees, funding, positions
                           FROM paper_equity ORDER BY ts DESC LIMIT 1""").fetchone()
    if not acct or last is None:
        return {"empty": True, "account": acct}
    start = float(acct.get("start_equity", 0.0))
    ts, equity, cash, unreal, gross, net_lev, mratio, fees, funding, positions = last
    positions = positions or {}
    span = conn.execute("SELECT extract(epoch FROM max(ts) - min(ts)) FROM paper_equity").fetchone()[0] or 60
    step = max(60, int(float(span) / MAX_POINTS))
    points = _rows(conn, """SELECT DISTINCT ON (b) b AS t, equity
                            FROM (SELECT date_bin(make_interval(secs => %s), ts, timestamptz '2000-01-01') AS b, ts, equity
                                  FROM paper_equity) x
                            ORDER BY b, ts DESC""", (step,))
    peak = max((p["equity"] for p in points), default=equity)
    fills = _rows(conn, "SELECT symbol, side, qty, price, liquidity FROM paper_fills ORDER BY id")
    fund_by = dict(conn.execute("SELECT symbol, sum(amount) FROM paper_funding GROUP BY symbol").fetchall())
    pos_rows = [{"symbol": s, "qty": p.get("qty", 0.0), "entry": p.get("entry", 0.0), "mark": p.get("mark"),
                 "notional": (p.get("qty") or 0.0) * (p.get("mark") or 0.0),
                 "unrealized": (p.get("qty") or 0.0) * ((p.get("mark") or 0.0) - (p.get("entry") or 0.0)),
                 "weight": (p.get("qty") or 0.0) * (p.get("mark") or 0.0) / equity if equity else 0.0}
                for s, p in positions.items() if p.get("qty")]
    fsum = conn.execute("""SELECT count(*), coalesce(sum(qty * price), 0),
                                  coalesce(sum(qty * price) FILTER (WHERE liquidity = 'maker'), 0),
                                  avg(slippage_bps), coalesce(sum(fee), 0) FROM paper_fills""").fetchone()
    fund = conn.execute("""SELECT coalesce(sum(amount) FILTER (WHERE amount > 0), 0),
                                  coalesce(-sum(amount) FILTER (WHERE amount < 0), 0), count(*) FROM paper_funding""").fetchone()
    return {
        "empty": False, "as_of": ts.isoformat(), "account": acct, "risk": _state(conn, "risk"),
        "kpi": {"equity": equity, "start": start, "cash": cash, "unrealized": unreal, "gross_lev": gross, "net_lev": net_lev,
                "margin_ratio": mratio, "drawdown": 1 - equity / peak if peak else 0.0, "peak": peak},
        "waterfall": waterfall(start, equity, fees, funding),
        "curve": [{"t": p["t"], "equity": p["equity"]} for p in points],
        "positions": sorted(pos_rows, key=lambda r: -abs(r["notional"])),
        "by_coin": pnl_by_coin(fills, {k: float(v) for k, v in fund_by.items()}, positions, start),
        "fills_summary": {"count": fsum[0], "traded": float(fsum[1]), "maker_share": float(fsum[2]) / float(fsum[1]) if fsum[1] else None,
                          "avg_slip_bps": float(fsum[3]) if fsum[3] is not None else None, "fees": float(fsum[4])},
        "funding_summary": {"paid": float(fund[0]), "received": float(fund[1]), "count": fund[2]},
        "decisions": _rows(conn, """SELECT ts, bar_time, action, reason, equity FROM paper_decisions
                                    WHERE action <> 'HOLD' OR id > (SELECT max(id) - 24 FROM paper_decisions)
                                    ORDER BY id DESC LIMIT 60"""),
        "orders": _rows(conn, """SELECT ts, symbol, kind, side, price, qty, filled, status, closed_at FROM paper_orders
                                 ORDER BY id DESC LIMIT 100"""),
        "fills": _rows(conn, """SELECT ts, symbol, side, qty, price, liquidity, fee, slippage_bps, latency_ms, beyond_book
                                FROM paper_fills ORDER BY id DESC LIMIT 200"""),
        "funding": _rows(conn, "SELECT ts, symbol, qty, mark, rate, amount FROM paper_funding ORDER BY id DESC LIMIT 200"),
    }


# ---------------------------------------------------------------------------------------------------------------
# market data

def market_light(conn: psycopg.Connection, settings: Settings) -> dict:
    symbols = list(dict.fromkeys(f.db_symbol for f in settings.candle_feeds()))
    health = audit.candle_health(conn, settings.all_intervals, symbols)
    stale = int(health["stale"].sum()) if len(health) else 0
    return {"streams_live": len(health) - stale, "streams_expected": len(settings.candle_feeds()),
            "total_candles": int(health["bars"].sum()) if len(health) else 0, "health": records(health),
            "fetches": records(audit.last_fetches(conn, 15)), "ws_events": records(audit.ws_events(conn)),
            "symbols": symbols, "intervals": settings.all_intervals}


def market_heavy(conn: psycopg.Connection, settings: Settings) -> dict:
    """The expensive audit queries (full scans): cached for minutes by the server, never per request."""
    symbols = list(dict.fromkeys(f.db_symbol for f in settings.candle_feeds()))
    gaps = audit.candle_gaps(conn, settings.all_intervals, symbols)
    return {"missing_7d": int(gaps["missing"].sum()) if len(gaps) else 0, "gaps": records(gaps),
            "ohlc_violations": records(audit.ohlc_violations(conn)), "resample_mismatch": records(audit.resample_mismatch(conn)),
            "feeds": records(audit.feed_health(conn)), "computed": datetime.now(timezone.utc).isoformat()}


def candles(conn: psycopg.Connection, symbol: str, interval: str, bars: int) -> dict:
    bars_df = audit._df(conn, """SELECT open_time, open::float8 AS open, high::float8 AS high, low::float8 AS low,
                                        close::float8 AS close, volume::float8 AS volume
                                 FROM candles WHERE symbol = %s AND interval = %s ORDER BY open_time DESC LIMIT %s""",
                        (symbol, interval, bars)).sort_values("open_time") if bars else pd.DataFrame()
    out = {"symbol": symbol, "interval": interval, "bars": records(bars_df), "oi": [], "funding": []}
    if len(bars_df):
        base, first = symbol.removesuffix(PERP_SUFFIX), bars_df["open_time"].min()
        out["oi"] = records(audit._df(conn, """SELECT ts, sum_open_interest_value::float8 AS v FROM open_interest
                                               WHERE symbol = %s AND ts >= %s ORDER BY ts""", (base, first)))
        out["funding"] = records(audit._df(conn, """SELECT ts, last_funding_rate::float8 AS v FROM premium_snap
                                                    WHERE symbol = %s AND ts >= %s ORDER BY ts""", (base, first)))
    return out
