"""Coin candidates: check -> shadow -> live. Requested from the dashboard's Coins tab; this worker does the work.

    uv run python -m alpha.coins          # worker (systemd alpha-coins): runs queued checks, updates shadows daily

Stages (table coin_candidates, every action also in coin_events):
    queued -> checking -> checked       basic rules (USD-M USDT perpetual, crypto, trading, >= MIN_AGE_DAYS listed,
                                        30-day median volume >= $20M), history download, backtest of today's list vs
                                        today's list + the coin from CHECK_START (V4-C as deployed: same models,
                                        band, costs, funding). Verdict: good / weak / reject (see verdict()).
    checked -> shadow                   user confirms (good or weak only). Re-replayed every day on new data only,
                                        from shadow_start: no hindsight.
    shadow -> live                      user confirms again, allowed after SHADOW_MIN_DAYS. The engine and collector
                                        notice the new list within the hour and restart themselves (systemd) with it.
    any -> rejected                     user, or a failed check. The report stays; the coin's downloaded history is
                                        deleted PURGE_AFTER_DAYS later (never for a coin on the trading list).
A coin is never added on a backtest alone. Live coins cannot be removed here yet: closing their positions first is
not built (do it by hand: stop alpha-paper, set the row to 'rejected', start it again once the coin is flat).
"""

import asyncio
import re
import statistics
import sys
import time
from datetime import date, datetime, timezone

import pandas as pd
import psycopg
from loguru import logger
from psycopg.types.json import Jsonb

from alpha.config import get_settings
from alpha.universe import MIN_AGE_DAYS, MIN_MEDIAN_QUOTE_VOL

SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}USDT$")
CHECK_START = pd.Timestamp("2022-01-01", tz="UTC")
HIST_MODELS = "research_models/hist"   # quarterly ridge bundles 2021-26 as the engine had them (tracked, 200 KB)
MIN_BACKTEST_DAYS = 180     # fewer eligible days than this -> no backtest verdict, shadow only
SHADOW_MIN_DAYS = 90
REJECT_T = -2.0             # reliably worse
HISTORY_SINCE = date(2021, 6, 1)
VISION_SETS = "klines,premium,metrics,bookDepth"
PURGE_AFTER_DAYS = 7        # rejected coins: keep the history this long (a re-check needs no new download)
# (table, symbol column value) for every row a check downloads; perp-named tables use SYMBOL.P
PURGE_TABLES = [("candles", "perp"), ("book_depth_5m", "perp"), ("premium_kline", "plain"),
                ("futures_metrics", "plain"), ("funding_rate", "plain"), ("vision_files", "plain")]

LIVE_SQL = "SELECT symbol FROM coin_candidates WHERE status = 'live' ORDER BY promoted_at"


def merge_symbols(base: list[str], live: list[str]) -> list[str]:
    return base + [s for s in live if s not in base]


async def trading_symbols_async(db, base: list[str]) -> list[str]:
    """Engine / collector (asyncpg): today's list + coins promoted to live from the dashboard."""
    return merge_symbols(base, [r["symbol"] for r in await db.pool.fetch(LIVE_SQL)])


async def watch_symbols(db, base: list[str], current: list[str], every_s: float = 600) -> None:
    """Returns when the trading list changes, so the service exits and systemd restarts it with the new list."""
    while True:
        await asyncio.sleep(every_s)
        try:
            new = await trading_symbols_async(db, base)
        except Exception:
            logger.exception("coin list check failed")
            continue
        if new != current:
            logger.warning("trading list changed: {} -> {}; restarting", current, new)
            return


def trading_symbols(conn: psycopg.Connection, base: list[str]) -> list[str]:
    try:
        return merge_symbols(base, [r[0] for r in conn.execute(LIVE_SQL).fetchall()])
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return base


def event(conn: psycopg.Connection, symbol: str, action: str, detail: dict | None = None) -> None:
    conn.execute("INSERT INTO coin_events (symbol, action, detail) VALUES (%s, %s, %s)", (symbol, action, Jsonb(detail or {})))


# ---------------- dashboard actions (sync, one connection) ----------------

def request_check(conn: psycopg.Connection, symbol: str, base: list[str]) -> str:
    symbol = symbol.strip().upper()
    if not SYMBOL_RE.match(symbol):
        raise ValueError("use a Binance USD-M symbol like INJUSDT")
    if symbol in base:
        raise ValueError(f"{symbol} is already in the trading list")
    row = conn.execute("SELECT status FROM coin_candidates WHERE symbol = %s", (symbol,)).fetchone()
    if row and row[0] in ("queued", "checking", "shadow", "live"):
        raise ValueError(f"{symbol} is already {row[0]}")
    conn.execute("""INSERT INTO coin_candidates (symbol, status, requested_at) VALUES (%s, 'queued', now())
                    ON CONFLICT (symbol) DO UPDATE SET status = 'queued', requested_at = now(), verdict = NULL,
                    report = NULL, error = NULL, checked_at = NULL, shadow_start = NULL, shadow = NULL,
                    purged_at = NULL""", (symbol,))
    event(conn, symbol, "check_requested")
    return symbol


def move(conn: psycopg.Connection, symbol: str, to: str) -> None:
    """User decisions: shadow (after a good/weak check), live (after SHADOW_MIN_DAYS of shadow), rejected."""
    row = conn.execute("SELECT status, verdict, shadow_start FROM coin_candidates WHERE symbol = %s", (symbol,)).fetchone()
    if not row:
        raise ValueError(f"{symbol} has not been checked")
    status, verdict, start = row
    if to == "shadow":
        if status != "checked" or verdict not in ("good", "weak"):
            raise ValueError("only a checked coin with a good or weak verdict can start shadow testing")
        conn.execute("UPDATE coin_candidates SET status = 'shadow', shadow_start = date_trunc('day', now()) "
                     "+ interval '1 day' WHERE symbol = %s", (symbol,))
    elif to == "live":
        days = (datetime.now(timezone.utc) - start).days if start else -1
        if status != "shadow" or days < SHADOW_MIN_DAYS:
            raise ValueError(f"needs {SHADOW_MIN_DAYS} days of shadow testing first ({max(days, 0)} so far)")
        conn.execute("UPDATE coin_candidates SET status = 'live', promoted_at = now() WHERE symbol = %s", (symbol,))
    elif to == "rejected":
        if status == "live":
            raise ValueError("live coins can't be removed from the page yet (see alpha.coins)")
        conn.execute("UPDATE coin_candidates SET status = 'rejected' WHERE symbol = %s", (symbol,))
    else:
        raise ValueError("unknown action")
    event(conn, symbol, to, {"from": status})


def purge(conn: psycopg.Connection, symbol: str, keep: set[str]) -> dict[str, int]:
    """Delete a rejected coin's downloaded history (report kept). Refuses anything still in use."""
    if symbol in keep:
        raise ValueError(f"{symbol} is in use; not deleting its data")
    out = {}
    for table, kind in PURGE_TABLES:
        out[table] = conn.execute(f"DELETE FROM {table} WHERE symbol = %s",
                                  (symbol + ".P" if kind == "perp" else symbol,)).rowcount
    conn.execute("UPDATE coin_candidates SET purged_at = now() WHERE symbol = %s", (symbol,))
    event(conn, symbol, "purged", out)
    return out


PURGE_DUE_SQL = f"""SELECT c.symbol FROM coin_candidates c WHERE c.status = 'rejected' AND c.purged_at IS NULL
    AND (SELECT max(ts) FROM coin_events e WHERE e.symbol = c.symbol) < now() - interval '{PURGE_AFTER_DAYS} days'"""
IN_USE_SQL = "SELECT symbol FROM coin_candidates WHERE status IN ('queued', 'checking', 'checked', 'shadow', 'live')"


def listing(conn: psycopg.Connection) -> list[dict]:
    cols = ["symbol", "status", "verdict", "requested_at", "checked_at", "shadow_start", "promoted_at", "report",
            "shadow", "error", "purged_at"]
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM coin_candidates ORDER BY requested_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("requested_at", "checked_at", "shadow_start", "promoted_at", "purged_at"):
            d[k] = d[k].isoformat() if d[k] else None
        out.append(d)
    return out


# ---------------- the check ----------------

def basic_rules(info: dict | None, daily_quote_vol: list[float], today: date) -> dict:
    """info: the coin's exchangeInfo entry (None = not listed). daily_quote_vol: last 30 completed days."""
    if info is None:
        return {"pass": False, "fails": ["not a Binance USD-M contract"]}
    listed = datetime.fromtimestamp(info["onboardDate"] / 1000, timezone.utc).date()
    med = statistics.median(daily_quote_vol) if daily_quote_vol else 0.0
    checks = {
        "USDT perpetual": info.get("contractType") == "PERPETUAL" and info.get("quoteAsset") == "USDT",
        "crypto (not a stock/gold contract)": info.get("underlyingType", "COIN") == "COIN",
        "trading now": info.get("status") == "TRADING",
        f"listed {MIN_AGE_DAYS}+ days": (today - listed).days >= MIN_AGE_DAYS,
        f"30-day median volume >= ${MIN_MEDIAN_QUOTE_VOL / 1e6:.0f}M": med >= MIN_MEDIAN_QUOTE_VOL,
    }
    return {"pass": all(checks.values()), "fails": [k for k, ok in checks.items() if not ok], "listed": str(listed),
            "age_days": (today - listed).days, "median_volume": med, "checks": checks}


def compare(base: pd.Series, test: pd.Series) -> dict:
    """Daily net returns (fractions) of today's list and list + coin -> annual difference, t-stat, halves."""
    d = (test - base).dropna()
    if len(d) < 2 or d.std() == 0:
        return {"days": len(d), "ann_diff": 0.0, "t": 0.0}
    half = len(d) // 2
    return {"days": len(d), "ann_diff": float(d.mean() * 365), "t": float(d.mean() / d.std() * len(d) ** .5),
            "first_half": float(d.iloc[:half].mean() * 365), "second_half": float(d.iloc[half:].mean() * 365),
            "base_ann": float(base.mean() * 365), "test_ann": float(test.mean() * 365),
            "base_total": float((1 + base).prod() - 1), "test_total": float((1 + test).prod() - 1)}


def verdict(rules: dict, bt: dict | None) -> tuple[str, str]:
    if not rules["pass"]:
        return "reject", "fails the basic rules: " + ", ".join(rules["fails"])
    if bt is None or bt["days"] < MIN_BACKTEST_DAYS:
        return "weak", f"too new for a backtest (< {MIN_BACKTEST_DAYS} eligible days): shadow test only"
    if bt["ann_diff"] < 0 and bt["t"] <= REJECT_T:
        return "reject", f"makes the bot reliably worse ({bt['ann_diff']:+.1%} a year, t = {bt['t']:.1f})"
    if bt["ann_diff"] >= 0:
        return "good", f"did not hurt ({bt['ann_diff']:+.1%} a year, t = {bt['t']:.1f}): good to shadow test"
    return "weak", f"slightly worse, within noise ({bt['ann_diff']:+.1%} a year, t = {bt['t']:.1f})"


def replay(conn: psycopg.Connection, base: list[str], coin: str, start: pd.Timestamp, end: pd.Timestamp,
           models_dir: str, costs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """V4-C daily net on [start, end) for today's list and for list + coin, one calendar year of panel at a time
    (memory); both books from the same panel, so the database is read once."""
    from alpha.live.report import p6_scores, shadow_from_weights
    from alpha.research import signals as sig
    from alpha.research.models import feature_frame
    from alpha.research.panel import build_panel
    from alpha.strategy import p6

    out: dict[str, list[pd.Series]] = {"base": [], "test": []}
    costs = costs.reindex(base + [coin]).fillna(costs.quantile(0.75))
    t = start
    while t < end:
        t2 = min(pd.Timestamp(f"{t.year + 1}-01-01", tz="UTC"), end)
        full = build_panel(conn, base + [coin], start=t - pd.Timedelta(days=p6.HISTORY_DAYS), end=t2)
        for k, panel in (("test", full), ("base", full.drop(index=coin, level="symbol", errors="ignore"))):
            X, _ = feature_frame(panel, sig.compute(panel))
            sc = p6_scores(X, t2, models_dir)
            if sc is not None:
                out[k].append(shadow_from_weights(panel, p6.book_weights(panel, sc), costs, t, t2)["net"])
            del X
        del full
        t = t2
    b, x = (pd.concat(out[k]) if out[k] else pd.Series(dtype=float) for k in ("base", "test"))
    return b, x.reindex(b.index).fillna(0.0)


def first_eligible(conn: psycopg.Connection, symbol: str) -> pd.Timestamp | None:
    from alpha.universe import universe_mask
    m = universe_mask(conn, [symbol]).iloc[:, 0]
    return m.idxmax() if m.any() else None


def costs_for(conn: psycopg.Connection, base: list[str], symbol: str) -> pd.DataFrame:
    """The research cost model for today's list, plus the coin's own when it traded in DEV (else replay() charges
    it the 75th percentile of the others)."""
    from alpha.research.panel import build_panel, cached_panel
    from alpha.research.screen import symbol_costs
    from alpha.research.splits import DEV

    c = symbol_costs(conn, cached_panel(conn, base))
    try:
        one = symbol_costs(conn, build_panel(conn, [symbol], start=DEV.start, end=DEV.end)).dropna()
    except ValueError:  # no rows in DEV
        one = c.iloc[:0]
    return pd.concat([c.drop(index=[symbol], errors="ignore"), one])


async def download(symbol: str, since: date) -> None:
    """History for one coin: data.binance.vision dumps + funding (REST). Resumable."""
    from alpha.binance.parse import ms_to_dt
    from alpha.binance.rest import BinanceREST
    from alpha.binance.vision import backfill
    from alpha.db import DB

    await backfill([symbol], VISION_SETS.split(","), ["1h", "1d"], since, workers=6)
    s = get_settings()
    db, rest = await DB.connect(s.database_url), BinanceREST(s.spot_rest, s.futures_rest)
    try:
        last = await db.last_time("funding_rate", "funding_time", symbol)
        start = int(last.timestamp() * 1000) + 1 if last else int(datetime(since.year, since.month, since.day,
                                                                             tzinfo=timezone.utc).timestamp() * 1000)
        while True:
            b = await rest.funding_rate(symbol, start)
            if not b:
                break
            await db.upsert("funding_rate", [(symbol, ms_to_dt(d["fundingTime"]), float(d["fundingRate"]),
                                              float(d["markPrice"]) if d.get("markPrice") else None) for d in b])
            if len(b) < 1000:
                break
            start = b[-1]["fundingTime"] + 1
    finally:
        await rest.close()
        await db.close()


def market_info(symbol: str, base_url: str) -> tuple[dict | None, list[float]]:
    import httpx
    with httpx.Client(base_url=base_url, timeout=30) as c:
        info = next((x for x in c.get("/fapi/v1/exchangeInfo").json()["symbols"] if x["symbol"] == symbol), None)
        if info is None:
            return None, []
        k = c.get("/fapi/v1/klines", params={"symbol": symbol, "interval": "1d", "limit": 31}).json()[:-1]
        return info, [float(x[7]) for x in k]


def run_check(conn: psycopg.Connection, symbol: str, base: list[str]) -> None:
    s = get_settings()
    conn.execute("UPDATE coin_candidates SET status = 'checking', error = NULL WHERE symbol = %s", (symbol,))
    info, vols = market_info(symbol, s.futures_rest)
    rules = basic_rules(info, vols, datetime.now(timezone.utc).date())
    report = {"rules": {k: v for k, v in rules.items() if k != "checks"} | {"checks": rules.get("checks", {})}}
    bt = None
    if rules["pass"]:
        asyncio.run(download(symbol, HISTORY_SINCE))
        fe = first_eligible(conn, symbol)
        report["first_eligible"] = str(fe.date()) if fe is not None else None
        if fe is not None:
            start, end = max(CHECK_START, fe), pd.Timestamp.now(tz="UTC").floor("D")
            costs = costs_for(conn, base, symbol)
            own = costs.loc[symbol] if symbol in costs.index else costs.quantile(0.75)
            report["cost_bps"] = {"taker": float(own["taker_bps"]), "maker": float(own["maker_bps"]),
                                  "estimated": symbol not in costs.index}
            if (end - start).days >= MIN_BACKTEST_DAYS:
                b, t = replay(conn, base, symbol, start, end, HIST_MODELS, costs)
                bt = compare(b, t) | {"start": str(start.date()), "end": str(end.date())}
                report["curve"] = [[str(i.date()), round(float(x), 4), round(float(y), 4)] for i, x, y in
                                   zip(b.index, (1 + b).cumprod(), (1 + t).cumprod())][::7]
            else:
                bt = {"days": (end - start).days}
    report["backtest"] = bt
    v, why = verdict(rules, bt)
    report["why"] = why
    conn.execute("UPDATE coin_candidates SET status = %s, verdict = %s, report = %s, checked_at = now() WHERE symbol = %s",
                 ("rejected" if v == "reject" else "checked", v, Jsonb(report), symbol))
    event(conn, symbol, "checked", {"verdict": v, "why": why})


def update_shadow(conn: psycopg.Connection, symbol: str, start: datetime, base: list[str]) -> None:
    """Forward-only: today's list vs list + coin on the days since shadow_start, with the live models."""
    s = get_settings()
    asyncio.run(download(symbol, HISTORY_SINCE))
    start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp.now(tz="UTC").floor("D")
    if end <= start:
        return
    costs = costs_for(conn, base, symbol)
    b, t = replay(conn, base, symbol, start, end, s.models_dir, costs)
    res = compare(b, t) | {"updated": datetime.now(timezone.utc).isoformat(), "ready": (end - start).days >= SHADOW_MIN_DAYS}
    conn.execute("UPDATE coin_candidates SET shadow = %s WHERE symbol = %s", (Jsonb(res), symbol))
    event(conn, symbol, "shadow_update", {k: res[k] for k in ("days", "ann_diff", "t") if k in res})


def worker() -> None:
    s = get_settings()
    last_shadow: date | None = None
    while True:
        with psycopg.connect(s.database_url, autocommit=True) as conn:
            base = s.symbols
            conn.execute("UPDATE coin_candidates SET status = 'queued' WHERE status = 'checking'")  # crashed mid-check
            for (sym,) in conn.execute("SELECT symbol FROM coin_candidates WHERE status = 'queued' ORDER BY requested_at").fetchall():
                logger.info("checking {}", sym)
                try:
                    run_check(conn, sym, trading_symbols(conn, base))
                except Exception as e:
                    logger.exception("check {} failed", sym)
                    conn.execute("UPDATE coin_candidates SET status = 'checked', verdict = NULL, error = %s WHERE symbol = %s",
                                 (str(e)[:500], sym))
                    event(conn, sym, "check_failed", {"error": str(e)[:500]})
            today = datetime.now(timezone.utc).date()
            if last_shadow != today and datetime.now(timezone.utc).hour >= 2:   # after the daily dumps are out
                for sym, start in conn.execute("SELECT symbol, shadow_start FROM coin_candidates WHERE status = 'shadow'").fetchall():
                    try:
                        update_shadow(conn, sym, start, trading_symbols(conn, base))
                    except Exception as e:
                        logger.exception("shadow {} failed", sym)
                        event(conn, sym, "shadow_failed", {"error": str(e)[:500]})
                last_shadow = today
                keep = set(trading_symbols(conn, base)) | {r[0] for r in conn.execute(IN_USE_SQL).fetchall()}
                for (sym,) in conn.execute(PURGE_DUE_SQL).fetchall():
                    try:
                        logger.info("purged {}: {}", sym, purge(conn, sym, keep))
                    except Exception:
                        logger.exception("purge {} failed", sym)
        time.sleep(30)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    worker()
