"""USD-M futures context: funding, premium/mark, open interest, long/short + taker ratios (polled),
and liquidations (websocket)."""

import asyncio
from datetime import datetime, timezone

from loguru import logger

from alpha.binance.parse import dt_to_ms, liquidation_row, ms_to_dt, now_ms
from alpha.binance.rest import BinanceREST
from alpha.binance.ws import combined_url, run_stream
from alpha.config import Settings
from alpha.db import DB

POLL_EVERY_S = 300
POLL_OFFSET_S = 20  # Binance publishes the 5m stats a few seconds after the bucket closes

# table -> (endpoint path, payload -> row)
STATS = {
    "open_interest": ("openInterestHist",
                      lambda s, d: (s, ms_to_dt(d["timestamp"]), float(d["sumOpenInterest"]), float(d["sumOpenInterestValue"]))),
    "long_short_ratio": ("globalLongShortAccountRatio",
                         lambda s, d: (s, ms_to_dt(d["timestamp"]), float(d["longShortRatio"]), float(d["longAccount"]), float(d["shortAccount"]))),
    "taker_ratio": ("takerlongshortRatio",
                    lambda s, d: (s, ms_to_dt(d["timestamp"]), float(d["buySellRatio"]), float(d["buyVol"]), float(d["sellVol"]))),
}
LOG_KIND = {"open_interest": "oi", "long_short_ratio": "ls_ratio", "taker_ratio": "taker_ratio"}


async def _poll_symbol(db: DB, rest: BinanceREST, settings: Settings, symbol: str) -> None:
    for table, (path, to_row) in STATS.items():
        last = await db.last_time(table, "ts", symbol)
        start = dt_to_ms(last) + 1 if last else 0
        async for batch in rest.futures_stats_range(path, symbol, start):
            rows = [to_row(symbol, d) for d in batch]
            await db.upsert(table, rows)
            await db.log_fetch(LOG_KIND[table], "rest", symbol=symbol, interval="5m",
                               ref_time=max(r[1] for r in rows), rows=len(rows))

    # Live rows for the research tables (history comes from data.binance.vision; same definitions, checked to
    # match within 0.3%). Dump values are kept; REST only fills rows / columns that are missing.
    await db.pool.execute(METRICS_FROM_LIVE_SQL, symbol)
    await _poll_premium_klines(db, rest, symbol)

    last = await db.last_time("funding_rate", "funding_time", symbol)
    start = dt_to_ms(last) + 1 if last else dt_to_ms(settings.backfill_start_dt)
    while True:
        batch = await rest.funding_rate(symbol, start)
        if not batch:
            break
        rows = [(symbol, ms_to_dt(d["fundingTime"]), float(d["fundingRate"]),
                 float(d["markPrice"]) if d.get("markPrice") else None) for d in batch]
        await db.upsert("funding_rate", rows)
        await db.log_fetch("funding", "rest", symbol=symbol, ref_time=rows[-1][1], rows=len(rows))
        if len(batch) < 1000:
            break
        start = batch[-1]["fundingTime"] + 1

    p = await rest.premium_index(symbol)
    row = (symbol, ms_to_dt(p["time"]), float(p["markPrice"]), float(p["indexPrice"]),
           float(p["lastFundingRate"]), ms_to_dt(p["nextFundingTime"]) if p.get("nextFundingTime") else None)
    await db.upsert("premium_snap", [row])
    await db.log_fetch("premium", "rest", symbol=symbol, ref_time=row[1], rows=1)


METRICS_FROM_LIVE_SQL = """
INSERT INTO futures_metrics (symbol, ts, sum_open_interest, sum_open_interest_value, ls_ratio, taker_ls_vol_ratio)
SELECT o.symbol, o.ts, o.sum_open_interest, o.sum_open_interest_value, l.long_short_ratio, t.buy_sell_ratio
FROM open_interest o
LEFT JOIN long_short_ratio l USING (symbol, ts)
LEFT JOIN taker_ratio t USING (symbol, ts)
WHERE o.symbol = $1 AND o.ts > now() - interval '2 days'
ON CONFLICT (symbol, ts) DO UPDATE SET
    sum_open_interest = COALESCE(futures_metrics.sum_open_interest, EXCLUDED.sum_open_interest),
    sum_open_interest_value = COALESCE(futures_metrics.sum_open_interest_value, EXCLUDED.sum_open_interest_value),
    ls_ratio = COALESCE(futures_metrics.ls_ratio, EXCLUDED.ls_ratio),
    taker_ls_vol_ratio = COALESCE(futures_metrics.taker_ls_vol_ratio, EXCLUDED.taker_ls_vol_ratio)
"""


async def _poll_premium_klines(db: DB, rest: BinanceREST, symbol: str, interval: str = "1h") -> None:
    last = await db.pool.fetchval("SELECT max(open_time) FROM premium_kline WHERE symbol = $1 AND interval = $2",
                                  symbol, interval)
    start = dt_to_ms(last) + 1 if last else now_ms() - 30 * 86_400_000
    while True:
        batch = await rest.premium_klines(symbol, interval, start)
        now = now_ms()
        rows = [(symbol, interval, ms_to_dt(k[0]), ms_to_dt(k[6]), float(k[1]), float(k[2]), float(k[3]),
                 float(k[4])) for k in batch if k[6] < now]
        await db.upsert("premium_kline", rows)
        if len(batch) < 500 or not rows:
            break
        start = batch[-1][0] + 1


async def poll_futures_once(db: DB, rest: BinanceREST, settings: Settings) -> None:
    for symbol in settings.symbols:
        try:
            await _poll_symbol(db, rest, settings, symbol)
        except Exception as e:
            logger.exception("futures poll {} failed", symbol)
            await db.log_fetch("futures", "rest", symbol=symbol, status="error", error=str(e))


async def run_futures_poller(db: DB, rest: BinanceREST, settings: Settings) -> None:
    while True:
        await poll_futures_once(db, rest, settings)
        now = datetime.now(timezone.utc).timestamp()
        next_run = (now // POLL_EVERY_S + 1) * POLL_EVERY_S + POLL_OFFSET_S
        await asyncio.sleep(max(1.0, next_run - now))


async def run_oi_live_poller(db: DB, rest: BinanceREST, settings: Settings) -> None:
    """Current open interest every OI_POLL_SECONDS, aligned to the clock (finer than the 5m history)."""
    every = max(10, settings.oi_poll_seconds)
    while True:
        rows = []
        for symbol in settings.symbols:
            try:
                d = await rest.open_interest(symbol)
                rows.append((symbol, ms_to_dt(d["time"]), float(d["openInterest"])))
            except Exception as e:
                logger.warning("open interest {} failed: {}", symbol, e)
                await db.log_fetch("oi_live", "rest", symbol=symbol, status="error", error=str(e))
        await db.upsert("open_interest_live", rows)
        now = datetime.now(timezone.utc).timestamp()
        await asyncio.sleep(max(1.0, (now // every + 1) * every - now))


async def run_liquidations(db: DB, settings: Settings) -> None:
    # The all-market stream is busy every few seconds, so the idle watchdog can tell "quiet" from "dead".
    # Per-symbol streams for 4 coins can be silent for minutes.
    wanted = set(settings.symbols)

    async def on_connect() -> None:
        await db.log_fetch("ws", "ws", interval="liquidations", status="connected")

    async def on_disconnect(reason: str) -> None:
        await db.log_fetch("ws", "ws", interval="liquidations", status="disconnected", error=reason)

    async def on_message(stream: str, data: dict) -> None:
        if data["o"]["s"] not in wanted:
            return
        row = liquidation_row(data["o"])
        await db.upsert("liquidations", [row])
        await db.log_fetch("liquidation", "ws", symbol=row[0], ref_time=row[1], rows=1)
        logger.debug("liquidation {} {} {} @ {}", row[0], row[2], row[5], row[3])

    await run_stream("liquidations", combined_url(settings.futures_ws, ["!forceOrder@arr"]),
                     on_message, on_connect, on_disconnect, idle_timeout=300)
