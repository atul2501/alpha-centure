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
