"""History backfill + gap repair for candles.

Run once by hand:   uv run python -m alpha.backfill
The collector also calls these on startup, after every websocket reconnect, and periodically.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger

from alpha.binance.parse import dt_to_ms, interval_td, rest_kline_to_row
from alpha.binance.rest import BinanceREST
from alpha.config import Feed, Settings, get_settings
from alpha.db import DB

_locks: dict[Feed, asyncio.Lock] = {}


def default_start(settings: Settings, interval: str) -> datetime:
    if interval == "1m":
        return datetime.now(timezone.utc) - timedelta(days=settings.backfill_1m_days)
    return settings.backfill_start_dt


async def fetch_and_store(db: DB, rest: BinanceREST, feed: Feed, start_ms: int,
                          end_ms: int | None = None, source: str = "rest") -> int:
    total = 0
    async for batch in rest.klines_range(feed.api_symbol, feed.interval, start_ms, end_ms, market=feed.market):
        rows = [rest_kline_to_row(feed.db_symbol, feed.interval, k) for k in batch]
        if source != "rest":
            rows = [r[:-1] + (source,) for r in rows]
        await db.upsert("candles", rows)
        total += len(rows)
        await db.log_fetch("candle", source, symbol=feed.db_symbol, interval=feed.interval,
                           ref_time=rows[-1][3], rows=len(rows))
    return total


async def backfill_candles(db: DB, rest: BinanceREST, settings: Settings, feed: Feed,
                           connected_at: datetime | None = None) -> int:
    """Fill from the last stored candle (or the configured start) up to now. Skips if already running.

    connected_at: when the live websocket (re)connected. Candles it writes after that point must not
    count as "already stored", otherwise the hole before them would never be backfilled."""
    lock = _locks.setdefault(feed, asyncio.Lock())
    if lock.locked():
        return 0
    symbol, interval = feed.db_symbol, feed.interval
    async with lock:
        step = interval_td(interval)
        before = connected_at - step if connected_at else None
        last = await db.last_time("candles", "open_time", symbol, interval, before=before)
        start = last + step if last else default_start(settings, interval)
        try:
            n = 0
            # Head: stored history starts later than configured (e.g. live rows landed first, or the
            # start date was moved back). For coins listed later Binance returns nothing here: cheap no-op.
            first = await db.pool.fetchval(
                "SELECT min(open_time) FROM candles WHERE symbol = $1 AND interval = $2", symbol, interval)
            head_from = default_start(settings, interval)
            if first and first - step > head_from:
                n += await fetch_and_store(db, rest, feed, dt_to_ms(head_from), dt_to_ms(first) - 1)
            # Tail: from the last candle stored before this connection up to now.
            n += await fetch_and_store(db, rest, feed, dt_to_ms(start))
        except Exception as e:
            logger.exception("backfill {} {} failed", symbol, interval)
            await db.log_fetch("candle", "rest", symbol=symbol, interval=interval, status="error", error=str(e))
            return 0
        if n:
            logger.info("backfill {} {}: {} candles", symbol, interval, n)
        return n


async def backfill_all_candles(db: DB, rest: BinanceREST, settings: Settings, market: str | None = None,
                               connected_at: datetime | None = None) -> None:
    # Coarse intervals first so higher timeframes are usable quickly; 1m (largest) last.
    feeds = sorted(settings.candle_feeds(market), key=lambda f: -interval_td(f.interval).total_seconds())
    for feed in feeds:
        await backfill_candles(db, rest, settings, feed, connected_at)


GAPS_SQL = """
SELECT symbol, interval, open_time AS gap_after, next_open AS gap_before
FROM (
    SELECT symbol, interval, open_time,
           lead(open_time) OVER (PARTITION BY symbol, interval ORDER BY open_time) AS next_open
    FROM candles
    WHERE symbol = $1 AND interval = $2 AND open_time > now() - $3::interval
) t
WHERE next_open - open_time > $4::interval
ORDER BY open_time
"""


async def repair_gaps(db: DB, rest: BinanceREST, settings: Settings, lookback: timedelta | None = timedelta(days=7)) -> int:
    """Re-fetch holes inside stored history (lookback=None: all of it).
    Real exchange outages stay as gaps; re-checking them costs one empty request each."""
    repaired = 0
    for feed in settings.candle_feeds():
        step = interval_td(feed.interval)
        window = max(lookback, step * 20) if lookback else timedelta(days=36500)
        gaps = await db.pool.fetch(GAPS_SQL, feed.db_symbol, feed.interval, window, step)
        for g in gaps:
            start = dt_to_ms(g["gap_after"] + step)
            end = dt_to_ms(g["gap_before"]) - 1
            n = await fetch_and_store(db, rest, feed, start, end, source="gap_repair")
            if n:
                logger.info("gap {} {} {} -> {}: repaired {}", feed.db_symbol, feed.interval,
                            g["gap_after"], g["gap_before"], n)
            repaired += n
    return repaired


async def main() -> None:
    settings = get_settings()
    db = await DB.connect(settings.database_url)
    rest = BinanceREST(settings.spot_rest, settings.futures_rest)
    try:
        await db.init_schema()
        await backfill_all_candles(db, rest, settings)
        await repair_gaps(db, rest, settings, lookback=None)
        if settings.futures_enabled:
            from alpha.collectors.futures import poll_futures_once

            await poll_futures_once(db, rest, settings)
    finally:
        await rest.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
