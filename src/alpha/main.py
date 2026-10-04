"""Collector entrypoint: uv run python -m alpha.main"""

import asyncio
import signal
import sys
from datetime import timedelta

from loguru import logger

from alpha.backfill import repair_gaps
from alpha.binance.rest import BinanceREST
from alpha.collectors.futures import run_futures_poller, run_liquidations
from alpha.collectors.klines import run_klines
from alpha.collectors.orderflow import run_orderflow
from alpha.config import get_settings
from alpha.db import DB

GAP_REPAIR_EVERY_S = 1800


async def maintenance(db: DB, rest: BinanceREST, settings) -> None:
    # Full-history scan once at startup (after the initial backfill has had time to run), then recent only.
    lookback = None
    while True:
        await asyncio.sleep(GAP_REPAIR_EVERY_S)
        try:
            await repair_gaps(db, rest, settings, lookback=lookback)
            lookback = timedelta(days=2)
            await db.prune_fetch_log(days=30)
        except Exception:
            logger.exception("maintenance failed")


async def main() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")

    settings = get_settings()
    db = await DB.connect(settings.database_url)
    await db.init_schema()
    rest = BinanceREST(settings.spot_rest, settings.futures_rest)
    logger.info("collecting {} spot {} perp {}", settings.symbols, settings.intervals, settings.perp_intervals)

    tasks = [
        asyncio.create_task(run_klines(db, rest, settings, "spot"), name="klines_spot"),
        asyncio.create_task(maintenance(db, rest, settings), name="maintenance"),
    ]
    if settings.orderflow_enabled:
        tasks.append(asyncio.create_task(run_orderflow(db, settings), name="orderflow"))
    if settings.futures_enabled:
        tasks.append(asyncio.create_task(run_klines(db, rest, settings, "perp"), name="klines_perp"))
        tasks.append(asyncio.create_task(run_futures_poller(db, rest, settings), name="futures"))
        tasks.append(asyncio.create_task(run_liquidations(db, settings), name="liquidations"))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    stopper = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait([*tasks, stopper], return_when=asyncio.FIRST_COMPLETED)
    for t in done:
        if t is not stopper and t.exception():
            logger.opt(exception=t.exception()).error("task {} crashed", t.get_name())
    logger.info("shutting down")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await rest.close()
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
