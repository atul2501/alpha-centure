"""Live closed candles over one combined websocket per market (spot, perp)."""

import asyncio
from datetime import datetime, timezone

from loguru import logger

from alpha.backfill import backfill_all_candles
from alpha.binance.parse import ws_kline_to_row
from alpha.binance.rest import BinanceREST
from alpha.binance.ws import combined_url, run_stream
from alpha.config import Settings
from alpha.db import DB


async def run_klines(db: DB, rest: BinanceREST, settings: Settings, market: str = "spot") -> None:
    feeds = settings.candle_feeds(market)
    if not feeds:
        return
    name = f"klines_{market}"
    db_symbol = {f.api_symbol: f.db_symbol for f in feeds}
    streams = [f"{f.api_symbol.lower()}@kline_{f.interval}" for f in feeds]
    base = settings.futures_ws if market == "perp" else settings.spot_ws
    background: set[asyncio.Task] = set()

    async def on_connect() -> None:
        await db.log_fetch("ws", "ws", interval=name, status="connected")
        # Fill whatever was missed while disconnected. Runs alongside the stream; upserts are idempotent.
        connected_at = datetime.now(timezone.utc)
        task = asyncio.create_task(backfill_all_candles(db, rest, settings, market, connected_at))
        background.add(task)
        task.add_done_callback(background.discard)

    async def on_disconnect(reason: str) -> None:
        await db.log_fetch("ws", "ws", interval=name, status="disconnected", error=reason)

    async def on_message(stream: str, data: dict) -> None:
        k = data["k"]
        if not k["x"]:  # candle still forming: never store
            return
        row = ws_kline_to_row(k, db_symbol[k["s"]])
        await db.upsert("candles", [row])
        await db.log_fetch("candle", "ws", symbol=row[0], interval=row[1], ref_time=row[3], rows=1)
        logger.debug("closed {} {} {} c={}", row[0], row[1], row[2], row[7])

    await run_stream(name, combined_url(base, streams), on_message, on_connect, on_disconnect)
