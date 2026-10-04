"""Spot order flow (aggTrade -> 1m buy/sell volume & delta) and per-minute order book snapshots."""

from loguru import logger

from alpha.binance.parse import FlowAggregator, ms_to_dt, now_ms, orderbook_row
from alpha.binance.ws import combined_url, run_stream
from alpha.config import Settings
from alpha.db import DB


async def run_orderflow(db: DB, settings: Settings) -> None:
    streams = [f"{s.lower()}@aggTrade" for s in settings.symbols] + [f"{s.lower()}@depth20" for s in settings.symbols]
    flow = FlowAggregator()
    last_book_minute: dict[str, int] = {}

    async def write_flow(rows: list[tuple]) -> None:
        if rows:
            await db.upsert("flow_1m", rows)
            for r in rows:
                await db.log_fetch("flow", "ws", symbol=r[0], interval="1m", ref_time=r[1], rows=1,
                                   status="ok" if r[-1] else "partial")

    async def on_connect() -> None:
        last_book_minute.clear()
        await write_flow(flow.reset())
        await db.log_fetch("ws", "ws", interval="orderflow", status="connected")

    async def on_disconnect(reason: str) -> None:
        await write_flow(flow.reset())
        await db.log_fetch("ws", "ws", interval="orderflow", status="disconnected", error=reason)

    async def on_message(stream: str, data: dict) -> None:
        if stream.lower().endswith("@aggtrade"):
            closed = flow.add_trade(data["s"], data["T"], float(data["p"]), float(data["q"]), data["m"])
            if closed:
                await write_flow([closed])
            return
        # depth20 payload carries no symbol or time: take them from the stream name / local clock.
        symbol = stream.split("@", 1)[0].upper()
        minute = now_ms() // 60_000
        if last_book_minute.get(symbol) == minute:
            return
        last_book_minute[symbol] = minute
        row = orderbook_row(symbol, ms_to_dt(minute * 60_000), data["bids"], data["asks"])
        if row:
            await db.upsert("orderbook_snap", [row])
            await db.log_fetch("orderbook", "ws", symbol=symbol, ref_time=row[1], rows=1)
            logger.debug("book {} imb={:.3f}", symbol, row[7])

    await run_stream("orderflow", combined_url(settings.spot_ws, streams), on_message, on_connect, on_disconnect)
