"""Perp microstructure from the USD-M futures websocket:

- aggTrade         -> 1m buy/sell volume & delta (flow_1m, stored as BTCUSDT.P)
- depth20@100ms    -> 1m top-20 book snapshot (orderbook_snap) + top-of-book every BOOK_TICK_SECONDS (book_tick)
- markPrice@1s     -> mark / index / funding once a minute (premium_snap, plain symbol like the other futures stats)
- every message    -> delivery latency per minute and stream kind (ws_latency)

Binance serves depth streams only on the /public route and trades / mark price only on /market, so this runs
two connections ("flow" and "book") that share one handler.
"""

import asyncio

from loguru import logger

from alpha.binance.parse import (FlowAggregator, LatencyStats, book_tick_row, ms_to_dt, now_ms, orderbook_row)
from alpha.binance.ws import combined_url, run_stream
from alpha.config import Settings, perp_symbol
from alpha.db import DB

FLUSH_EVERY_S = 5


async def run_orderflow(db: DB, settings: Settings) -> None:
    flow_streams = [f"{s.lower()}@{k}" for s in settings.symbols for k in ("aggTrade", "markPrice@1s")]
    book_streams = [f"{s.lower()}@depth20@100ms" for s in settings.symbols]
    flow = FlowAggregator()
    latency = LatencyStats()
    tick_ms = max(1, settings.book_tick_seconds) * 1000
    last_book_minute: dict[str, int] = {}
    last_tick: dict[str, int] = {}
    last_mark_minute: dict[str, int] = {}
    ticks: list[tuple] = []

    async def write_flow(rows: list[tuple]) -> None:
        if rows:
            await db.upsert("flow_1m", rows)
            for r in rows:
                await db.log_fetch("flow", "ws", symbol=r[0], interval="1m", ref_time=r[1], rows=1,
                                   status="ok" if r[-1] else "partial")

    async def flusher() -> None:
        while True:
            await asyncio.sleep(FLUSH_EVERY_S)
            batch = ticks[:]
            ticks.clear()
            try:
                await db.upsert("book_tick", batch)
            except Exception:
                logger.exception("book_tick flush failed")

    async def flow_connect() -> None:
        await write_flow(flow.reset())
        await db.log_fetch("ws", "ws", interval="flow_perp", status="connected")

    async def flow_disconnect(reason: str) -> None:
        await write_flow(flow.reset())
        await db.log_fetch("ws", "ws", interval="flow_perp", status="disconnected", error=reason)

    async def book_connect() -> None:
        last_book_minute.clear()
        await db.log_fetch("ws", "ws", interval="book_perp", status="connected")

    async def book_disconnect(reason: str) -> None:
        await db.log_fetch("ws", "ws", interval="book_perp", status="disconnected", error=reason)

    async def on_message(stream: str, data: dict) -> None:
        kind = stream.split("@", 2)[1].lower()  # aggtrade | depth20 | markprice
        done = latency.add(kind, data["E"], now_ms())
        if done:
            await db.upsert("ws_latency", done)
        symbol = data["s"]
        if kind == "aggtrade":
            closed = flow.add_trade(perp_symbol(symbol), data["T"], float(data["p"]), float(data["q"]), data["m"])
            if closed:
                await write_flow([closed])
        elif kind == "depth20":
            event = data["E"]
            bucket = event - event % tick_ms
            if last_tick.get(symbol) != bucket:
                last_tick[symbol] = bucket
                row = book_tick_row(perp_symbol(symbol), ms_to_dt(bucket), data["b"], data["a"])
                if row:
                    ticks.append(row)
            minute = event // 60_000
            if last_book_minute.get(symbol) != minute:
                last_book_minute[symbol] = minute
                row = orderbook_row(perp_symbol(symbol), ms_to_dt(minute * 60_000), data["b"], data["a"])
                if row:
                    await db.upsert("orderbook_snap", [row])
                    await db.log_fetch("orderbook", "ws", symbol=row[0], ref_time=row[1], rows=1)
        elif kind == "markprice":
            minute = data["E"] // 60_000
            if last_mark_minute.get(symbol) != minute:
                last_mark_minute[symbol] = minute
                row = (symbol, ms_to_dt(data["E"]), float(data["p"]), float(data["i"]), float(data["r"]),
                       ms_to_dt(data["T"]) if data.get("T") else None)
                await db.upsert("premium_snap", [row])

    flush_task = asyncio.create_task(flusher(), name="book_tick_flush")
    try:
        await asyncio.gather(
            run_stream("flow_perp", combined_url(settings.futures_ws, flow_streams), on_message, flow_connect,
                       flow_disconnect),
            run_stream("book_perp", combined_url(settings.futures_ws_public, book_streams), on_message, book_connect,
                       book_disconnect),
        )
    finally:
        flush_task.cancel()
        await db.upsert("ws_latency", latency.flush())
