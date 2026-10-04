import json
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
from loguru import logger

from alpha.binance.parse import CANDLE_COLS

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

# table -> (columns, primary key columns)
TABLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "candles": (CANDLE_COLS, ("symbol", "interval", "open_time")),
    "funding_rate": (("symbol", "funding_time", "funding_rate", "mark_price"), ("symbol", "funding_time")),
    "premium_snap": (("symbol", "ts", "mark_price", "index_price", "last_funding_rate", "next_funding_time"), ("symbol", "ts")),
    "open_interest": (("symbol", "ts", "sum_open_interest", "sum_open_interest_value"), ("symbol", "ts")),
    "long_short_ratio": (("symbol", "ts", "long_short_ratio", "long_account", "short_account"), ("symbol", "ts")),
    "taker_ratio": (("symbol", "ts", "buy_sell_ratio", "buy_vol", "sell_vol"), ("symbol", "ts")),
    "liquidations": (("symbol", "ts", "side", "price", "avg_price", "qty", "filled_qty", "status"), ("symbol", "ts", "side", "price", "qty")),
    "flow_1m": (("symbol", "minute", "buy_vol", "sell_vol", "delta", "buy_quote", "sell_quote", "trades", "max_trade_quote", "complete"), ("symbol", "minute")),
    "orderbook_snap": (("symbol", "ts", "best_bid", "best_ask", "spread_bps", "bid_qty_20", "ask_qty_20", "imbalance", "bids", "asks"), ("symbol", "ts")),
    "signals": (("symbol", "tf", "bar_time", "strategy", "side", "model_version", "action", "reason", "regime",
                 "regime_probs", "p_win", "ev_r", "close_px", "stop", "target", "max_bars", "features", "exit_policy"),
                ("symbol", "tf", "bar_time", "strategy", "side", "model_version")),
}


def upsert_sql(table: str) -> str:
    cols, pk = TABLES[table]
    placeholders = ", ".join(f"${i}" for i in range(1, len(cols) + 1))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in pk)
    if table == "candles":
        updates += ", ingested_at = now()"
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {updates}"
    )


async def _init_conn(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


class DB:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "DB":
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=8, init=_init_conn)
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def init_schema(self) -> None:
        async with self.pool.acquire() as conn:
            for path in sorted(SQL_DIR.glob("*.sql")):
                await conn.execute(path.read_text())
        logger.info("schema ready")

    async def upsert(self, table: str, rows: list[tuple]) -> int:
        if not rows:
            return 0
        async with self.pool.acquire() as conn:
            await conn.executemany(upsert_sql(table), rows)
        return len(rows)

    async def log_fetch(
        self,
        kind: str,
        source: str,
        *,
        symbol: str | None = None,
        interval: str | None = None,
        ref_time: datetime | None = None,
        rows: int = 0,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        latency_ms = None
        if ref_time is not None:
            latency_ms = int((datetime.now(timezone.utc) - ref_time).total_seconds() * 1000)
        try:
            await self.pool.execute(
                "INSERT INTO fetch_log (kind, symbol, interval, ref_time, source, rows, latency_ms, status, error) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
                kind, symbol, interval, ref_time, source, rows, latency_ms, status, error and error[:1000],
            )
        except Exception as e:  # audit logging must never take the collector down
            logger.warning("fetch_log insert failed: {}", e)

    async def last_time(self, table: str, time_col: str, symbol: str, interval: str | None = None,
                        before: datetime | None = None) -> datetime | None:
        """Newest time for symbol (and interval), optionally only counting rows older than `before`."""
        sql = f"SELECT max({time_col}) FROM {table} WHERE symbol = $1 AND ($2::timestamptz IS NULL OR {time_col} < $2)"
        if interval is None:
            return await self.pool.fetchval(sql, symbol, before)
        return await self.pool.fetchval(sql + " AND interval = $3", symbol, before, interval)

    async def prune_fetch_log(self, days: int = 30) -> None:
        await self.pool.execute("DELETE FROM fetch_log WHERE fetched_at < now() - make_interval(days => $1)", days)
