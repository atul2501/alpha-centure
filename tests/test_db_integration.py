"""Runs against a real Postgres when TEST_DATABASE_URL is set (e.g. postgresql://localhost/alpha_test)."""

import os

import pytest

from alpha.binance.parse import rest_kline_to_row
from alpha.db import DB

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")


async def test_candle_upsert_is_idempotent():
    db = await DB.connect(DSN)
    try:
        await db.init_schema()
        await db.pool.execute("DELETE FROM candles WHERE symbol = 'TESTUSDT'")
        k = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000059999, "15", 3, "4", "6", "0"]
        row = rest_kline_to_row("TESTUSDT", "1m", k)
        await db.upsert("candles", [row])
        await db.upsert("candles", [row])
        changed = list(row)
        changed[7] = changed[7] + 1  # close revised
        await db.upsert("candles", [tuple(changed)])
        rows = await db.pool.fetch("SELECT close FROM candles WHERE symbol = 'TESTUSDT'")
        assert len(rows) == 1 and rows[0]["close"] == row[7] + 1
    finally:
        await db.pool.execute("DELETE FROM candles WHERE symbol = 'TESTUSDT'")
        await db.close()
