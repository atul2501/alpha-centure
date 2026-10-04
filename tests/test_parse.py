from datetime import timezone
from decimal import Decimal

import pytest

from alpha.binance.parse import (
    FlowAggregator, interval_ms, liquidation_row, ms_to_dt, orderbook_row, rest_kline_to_row, ws_kline_to_row,
)
from alpha.db import upsert_sql

REST_K = [1700000000000, "100.5", "101", "99.5", "100.8", "12.5", 1700000059999, "1260.1", 42, "7.5", "756.0", "0"]
WS_K = {"t": 1700000000000, "T": 1700000059999, "s": "BTCUSDT", "i": "1m", "o": "100.5", "c": "100.8",
        "h": "101", "l": "99.5", "v": "12.5", "n": 42, "x": True, "q": "1260.1", "V": "7.5", "Q": "756.0"}


def test_rest_and_ws_kline_produce_same_row_except_source():
    r = rest_kline_to_row("BTCUSDT", "1m", REST_K)
    w = ws_kline_to_row(WS_K)
    assert r[:-1] == w[:-1]
    assert (r[-1], w[-1]) == ("rest", "ws")
    assert r[4] == Decimal("100.5") and r[10] == 42
    assert r[2].tzinfo == timezone.utc


def test_interval_ms():
    assert interval_ms("4h") == 4 * 3600 * 1000
    assert interval_ms("1w") == 7 * 86400 * 1000
    with pytest.raises(ValueError):
        interval_ms("7m")


def test_orderbook_row_imbalance_and_spread():
    row = orderbook_row("BTCUSDT", ms_to_dt(0), [["100", "3"], ["99", "1"]], [["101", "1"], ["102", "1"]])
    _, _, bid, ask, spread_bps, bq, aq, imb, *_ = row
    assert (bid, ask, bq, aq) == (100.0, 101.0, 4.0, 2.0)
    assert imb == pytest.approx(1 / 3)
    assert spread_bps == pytest.approx(1 / 100.5 * 10_000)
    assert orderbook_row("X", ms_to_dt(0), [], [["1", "1"]]) is None


def test_liquidation_row():
    o = {"s": "ETHUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "2", "p": "3000", "ap": "2999",
         "X": "FILLED", "l": "2", "z": "2", "T": 1700000000000}
    assert liquidation_row(o)[1:] == (ms_to_dt(1700000000000), "SELL", 3000.0, 2999.0, 2.0, 2.0, "FILLED")


def test_flow_aggregator_buckets_and_completeness():
    f = FlowAggregator()
    m0, m1, m2 = 60_000 * 100, 60_000 * 101, 60_000 * 102
    assert f.add_trade("BTC", m0 + 5, 10.0, 1.0, False) is None   # aggressive buy
    assert f.add_trade("BTC", m0 + 9, 10.0, 0.5, True) is None    # aggressive sell
    row0 = f.add_trade("BTC", m1 + 1, 10.0, 2.0, False)
    # first minute after connect is partial
    assert row0[1] == ms_to_dt(m0) and row0[2:5] == (1.0, 0.5, 0.5) and row0[7] == 2 and row0[-1] is False
    row1 = f.add_trade("BTC", m2, 10.0, 1.0, True)
    assert row1[1] == ms_to_dt(m1) and row1[2] == 2.0 and row1[-1] is True
    # reconnect: open bucket flushed as partial and next minute partial again
    flushed = f.reset()
    assert len(flushed) == 1 and flushed[0][-1] is False
    f.add_trade("BTC", m2 + 60_000, 10.0, 1.0, False)
    assert f.add_trade("BTC", m2 + 120_000, 10.0, 1.0, False)[-1] is False


def test_upsert_sql_candles_refreshes_ingested_at():
    sql = upsert_sql("candles")
    assert "ON CONFLICT (symbol, interval, open_time)" in sql
    assert "ingested_at = now()" in sql
    assert "symbol = EXCLUDED.symbol" not in sql
