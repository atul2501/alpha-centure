from datetime import datetime, timezone
from decimal import Decimal

import pytest

from alpha.binance.parse import (
    FlowAggregator, LatencyStats, book_tick_row, interval_ms, liquidation_row, ms_to_dt, orderbook_row,
    rest_kline_to_row, ws_kline_to_row,
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


def test_book_tick_row_depth_and_reach():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bids = [["100.0", "2"], ["99.9", "1"]]
    asks = [["100.1", "3"], ["100.3", "1"]]
    row = book_tick_row("BTCUSDT.P", ts, bids, asks)
    sym, t, bb, ba, bq, aq, bquote, aquote, breach, areach = row
    assert (sym, t, bb, ba, bq, aq) == ("BTCUSDT.P", ts, 100.0, 100.1, 2.0, 3.0)
    assert bquote == pytest.approx(200.0 + 99.9) and aquote == pytest.approx(300.3 + 100.3)
    mid = 100.05
    assert breach == pytest.approx((mid - 99.9) / mid * 1e4) and areach == pytest.approx((100.3 - mid) / mid * 1e4)
    assert book_tick_row("X", ts, [], asks) is None


def test_latency_stats_emits_finished_minutes_only():
    lat = LatencyStats()
    base = 1_767_225_600_000  # 2026-01-01 00:00 UTC
    for i, d in enumerate([10, 20, 30, 40]):
        assert lat.add("aggtrade", base + i * 1000, base + i * 1000 + d) == []
    rows = lat.add("aggtrade", base + 60_000, base + 60_005)  # first sample of the next minute
    assert len(rows) == 1
    minute, stream, n, p50, p95, mx = rows[0]
    assert (minute, stream, n, mx) == (datetime(2026, 1, 1, tzinfo=timezone.utc), "aggtrade", 4, 40.0)
    assert p50 == 30.0 and p95 == 40.0
    assert [r[2] for r in lat.flush()] == [1]
