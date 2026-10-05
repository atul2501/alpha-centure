import io
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from alpha.binance.vision import (AGG_COLS, Job, agg_trades_to_flow, days, months, parse_book_depth, parse_klines,
                                  parse_metrics, parse_premium, plan)

K1 = "1704067200000,42283.5,42300.0,42200.1,42250.3,120.5,1704067259999,5091234.5,1500,60.2,2543210.1,0\n"
K2 = "1704067260000,42250.3,42260.0,42240.0,42255.0,80.0,1704067319999,3380400.0,900,30.0,1267650.0,0\n"
HDR = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
UTC = timezone.utc


@pytest.mark.parametrize("text", [K1 + K2, HDR + K1 + K2])
def test_parse_klines_with_and_without_header(text):
    rows = parse_klines(io.BytesIO(text.encode()), "BTCUSDT.P", "1m")
    assert len(rows) == 2
    r = rows[0]
    assert r[:4] == ("BTCUSDT.P", "1m", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, 0, 0, 59, 999000, tzinfo=UTC))
    assert r[4:10] == ("42283.5", "42300.0", "42200.1", "42250.3", "120.5", "5091234.5")  # exact strings
    assert r[10:] == (1500, "60.2", "2543210.1", "vision")


def test_microsecond_timestamps_are_normalized():
    us = K1.replace("1704067200000", "1704067200000000").replace("1704067259999", "1704067259999999")
    rows = parse_klines(io.BytesIO(us.encode()), "X.P", "1m")
    assert rows[0][2] == datetime(2024, 1, 1, tzinfo=UTC)


def test_parse_premium_floats():
    rows = parse_premium(io.BytesIO(K1.encode()), "BTCUSDT", "1h")
    assert rows[0][0:3] == ("BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=UTC)) and rows[0][7] == 42250.3


def test_parse_metrics():
    text = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
            "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
            "2024-01-01 00:00:00,BTCUSDT,74006.266,3131493738.8974,1.3682031,1.253668,1.50710938,\n")
    (row,) = parse_metrics(io.StringIO(text), "BTCUSDT")
    assert row[:3] == ("BTCUSDT", datetime(2024, 1, 1, tzinfo=UTC), 74006.266)
    assert row[6] == 1.50710938 and row[7] is None


def test_parse_book_depth_keeps_last_snapshot_per_5m():
    lines = ["timestamp,percentage,depth,notional"]
    for t, base in (("2024-01-01 00:00:10", 100), ("2024-01-01 00:04:40", 200), ("2024-01-01 00:05:10", 300)):
        for p in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
            lines.append(f"{t},{p},1,{base + abs(p) * (1 if p > 0 else -1)}")
    lines.append("2024-01-01 00:04:40,0.2,1,999")  # non-integer bands are ignored
    rows = parse_book_depth(io.StringIO("\n".join(lines)), "BTCUSDT.P")
    assert [r[1] for r in rows] == [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, 0, 5, tzinfo=UTC)]
    first = rows[0]
    assert first[2] == datetime(2024, 1, 1, 0, 4, 40, tzinfo=UTC)
    assert first[3:8] == (199.0, 198.0, 197.0, 196.0, 195.0) and first[8:] == (201.0, 202.0, 203.0, 204.0, 205.0)


def test_agg_trades_to_flow_merges_minutes_split_across_chunks():
    t0 = 1704067200000
    a = pd.DataFrame([[1, 100.0, 2.0, 0, 0, t0 + 1000, False], [2, 100.0, 1.0, 0, 0, t0 + 2000, True]], columns=AGG_COLS)
    b = pd.DataFrame([[3, 101.0, 3.0, 0, 0, t0 + 59_000, "true"], [4, 102.0, 1.0, 0, 0, t0 + 60_000, "False"]],
                     columns=AGG_COLS)
    rows = agg_trades_to_flow([a[["price", "quantity", "transact_time", "is_buyer_maker"]],
                               b[["price", "quantity", "transact_time", "is_buyer_maker"]]], "BTCUSDT.P")
    assert len(rows) == 2
    sym, minute, buy, sell, delta, bq, sq, n, mx, complete = rows[0]
    assert (sym, minute) == ("BTCUSDT.P", datetime(2024, 1, 1, tzinfo=UTC))
    assert (buy, sell, delta, bq, sq, n, mx, complete) == (2.0, 4.0, -2.0, 200.0, 403.0, 3, 303.0, True)
    assert rows[1][2] == 1.0 and rows[1][7] == 1


def test_plan_monthly_then_daily_and_respects_listing():
    today = date(2026, 3, 4)
    assert months(date(2025, 11, 1), date(2026, 3, 1)) == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert days(date(2026, 3, 1), date(2026, 3, 3)) == ["2026-03-01", "2026-03-02", "2026-03-03"]
    jobs = plan(["HYPEUSDT"], ["klines", "metrics"], ["1h"], date(2020, 1, 1), {"HYPEUSDT": date(2026, 1, 15)}, today)
    kl = [j.period for j in jobs if j.dataset == "klines"]
    assert kl == ["2026-01", "2026-02", "2026-03-01", "2026-03-02", "2026-03-03"]
    met = [j.period for j in jobs if j.dataset == "metrics"]
    assert met[0] == "2026-01-15" and met[-1] == "2026-03-03"
    j = Job("klines", "BTCUSDT", "2024-02", "1m")
    assert j.path == "monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-02.zip" and j.period_end == date(2024, 2, 29)
    assert Job("premium", "BTCUSDT", "2024-02-03", "5m").path == \
        "daily/premiumIndexKlines/BTCUSDT/5m/BTCUSDT-5m-2024-02-03.zip"
