"""Leak test for the DB feature builder. Uses the dev DB (DATABASE_URL); skipped when it has no data."""

import numpy as np
import pandas as pd
import psycopg
import pytest

from alpha.config import get_settings
from alpha.features.build import CORE_COLS, build_frame


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, connect_timeout=3)
    except Exception:
        pytest.skip("dev database not reachable")
    n = c.execute("SELECT count(*) FROM candles WHERE symbol = 'ETHUSDT' AND interval = '1h'").fetchone()[0]
    if n < 1000:
        pytest.skip("not enough candles in dev database")
    yield c
    c.close()


@pytest.mark.parametrize("symbol,tf", [("ETHUSDT", "1h"), ("SOLUSDT.P", "15m")])
def test_build_frame_row_identical_when_future_removed(conn, symbol, tf):
    full = build_frame(conn, symbol, tf)
    assert len(full) > 500
    for t in full.index[[-400, -150, -2]]:
        cut = build_frame(conn, symbol, tf, end=t + pd.Timedelta(seconds=1))
        a = full.loc[t, CORE_COLS].astype(float)
        b = cut.loc[t, CORE_COLS].astype(float)
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-9, obj=f"{symbol} {tf} {t}")
    assert full[CORE_COLS].iloc[-50:].notna().mean().min() > 0.9, "core features mostly missing"


def test_frame_is_utc_regardless_of_session_timezone(conn):
    conn.execute("SET TIME ZONE 'Asia/Kolkata'")
    f = build_frame(conn, "BTCUSDT", "1h")
    assert str(f.index.tz) == "UTC" and str(f["close_time"].dt.tz) == "UTC"
    assert (f["hour"] == f.index.hour).all() and f.index[-1].minute == 0
