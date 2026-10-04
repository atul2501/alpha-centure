"""Live/research parity: the live predictor builds features from a short recent window; research builds them
from full history. For the same bar and setup they must match. Uses the dev DB; skipped without data."""

import numpy as np
import pandas as pd
import psycopg
import pytest

from alpha.binance.parse import interval_td
from alpha.config import get_settings
from alpha.features.build import CORE_COLS, build_frame
from alpha.predict import LIVE_HISTORY_BARS
from alpha.research.dataset import with_features
from alpha.strategies.playbook import all_setups


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, connect_timeout=3)
    except Exception:
        pytest.skip("dev database not reachable")
    if c.execute("SELECT count(*) FROM candles WHERE symbol = 'BTCUSDT.P' AND interval = '1h'").fetchone()[0] < 3000:
        pytest.skip("not enough perp candles in dev database")
    yield c
    c.close()


@pytest.mark.parametrize("symbol,tf", [("BTCUSDT.P", "1h"), ("SOLUSDT.P", "15m")])
def test_live_window_matches_full_history(conn, symbol, tf):
    full = build_frame(conn, symbol, tf)
    full_setups = all_setups(full, tf)
    recent = full_setups[full_setups.index >= full.index[-300]]
    assert len(recent) > 0, "no recent setups to compare"
    step = interval_td(tf)
    checked = 0
    for t in recent.index.unique()[-5:]:
        live = build_frame(conn, symbol, tf, start=t - step * LIVE_HISTORY_BARS, end=t + step)
        ls = all_setups(live, tf)
        ls, fs = ls[ls.index == t], full_setups[full_setups.index == t]
        assert list(ls["strategy"]) == list(fs["strategy"]), f"different setups fired at {t}"
        a = with_features(ls, live, symbol, tf, CORE_COLS)
        b = with_features(fs, full, symbol, tf, CORE_COLS)
        num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and c != "max_bars"]
        np.testing.assert_allclose(a[num].to_numpy(float), b[num].to_numpy(float), rtol=1e-4, atol=1e-8,
                                   err_msg=f"{symbol} {tf} {t}")
        checked += 1
    assert checked > 0
