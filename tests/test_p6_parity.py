"""Live P6 targets (200-day window, as the paper engine computes them) == research targets on full history."""

import numpy as np
import pandas as pd
import psycopg
import pytest

from alpha.config import get_settings
from alpha.research import signals as sig
from alpha.research.models import feature_frame, target
from alpha.research.panel import build_panel
from alpha.strategy import p6

T = pd.Timestamp("2026-09-20 00:00", tz="UTC")  # a rebalance bar? any closed hour works for parity


@pytest.fixture(scope="module")
def setup():
    s = get_settings()
    try:
        conn = psycopg.connect(s.database_url, connect_timeout=3)
    except Exception:
        pytest.skip("dev database not reachable")
    if conn.execute("SELECT count(*) FROM candles WHERE symbol = 'BTCUSDT.P' AND interval = '1h' "
                    "AND open_time > '2026-01-01'").fetchone()[0] < 5000:
        pytest.skip("not enough recent perp candles")
    full = build_panel(conn, s.symbols, end=T + pd.Timedelta(hours=1))
    X, vol = feature_frame(full, sig.compute(full))
    b = p6.fit(X, target(full, vol, p6.H), pd.Timestamp("2026-06-01", tz="UTC"))
    yield conn, s, full, X, b
    conn.close()


def test_live_window_targets_equal_full_history(setup):
    conn, s, full, X, b = setup
    w_full = p6.target_weights(full, p6.score(b, X))
    last, w_live, _ = p6.live_targets(conn, s.symbols, b, T + pd.Timedelta(hours=1, seconds=90))
    assert last == T
    a = w_full.loc[T].reindex(w_live.index).fillna(0.0)
    np.testing.assert_allclose(w_live.to_numpy(), a.to_numpy(), atol=1e-9)
    assert w_live.abs().sum() > 0.05  # not trivially zero
