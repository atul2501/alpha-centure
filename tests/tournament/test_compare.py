"""Multi-coin loader and the COMPARE-2025-10 folds (uses the dev DB; skipped when it is not reachable)."""

import pandas as pd
import psycopg
import pytest

from alpha.config import get_settings
from alpha.tournament.data import dataset as dsm
from alpha.tournament.experiments import compare
from alpha.tournament.validation.guard import HeldOutAccess
from alpha.tournament.validation.splits import walk_forward


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, connect_timeout=3)
    except Exception:
        pytest.skip("dev database not reachable")
    yield c
    c.close()


def test_target_coin_is_mapped_into_sol_columns(conn):
    end = pd.Timestamp("2023-02-01", tz="UTC")
    btc = dsm.load_raw(conn, end=end, start=pd.Timestamp("2023-01-01", tz="UTC"), symbol="BTCUSDT")
    assert len(btc) > 2000
    pd.testing.assert_series_equal(btc["sol_close"], btc["btc_close"], check_names=False)
    sol = dsm.load_raw(conn, end=end, start=pd.Timestamp("2023-01-01", tz="UTC"))
    assert (sol["sol_close"] < 100).mean() > 0.9 and (btc["sol_close"] > 10_000).all()


def test_forward_needs_explicit_permission(conn):
    with pytest.raises(HeldOutAccess):
        dsm.load_raw(conn, end=compare.END, symbol="ETHUSDT")


def test_compare_folds_cover_the_window_only():
    folds = walk_forward(pd.Timestamp("2020-01-01", tz="UTC"), oos_start=compare.START, end=compare.END)
    assert len(folds) == 4
    assert folds[0].test_start == compare.START and folds[-1].test_end == compare.END
    assert all(f.val_end < f.test_start for f in folds)
