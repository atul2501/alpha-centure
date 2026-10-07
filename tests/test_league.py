"""Shadow league: candidate books respect the caps, the cost band only trades at rebalances, the pre-registered
switch rule, and league P6 == report.shadow (the paper report's research path) on the live DB."""

import numpy as np
import pandas as pd
import psycopg
import pytest

from alpha.config import get_settings
from alpha.live import league
from alpha.live.report import shadow, shadow_from_weights
from alpha.research.phase4 import MAX_COIN, MAX_GROSS

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def _panel(days: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=days * 24, freq="h", tz="UTC")
    parts = []
    for i, s in enumerate(SYMS):
        ret = rng.normal(0.0001 * (i - 1), 0.01, len(t))
        parts.append(pd.DataFrame({"symbol": s, "ret": ret, "close": 100 * np.exp(np.cumsum(ret)),
                                   "eligible": True, "funding_rate": np.where(t.hour % 8 == 0, 1e-4, 0.0)},
                                  index=pd.Index(t, name="time")))
    return pd.concat(parts).set_index("symbol", append=True).sort_index()


def _wide(panel, rng, scale=1.0):
    t = panel.index.get_level_values("time").unique()
    return pd.DataFrame(rng.normal(0, scale, (len(t), len(SYMS))), index=t, columns=SYMS)


@pytest.fixture(scope="module")
def books():
    panel = _panel()
    rng = np.random.default_rng(1)
    sc, carry = _wide(panel, rng), _wide(panel, rng)
    costs = pd.DataFrame({"maker_bps": 1.0, "taker_bps": 6.0, "spread_bps": 2.0}, index=SYMS)
    return panel, costs, league.candidates(panel, {"xscarry": carry}, sc, costs)


def test_candidates_are_the_preregistered_set(books):
    assert list(books[2]) == ["P6", "R1_ensemble", "R5_cost_band", "R6_carry_sleeve", "N1_R1_revol", "BENCH_BTC"]


def test_weights_respect_caps(books):
    for name, (w, _) in books[2].items():
        assert (w.abs().sum(axis=1) <= MAX_GROSS + 1e-9).all(), name
        assert (w.abs() <= MAX_COIN + 1e-9).all().all() or name == league.BENCHMARK, name


def test_cost_band_trades_only_at_rebalances(books):
    panel, costs, c = books
    w, band_fn = c["R5_cost_band"]
    ret = panel["ret"].unstack("symbol")
    out = band_fn(w, ret, ret.rolling(168, min_periods=48).std(), 72)
    hours = (out.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
    changed = out.diff().abs().sum(axis=1) > 0
    assert changed.any() and (hours[changed.to_numpy()] % 72 == 0).all()


def test_every_candidate_simulates(books):
    panel, costs, c = books
    start, end = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")
    for name, (w, band_fn) in c.items():
        d = shadow_from_weights(panel, w, costs, start, end, band_fn)
        assert len(d) == 30 and np.isfinite(d["net"]).all(), name


def _daily(ref, **others):
    return pd.DataFrame({"P6": ref, "BENCH_BTC": ref * 0, **others})


def test_rule_keeps_p6_on_a_tie_and_noise():
    rng = np.random.default_rng(2)
    p = pd.Series(rng.normal(0.001, 0.01, 90))
    r = league.rule(_daily(p, SAME=p, NOISE=p + rng.normal(0, 0.01, 90)))
    assert not r["switch"].any()
    assert "BENCH_BTC" not in r.index


def test_rule_switches_only_on_a_clear_win_without_extra_drawdown():
    rng = np.random.default_rng(3)
    p = pd.Series(rng.normal(0.0, 0.004, 180))
    clear = p + 0.003 + rng.normal(0, 0.001, 180)
    crash = clear.copy()
    crash.iloc[100:110] -= 0.015  # same total edge, but a 15% drawdown on the way
    crash.iloc[110:120] += 0.015
    r = league.rule(_daily(p, CLEAR=clear, CRASH=crash))
    assert r.loc["CLEAR", "switch"]
    assert r.loc["CRASH", "t_ok"] and not r.loc["CRASH", "dd_ok"] and not r.loc["CRASH", "switch"]


def test_league_p6_equals_report_shadow():
    s = get_settings()
    try:
        conn = psycopg.connect(s.database_url, connect_timeout=3)
    except Exception:
        pytest.skip("dev database not reachable")
    with conn:
        start, end = pd.Timestamp("2026-10-02", tz="UTC"), pd.Timestamp("2026-10-06", tz="UTC")
        if conn.execute("SELECT count(*) FROM candles WHERE symbol = 'BTCUSDT.P' AND interval = '1h' "
                        "AND open_time >= %s AND open_time < %s", (start, end)).fetchone()[0] < 90:
            pytest.skip("no perp candles for the parity window")
        costs = pd.DataFrame({"maker_bps": 1.0, "taker_bps": 6.0, "spread_bps": 2.0}, index=s.symbols)
        df = league.run(conn, s.symbols, s.models_dir, start, end, costs)
        ref = shadow(conn, s.symbols, start, end, costs)
    lg = df[df["strategy"] == "P6"].set_index("day")["net"]
    assert len(lg) == 4
    np.testing.assert_allclose(lg.to_numpy(), ref.to_numpy(), atol=1e-12)
