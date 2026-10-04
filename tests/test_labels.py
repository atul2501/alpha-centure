import numpy as np
import pandas as pd
import pytest

from alpha.strategies.labels import Costs, _count_settlements, label_setups

NO_COST = Costs(fee=0.0, slippage=0.0)


def frame(bars, start="2026-01-01 01:00"):
    """bars: list of (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(bars), freq="1h", tz="UTC")
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx).assign(funding=0.0)


def setup(f, i, side, stop, target, max_bars=5):
    return pd.DataFrame({"strategy": "t", "side": side, "stop": stop, "target": target, "max_bars": max_bars},
                        index=f.index[[i]])


def test_long_hits_target():
    f = frame([(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 104, 100, 103), (103, 103, 103, 103)])
    r = label_setups(setup(f, 0, 1, 98, 103), f, NO_COST).iloc[0]
    assert r["outcome"] == "target" and r["entry"] == 100 and r["exit"] == 103
    assert r["bars_held"] == 2 and r["r"] == pytest.approx(1.5) and r["win"]


def test_short_hits_stop():
    f = frame([(100, 100, 100, 100), (100, 100.5, 99, 100), (100, 102.5, 99, 102), (102, 102, 102, 102)])
    r = label_setups(setup(f, 0, -1, 102, 96), f, NO_COST).iloc[0]
    assert r["outcome"] == "stop" and r["exit"] == 102 and r["r"] == pytest.approx(-1.0) and not r["win"]


def test_same_bar_tie_counts_as_stop():
    f = frame([(100, 100, 100, 100), (100, 105, 95, 100), (100, 100, 100, 100)])
    assert label_setups(setup(f, 0, 1, 97, 103), f, NO_COST).iloc[0]["outcome"] == "stop"


def test_timeout_exits_at_close():
    f = frame([(100, 100, 100, 100)] + [(100, 100.5, 99.5, 100.2)] * 3 + [(100, 100, 100, 100)] * 2)
    r = label_setups(setup(f, 0, 1, 98, 110, max_bars=3), f, NO_COST).iloc[0]
    assert r["outcome"] == "timeout" and r["exit"] == pytest.approx(100.2) and r["bars_held"] == 3


def test_gap_through_stop_drops_setup_and_end_of_data_drops():
    f = frame([(100, 100, 100, 100), (97, 98, 96, 97), (97, 97, 97, 97)])
    assert label_setups(setup(f, 0, 1, 98, 103), f, NO_COST).empty
    assert label_setups(setup(f, 1, 1, 90, 110, max_bars=5), f, NO_COST).empty


def test_costs_and_funding_applied():
    # entry 01:00+1h = 02:00, exit bar 09:00 -> crosses the 08:00 settlement once
    bars = [(100, 100, 100, 100)] * 12
    bars[8] = (100, 103, 100, 102)
    f = frame(bars).assign(funding=0.001)
    r = label_setups(setup(f, 0, 1, 98, 102, max_bars=10), f, Costs(fee=0.0005, slippage=0.0001)).iloc[0]
    assert r["exit_time"] == pd.Timestamp("2026-01-01 09:00", tz="UTC")
    assert r["funding_cost"] == pytest.approx(0.001)
    assert r["pnl"] == pytest.approx(0.02 - 0.0012 - 0.001)


def test_count_settlements():
    t = lambda s: pd.Timestamp(s, tz="UTC")
    assert _count_settlements(t("2026-01-01 07:00"), t("2026-01-01 08:00"), (0, 8, 16)) == 1
    assert _count_settlements(t("2026-01-01 08:00"), t("2026-01-01 15:00"), (0, 8, 16)) == 0
    assert _count_settlements(t("2026-01-01 23:00"), t("2026-01-02 16:00"), (0, 8, 16)) == 3


def test_gap_through_stop_fills_at_open():
    f = frame([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (95, 96, 94, 95), (95, 95, 95, 95)])
    r = label_setups(setup(f, 0, 1, 98, 105), f, NO_COST).iloc[0]
    assert r["outcome"] == "stop" and r["exit"] == 95 and r["r"] == pytest.approx(-2.5)


def test_duplicate_signal_times_do_not_multiply_rows():
    f = frame([(100, 100, 100, 100), (100, 104, 99.5, 103), (103, 103, 103, 103), (103, 103, 103, 103)])
    s = pd.concat([setup(f, 0, 1, 98, 103), setup(f, 0, 1, 99, 102), setup(f, 0, -1, 102, 97)])
    out = label_setups(s, f, NO_COST)
    assert len(out) == 3 and list(out["outcome"]) == ["target", "target", "stop"]
