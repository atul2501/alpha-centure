import numpy as np
import pandas as pd
import pytest

from alpha.exec.costs import CostModel, funding_crossed, gross_edge_clears
from alpha.research.scorecard import Card, deflated_sharpe, pbo_cscv, rank, score, verdict
from alpha.research.splits import DEV, VALID_A, LockboxError, dev_only, open_lockbox


# ---------- costs ----------

def test_taker_cost_components():
    cm = CostModel(latency_ms=0)
    # fee 5 + half spread 1 + impact 0.5 * 10k / 1M * 100 = 0.5
    assert cm.taker_bps(10_000, 2.0, 1_000_000) == pytest.approx(6.5)
    assert cm.stressed(2).taker_bps(10_000, 2.0, 1_000_000) == pytest.approx(13.0)
    lat = CostModel(latency_ms=1000).latency_bps(10.0)  # 0.4 * 10 bps/sqrt(s) * 1s
    assert lat == pytest.approx(4.0)


def test_maker_cost_defaults_to_no_spread_capture():
    assert CostModel().maker_bps(2.0) == pytest.approx(2.0)
    assert CostModel(adverse_selection_bps=0.5).maker_bps(2.0) == pytest.approx(1.5)


def test_funding_sign_and_window():
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00"], utc=True)
    rates = pd.Series([0.0001, 0.0002, -0.0001], index=idx)
    crossed = funding_crossed(rates, pd.Timestamp("2024-01-01 00:00", tz="UTC"), pd.Timestamp("2024-01-01 08:00", tz="UTC"))
    assert list(crossed) == [0.0002]                  # entry instant excluded, exit instant included
    assert CostModel().funding_bps(1, crossed) == pytest.approx(2.0)   # long pays positive funding
    assert CostModel().funding_bps(-1, crossed) == pytest.approx(-2.0)  # short receives it


def test_trade_gate():
    assert gross_edge_clears(15, 10) and not gross_edge_clears(14.9, 10) and not gross_edge_clears(np.nan, 1)


# ---------- splits / lockbox ----------

def test_dev_only_and_split_masks():
    idx = pd.to_datetime(["2023-12-31 00:00", "2024-06-30 23:59", "2024-07-01 00:00", "2025-09-30 00:00"], utc=True)
    df = pd.DataFrame({"x": range(4)}, index=idx)
    assert list(dev_only(df)["x"]) == [0, 1]
    assert list(df[VALID_A.mask(df.index)]["x"]) == [2, 3]
    assert not DEV.mask(idx[2:]).any()


def test_lockbox_once_per_candidate_and_max_three(tmp_path):
    ledger = tmp_path / "lockbox.jsonl"
    open_lockbox("a", "VALID-A", ledger)
    with pytest.raises(LockboxError):
        open_lockbox("a", "VALID-A", ledger)
    open_lockbox("b", "VALID-A", ledger)
    open_lockbox("c", "VALID-A", ledger)
    with pytest.raises(LockboxError):
        open_lockbox("d", "VALID-A", ledger)
    open_lockbox("a", "VALID-B", ledger)
    with pytest.raises(ValueError):
        open_lockbox("a", "DEV", ledger)


# ---------- scorecard ----------

def _trades(n=600, edge_bps=20.0, cost_bps=5.0, seed=0, symbols=8):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({
        "symbol": [f"S{i % symbols}" for i in range(n)], "side": rng.choice([1, -1], n), "weight": 1.0,
        "gross_bps": edge_bps + rng.normal(0, 60, n), "cost_bps": cost_bps, "funding_bps": 0.5,
    }, index=idx)


def test_score_basic_accounting():
    c = score("x", _trades())
    t = _trades()
    assert c.trades == 600
    assert c.net == pytest.approx(((t.gross_bps - 5 - 0.5) / 1e4).sum())
    assert c.net_bps == pytest.approx(c.gross_bps - 5.5)
    assert c.stress[2.0] == pytest.approx(c.net - 600 * 5 / 1e4)
    assert set(c.by_symbol.index) == {f"S{i}" for i in range(8)}
    assert score("empty", _trades().iloc[:0]).trades == 0


def test_deflated_sharpe_drops_with_more_trials():
    rng = np.random.default_rng(1)
    d = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert deflated_sharpe(d, 1) > deflated_sharpe(d, 1000)


def test_pbo_high_for_noise_low_for_real_edge():
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.normal(0, 1, (1000, 20)))
    assert pbo_cscv(noise) > 0.3
    edge = noise.copy()
    edge[0] += 0.5  # one candidate is genuinely better everywhere
    assert pbo_cscv(edge) < 0.05


def _passing(c: Card) -> Card:
    c.pbo, c.dsr, c.shuffled_t, c.param_stable = 0.05, 0.99, 0.1, True
    return c


def test_verdict_fails_closed_on_missing_checks_and_low_edge():
    c = score("x", _trades())
    ok, fails = verdict(c)
    assert not ok and any("PBO" in f for f in fails)  # not computed -> fail
    ok, fails = verdict(_passing(score("x", _trades())))
    assert ok, fails
    weak = _passing(score("weak", _trades(edge_bps=7.0)))  # gross 7 < 1.5 * 5.5
    assert not verdict(weak)[0]


def test_rank_prefers_simpler_within_one_se():
    simple = _passing(score("rules", _trades(edge_bps=20.0, seed=3), complexity=0))
    fancy = _passing(score("deep", _trades(edge_bps=21.0, seed=3), complexity=3))
    fragile = score("fragile", _trades(edge_bps=60.0, seed=3), complexity=2)  # huge but fails gates
    r = rank([fancy, fragile, simple])
    assert list(r["name"]) == ["rules", "deep", "fragile"]
