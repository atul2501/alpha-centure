"""Loss diagnosis: hit rate / payoff arithmetic, 72h holding, per-period P&L and the strategy-part split."""

import numpy as np
import pandas as pd
import pytest

from alpha.live import diagnose as dg
from alpha.research.panel import wide
from tests.test_book import SYMS, book  # noqa: F401  (module-scoped synthetic panel fixture)


def test_direction_stats_payoff_beats_hit_rate():
    s = dg.direction_stats(pd.Series([3.0, 3.0, -1.0, -1.0, -1.0, 0.0]))  # 40% right, still profitable
    assert s["n"] == 5 and s["hit"] == pytest.approx(0.4)
    assert s["payoff"] == pytest.approx(3.0) and s["net"] == pytest.approx(3.0)
    assert s["breakeven_hit"] == pytest.approx(0.25)


def test_held_on_schedule_only_changes_on_rebalance_hours():
    t = pd.date_range("2026-01-01", periods=300, freq="h", tz="UTC")
    w = pd.DataFrame({"A": np.arange(300.0)}, index=t)
    h = dg.held_on_schedule(w)
    hours = (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
    changes = t[h["A"].diff().fillna(0) != 0]
    assert all(((c - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)) % 72 == 0 for c in changes)
    assert h["A"].iloc[-1] == w["A"][np.asarray(hours % 72 == 0)].iloc[-1]


def test_period_pnl_sign_and_funding():
    t = pd.date_range("2026-01-01", periods=144, freq="h", tz="UTC")
    held = pd.DataFrame({"A": 1.0, "B": -1.0}, index=t)
    ret = pd.DataFrame({"A": 0.001, "B": 0.001}, index=t)
    fund = pd.DataFrame({"A": 0.0, "B": 0.0}, index=t)
    fund.iloc[10] = 0.0001  # longs pay, shorts receive
    pp = dg.period_pnl(held, ret, fund)
    a, b = pp[pp.symbol == "A"], pp[pp.symbol == "B"]
    assert (a["price"] > 0).all() and (b["price"] < 0).all()
    assert a["funding"].sum() == pytest.approx(-0.0001) and b["funding"].sum() == pytest.approx(0.0001)
    assert set(pp["side"]) == {"long", "short"}


def test_sleeve_contributions_add_up_to_book(book):  # noqa: F811
    panel, sc, w = book
    parts = dg.sleeves(panel, sc)
    total = sum(parts[f"{k} (in book)"] for k in dg.MIX)
    uncapped = (w.abs() < 0.5 - 1e-9).all(axis=1)  # the per-coin clip is the only non-linear step
    np.testing.assert_allclose(total[uncapped].abs().sum(axis=1), w[uncapped].abs().sum(axis=1), rtol=1e-9)
    np.testing.assert_allclose(total[uncapped].to_numpy(), w[uncapped].to_numpy(), atol=1e-9)


def test_forecast_quality_runs_and_sees_a_perfect_forecast(book):  # noqa: F811
    panel, sc, _ = book
    start = wide(panel, "ret").index[24 * 40]
    fq, per_coin = dg.forecast_quality(panel, sc, start)
    assert fq["rebalances"] > 10 and abs(fq["rank_ic_mean"]) < 0.3  # random scores: no skill
    cum = wide(panel, "ret").fillna(0).cumsum()
    perfect = (cum.shift(-72) - cum)
    fq2, _ = dg.forecast_quality(panel, perfect, start)
    assert fq2["sign_right"] > 0.99 and fq2["rank_ic_mean"] > 0.99
    assert "XRPUSDT" not in per_coin.dropna(subset=["sign_right"]).index  # ineligible coin is skipped


def test_sections_run_end_to_end_on_fake_data(book, monkeypatch, capsys):  # noqa: F811
    panel, sc, _ = book
    t = pd.date_range("2026-03-01", periods=600, freq="min", tz="UTC")
    marks = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 1e-4, len(t))))
    eq = pd.DataFrame({"ts": t, "equity": 1000.0, "fees": 0.0, "funding": 0.0,
                       "positions": [{"BTCUSDT": {"qty": 1.0, "entry": 100.0, "mark": m},
                                      "ETHUSDT": {"qty": -2.0, "entry": 100.0, "mark": m}} for m in marks]})
    fills = pd.DataFrame({"symbol": ["BTCUSDT"], "side": [1], "price": [100.0], "qty": [1.0], "fee": [0.02],
                          "liquidity": ["maker"], "slippage_bps": [1.0], "arrival_mid": [100.0]})
    fund = pd.DataFrame({"ts": [t[100]], "symbol": ["BTCUSDT"], "amount": [0.01]})
    monkeypatch.setattr(dg, "q", lambda conn, sql, params=None: eq if "paper_equity" in sql else
                        fills if "paper_fills" in sql else fund)
    out = dg.paper_section(None, t[0])
    assert out["price"] == pytest.approx(-(marks[-1] - marks[0]), rel=1e-9)  # +1 and -2 of the same mark
    assert out["by_side"]["long"]["net"] == pytest.approx(marks[-1] - marks[0])

    monkeypatch.setattr(dg, "build_panel", lambda *a, **k: panel)
    monkeypatch.setattr(dg, "p6_scores", lambda X, end, d: sc)
    monkeypatch.setattr(dg, "cached_panel", lambda *a, **k: panel)
    monkeypatch.setattr(dg, "symbol_costs", lambda conn, p: pd.DataFrame({"maker_bps": 2.0, "taker_bps": 5.0},
                                                                          index=SYMS))
    idx = wide(panel, "ret").index
    res = dg.strategy_section(None, SYMS, idx[24 * 60], idx[-1], "models")
    assert {"ridge", "rules", "carry", "book"} <= set(res["parts"])
    assert res["parts"]["book"]["cost"] > 0
    assert "Ridge forecast" in capsys.readouterr().out
