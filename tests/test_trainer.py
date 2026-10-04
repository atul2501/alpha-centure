import numpy as np
import pandas as pd
import pytest

from alpha.models.bundle import drift_reference, psi
from alpha.trainer import should_promote


def m(trades, avg_r):
    return {"trades": trades, "avg_r": avg_r}


def test_promotion_rules():
    assert should_promote(m(0, 0.0), None)[0]                      # first model always goes live
    assert not should_promote(m(3, 0.9), m(20, 0.1))[0]            # too few holdout trades
    assert not should_promote(m(30, -0.05), m(0, 0.0))[0]          # must be positive out-of-sample
    assert not should_promote(m(30, 0.10), m(25, 0.20))[0]         # must beat champion's live record
    assert should_promote(m(30, 0.25), m(25, 0.20))[0]
    assert should_promote(m(30, 0.05), m(2, 0.9))[0]               # champion live record too thin -> compare to 0


def test_psi_detects_shift_but_not_noise():
    rng = np.random.default_rng(0)
    ref = drift_reference(pd.DataFrame({"atr_pct": rng.normal(0, 1, 5000)}))["atr_pct"]
    assert psi(ref, rng.normal(0, 1, 2000)) < 0.05
    assert psi(ref, rng.normal(1.5, 1, 2000)) > 0.25


def test_fit_bundle_end_to_end_on_synthetic_setups():
    """Exercises fit_bundle + evaluate on a tiny synthetic dataset (catches wiring/import errors)."""
    from alpha.features.build import CORE_COLS
    from alpha.models.meta import GEOMETRY_COLS
    from alpha.regime.hmm import regime_features
    from alpha.trainer import evaluate, fit_bundle
    from tests.test_regime import regime_path

    rng = np.random.default_rng(0)
    frames = {"BTCUSDT": regime_features(regime_path(0))}
    times = frames["BTCUSDT"]["close_time"].iloc[100::2]
    n = len(times)
    ds = pd.DataFrame({c: rng.normal(size=n) for c in CORE_COLS + GEOMETRY_COLS},
                      index=pd.DatetimeIndex(times.to_numpy(), name="signal_time"))
    ds["close_time"] = times.to_numpy()
    ds["exit_time"] = ds["close_time"] + pd.Timedelta(hours=8)
    ds["strategy"] = rng.choice(["breakout", "mean_reversion"], n)
    ds["tf"], ds["symbol"] = "1h", "BTCUSDT.P"
    ds["side"] = rng.choice([1, -1], n)
    ds["r"] = rng.normal(0.05, 1.0, n)
    ds["win"] = ds["r"] > 0
    ds["risk"] = 0.01
    ds["gross"] = ds["r"] * ds["risk"] + 0.0012
    ds["close_px"], ds["stop"] = 100.0, 99.0
    cut = ds.index[int(n * 0.8)]
    b = fit_bundle(ds, frames, cut)
    assert b.metrics["train_setups"] > 0 and "drift_ref" in b.metrics
    m = evaluate(b, ds, frames, cut, ds.index[-1] + pd.Timedelta(hours=1))
    assert set(m) >= {"trades", "avg_r", "max_dd"}


def test_json_safe_removes_nan():
    from alpha.models.bundle import json_safe
    assert json_safe({"a": float("nan"), "b": [np.float64(1.5), np.inf], "c": np.int64(3)}) == {"a": None, "b": [1.5, None], "c": 3}
