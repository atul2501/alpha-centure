import pandas as pd

from alpha.coins import SYMBOL_RE, basic_rules, compare, merge_symbols, verdict

INFO = {"contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "COIN", "status": "TRADING",
        "onboardDate": pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000}
TODAY = pd.Timestamp("2026-10-17").date()


def test_basic_rules():
    assert basic_rules(INFO, [30e6] * 30, TODAY)["pass"]
    assert basic_rules(INFO, [100e6] + [5e6] * 29, TODAY)["fails"] == ["30-day median volume >= $20M"]  # one-day pump
    assert not basic_rules(INFO | {"underlyingType": "COMMODITY"}, [30e6] * 30, TODAY)["pass"]
    assert not basic_rules(INFO | {"onboardDate": pd.Timestamp("2026-09-01", tz="UTC").timestamp() * 1000}, [30e6] * 30, TODAY)["pass"]
    assert basic_rules(None, [], TODAY)["fails"] == ["not a Binance USD-M contract"]


def test_verdict_and_compare():
    ok = {"pass": True, "fails": []}
    assert verdict({"pass": False, "fails": ["x"]}, None)[0] == "reject"
    assert verdict(ok, {"days": 30})[0] == "weak"                                  # too new: shadow only
    assert verdict(ok, {"days": 400, "ann_diff": 0.02, "t": 0.4})[0] == "good"
    assert verdict(ok, {"days": 400, "ann_diff": -0.03, "t": -1.0})[0] == "weak"
    assert verdict(ok, {"days": 400, "ann_diff": -0.10, "t": -2.5})[0] == "reject"
    i = pd.date_range("2024-01-01", periods=400, freq="D")
    base = pd.Series(0.001, index=i)
    c = compare(base, base + pd.Series([0.0002, -0.0001] * 200, index=i))
    assert c["days"] == 400 and c["ann_diff"] > 0


def test_symbols():
    assert merge_symbols(["BTCUSDT", "ETHUSDT"], ["INJUSDT", "BTCUSDT"]) == ["BTCUSDT", "ETHUSDT", "INJUSDT"]
    assert SYMBOL_RE.match("1000PEPEUSDT") and not SYMBOL_RE.match("BTC'; DROP") and not SYMBOL_RE.match("BTCUSDC")
