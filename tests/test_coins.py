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


def test_purge_deletes_only_that_coin_and_refuses_in_use():
    import os

    import psycopg
    import pytest

    from alpha.coins import purge
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set")
    with psycopg.connect(dsn) as c:   # one transaction, rolled back at the end
        c.execute("INSERT INTO coin_candidates (symbol, status) VALUES ('ZZTESTUSDT', 'rejected')")
        c.execute("INSERT INTO funding_rate (symbol, funding_time, funding_rate) VALUES ('ZZTESTUSDT', now(), 0.0001)")
        c.execute("INSERT INTO vision_files (path, dataset, symbol, period, rows) VALUES ('x/ZZTESTUSDT/1', 'klines', 'ZZTESTUSDT', '2026-01', 1)")
        before = c.execute("SELECT count(*) FROM funding_rate WHERE symbol <> 'ZZTESTUSDT'").fetchone()[0]
        with pytest.raises(ValueError):
            purge(c, "BTCUSDT", {"BTCUSDT"})
        out = purge(c, "ZZTESTUSDT", {"BTCUSDT"})
        assert out["funding_rate"] == 1 and out["vision_files"] == 1
        assert c.execute("SELECT count(*) FROM funding_rate WHERE symbol <> 'ZZTESTUSDT'").fetchone()[0] == before
        assert c.execute("SELECT purged_at IS NOT NULL FROM coin_candidates WHERE symbol = 'ZZTESTUSDT'").fetchone()[0]
        c.rollback()
