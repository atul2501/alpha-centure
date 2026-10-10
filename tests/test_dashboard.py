"""Dashboard: P&L arithmetic (waterfall, per-coin), the public-bind guard, basic auth, parameter whitelist, and a
DB-backed smoke test of every endpoint."""

import base64

import psycopg
import pytest
from starlette.testclient import TestClient

from alpha.config import get_settings
from alpha.exec.account import Account
from dashboard import data
from dashboard.server import check_bind, create_app


def _sample():
    fills = [dict(symbol="BTCUSDT", side=1, qty=0.01, price=60000, liquidity="maker"),
             dict(symbol="ETHUSDT", side=-1, qty=0.5, price=3000, liquidity="taker"),
             dict(symbol="BTCUSDT", side=-1, qty=0.004, price=61000, liquidity="maker"),
             dict(symbol="ETHUSDT", side=1, qty=0.2, price=2900, liquidity="maker")]
    acct = Account(1000.0)
    for f in fills:
        acct.apply_fill(f["symbol"], f["side"], f["qty"], f["price"], f["liquidity"])
    funding = {"BTCUSDT": acct.apply_funding("BTCUSDT", 61500, 0.0001), "ETHUSDT": acct.apply_funding("ETHUSDT", 2950, 0.0002)}
    marks = {"BTCUSDT": 61500.0, "ETHUSDT": 2950.0}
    positions = {s: {"qty": p.qty, "entry": p.entry, "mark": marks[s]} for s, p in acct.positions.items()}
    equity = acct.cash + sum(p["qty"] * (p["mark"] - p["entry"]) for p in positions.values())
    return fills, funding, positions, acct, equity


def test_waterfall_adds_up():
    _, _, _, acct, equity = _sample()
    w = data.waterfall(1000.0, equity, acct.fees, acct.funding)
    assert w["start"] + w["trading"] - w["fees"] - w["funding"] == pytest.approx(w["equity"])
    assert w["total"] == pytest.approx(equity - 1000.0)
    assert acct.fees > 0 and acct.funding != 0


def test_pnl_by_coin_sums_to_total():
    fills, funding, positions, acct, equity = _sample()
    rows = data.pnl_by_coin(fills, funding, positions, 1000.0)
    assert {r["symbol"] for r in rows} == {"BTCUSDT", "ETHUSDT"}
    assert sum(r["net"] for r in rows) == pytest.approx(equity - 1000.0)
    assert sum(r["fees"] for r in rows) == pytest.approx(acct.fees)
    assert sum(r["realized"] for r in rows) == pytest.approx(acct.realized)


def test_public_bind_without_password_is_allowed():
    check_bind("0.0.0.0", None)
    check_bind("0.0.0.0", "secret")
    check_bind("127.0.0.1", None)


def test_no_password_means_no_login():
    assert TestClient(create_app(None)).get("/").status_code == 200


def test_basic_auth_and_parameter_whitelist():
    client = TestClient(create_app("secret"))
    assert client.get("/api/candles").status_code == 401
    bad = {"Authorization": "Basic " + base64.b64encode(b"me:wrong").decode()}
    assert client.get("/api/candles", headers=bad).status_code == 401
    ok = {"Authorization": "Basic " + base64.b64encode(b"me:secret").decode()}
    r = client.get("/api/candles?symbol=BTCUSDT.P;DROP&interval=1h&bars=200", headers=ok)
    assert r.status_code == 400
    assert client.get("/api/candles?symbol=BTCUSDT.P&interval=7m&bars=200", headers=ok).status_code == 400
    assert client.get("/api/candles?symbol=BTCUSDT.P&interval=1h&bars=9999", headers=ok).status_code == 400
    assert client.get("/", headers=ok).status_code == 200


def test_endpoints_against_the_database():
    try:
        psycopg.connect(get_settings().database_url, connect_timeout=3).close()
    except Exception:
        pytest.skip("dev database not reachable")
    client = TestClient(create_app(None))
    wf = client.get("/api/workflow").json()
    assert len(wf["stages"]) == 7 and "banner" in wf and "activity" in wf
    lg = client.get("/api/ledger").json()
    assert lg["empty"] or {"kpi", "waterfall", "positions", "by_coin", "decisions", "fills", "funding"} <= set(lg)
    sym = next(f.db_symbol for f in get_settings().candle_feeds())
    c = client.get(f"/api/candles?symbol={sym}&interval=1h&bars=100").json()
    assert c["symbol"] == sym and len(c["bars"]) > 0
