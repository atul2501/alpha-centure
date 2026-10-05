"""Paper-only guarantee: no code path can reach an authenticated / order endpoint, and the engine refuses any
mode other than paper."""

from pathlib import Path

import pytest

from alpha.config import Settings

FORBIDDEN = ["/fapi/v1/order", "/fapi/v1/batchOrders", "/fapi/v1/leverage", "/fapi/v1/marginType",
             "X-MBX-APIKEY", "signature=", "api_secret", "listenKey", "/fapi/v2/account"]


def test_source_has_no_order_or_auth_endpoints():
    root = Path(__file__).resolve().parents[1] / "src"
    hits = [(str(p), f) for p in root.rglob("*.py") for f in FORBIDDEN if f in p.read_text()]
    assert hits == []


def test_engine_refuses_non_paper_mode():
    from alpha.live.engine import PaperEngine

    with pytest.raises(RuntimeError):
        PaperEngine(Settings(trading_mode="live"), db=None)
