"""Order safety: only alpha.binance.trade may reach authenticated / order endpoints, signing matches Binance, and
live mode refuses to start without keys."""

from pathlib import Path

import pytest

from alpha.binance.trade import num, sign
from alpha.config import Settings

FORBIDDEN = ["/fapi/v1/order", "/fapi/v1/batchOrders", "/fapi/v1/leverage", "/fapi/v1/marginType",
             "X-MBX-APIKEY", "signature=", "listenKey", "/fapi/v2/account"]
ORDER_MODULE = "alpha/binance/trade.py"


def test_order_endpoints_only_in_trade_module():
    root = Path(__file__).resolve().parents[1] / "src"
    hits = [(str(p), f) for p in root.rglob("*.py") if not str(p).endswith(ORDER_MODULE)
            for f in FORBIDDEN if f in p.read_text()]
    assert hits == []


def test_signature_matches_binance_docs_example():
    q = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    assert sign(secret, q) == "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"


def test_order_numbers_are_plain_decimals():
    assert num(0.1 * 497.3) == "49.73" and num(1e-05) == "0.00001" and num(3.0) == "3"


def test_live_needs_keys_and_mode_is_checked():
    from alpha.live.engine import PaperEngine

    with pytest.raises(RuntimeError, match="BINANCE_API_KEY"):
        PaperEngine(Settings(trading_mode="live", binance_api_key="", binance_api_secret=""), db=None)
    with pytest.raises(ValueError):
        Settings(trading_mode="real").ledger_schema
    assert Settings(trading_mode="live").ledger_conn == {"options": "-c search_path=live,public"}
    assert Settings(trading_mode="paper").ledger_conn == {}
