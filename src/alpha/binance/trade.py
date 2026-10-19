"""Binance USD-M signed client: the ONLY module allowed to place orders (tests/test_no_real_orders.py enforces it).

Used by alpha.live.engine when TRADING_MODE=live. No retries on order placement: every order carries our own
clientOrderId (alpha<paper_orders.id>), so after any error the engine asks the exchange what really happened.
"""

import hashlib
import hmac
import time
from urllib.parse import urlencode

import httpx

LEVERAGE = 5  # per coin; the strategy caps a coin at 0.5x and the book at 3x gross, cross margin covers the rest
UNKNOWN_ORDER = (-2011, -2013)


class BinanceError(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(f"Binance {code}: {msg}")
        self.code = code


def num(x: float) -> str:
    """Plain decimal for order params: 49.730000000000004 -> '49.73', 1e-05 -> '0.00001'."""
    return f"{x:.10f}".rstrip("0").rstrip(".")


def sign(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def cid(order_id: int) -> str:
    return f"alpha{order_id}"


class BinanceTrader:
    def __init__(self, base: str, key: str, secret: str):
        if not key or not secret:
            raise RuntimeError("TRADING_MODE=live needs BINANCE_API_KEY and BINANCE_API_SECRET")
        self.base, self.secret = base, secret
        self.client = httpx.AsyncClient(timeout=10, headers={"X-MBX-APIKEY": key})

    async def _call(self, method: str, path: str, **params):
        params = {k: v for k, v in params.items() if v is not None}
        params.update(timestamp=int(time.time() * 1000), recvWindow=5000)
        q = urlencode(params)
        r = await self.client.request(method, f"{self.base}{path}?{q}&signature={sign(self.secret, q)}")
        d = r.json()
        if r.status_code >= 400 or (isinstance(d, dict) and d.get("code", 0) < 0):
            raise BinanceError(d.get("code", r.status_code), d.get("msg", r.text))
        return d

    async def setup(self, symbols: list[str]) -> None:
        """One-way mode, cross margin, LEVERAGE on every coin. Refuses a hedge-mode account."""
        if (await self._call("GET", "/fapi/v1/positionSide/dual"))["dualSidePosition"]:
            raise RuntimeError("account is in Hedge mode: switch Binance futures to One-way mode")
        for s in symbols:
            try:
                await self._call("POST", "/fapi/v1/marginType", symbol=s, marginType="CROSSED")
            except BinanceError as e:
                if e.code != -4046:  # "No need to change margin type"
                    raise
            await self._call("POST", "/fapi/v1/leverage", symbol=s, leverage=LEVERAGE)

    async def account(self) -> dict:
        return await self._call("GET", "/fapi/v2/account")

    async def order(self, symbol: str, side: int, qty: float, order_id: int, price: float | None = None) -> dict:
        """price set: post-only limit (GTX, expires instead of taking). price None: market."""
        kind = {"type": "LIMIT", "timeInForce": "GTX", "price": num(price)} if price else {"type": "MARKET"}
        return await self._call("POST", "/fapi/v1/order", symbol=symbol, side="BUY" if side > 0 else "SELL",
                                quantity=num(qty), newClientOrderId=cid(order_id), **kind)

    async def finish(self, symbol: str, order_id: int) -> dict | None:
        """Cancel if still open, return the final order (None: the exchange never got it)."""
        try:
            return await self._call("DELETE", "/fapi/v1/order", symbol=symbol, origClientOrderId=cid(order_id))
        except BinanceError as e:
            if e.code not in UNKNOWN_ORDER:
                raise
        try:  # already filled / expired: nothing to cancel
            return await self._call("GET", "/fapi/v1/order", symbol=symbol, origClientOrderId=cid(order_id))
        except BinanceError as e:
            if e.code in UNKNOWN_ORDER:
                return None
            raise

    async def trades(self, symbol: str, exchange_order_id: int) -> list[dict]:
        return await self._call("GET", "/fapi/v1/userTrades", symbol=symbol, orderId=exchange_order_id)

    async def funding(self, since_ms: int) -> list[dict]:
        return await self._call("GET", "/fapi/v1/income", incomeType="FUNDING_FEE", startTime=since_ms, limit=1000)
