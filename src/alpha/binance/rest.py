"""Binance spot + USD-M futures REST client (public market data, no API key)."""

import asyncio

import httpx
from loguru import logger

from alpha.binance.parse import interval_ms, now_ms

FUTURES_STATS_PERIOD_MS = 5 * 60_000
FUTURES_STATS_MAX_AGE_MS = 29 * 86_400_000  # Binance serves only the last ~30 days


class BinanceREST:
    def __init__(self, spot_base: str, futures_base: str):
        self.spot_base = spot_base
        self.futures_base = futures_base
        self.client = httpx.AsyncClient(timeout=20)

    async def close(self) -> None:
        await self.client.aclose()

    async def _get(self, url: str, params: dict, retries: int = 5):
        for attempt in range(retries):
            try:
                r = await self.client.get(url, params=params)
            except httpx.HTTPError as e:
                wait = 2**attempt
                logger.warning("GET {} failed ({}), retry in {}s", url, e, wait)
                await asyncio.sleep(wait)
                continue
            if r.status_code in (403, 418, 429):  # 403 = Binance WAF limit (e.g. fundingRate's 500 req / 5 min per IP)
                wait = int(r.headers.get("Retry-After", 60))
                logger.warning("rate limited ({}) on {}, sleeping {}s", r.status_code, url, wait)
                await asyncio.sleep(wait)
                continue
            if r.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            r.raise_for_status()
            # Weight limits: spot 6000/min, futures 2400/min. Back off well before either.
            used = int(r.headers.get("x-mbx-used-weight-1m", 0) or 0)
            if used > (1800 if url.startswith(self.futures_base) else 4500):
                logger.info("weight {} used this minute, pausing 15s", used)
                await asyncio.sleep(15)
            return r.json()
        raise RuntimeError(f"GET {url} {params} failed after {retries} attempts")

    # ---------- klines (spot + perp share the same format) ----------

    async def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int | None = None,
                     limit: int = 1000, market: str = "spot") -> list[list]:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        url = f"{self.futures_base}/fapi/v1/klines" if market == "perp" else f"{self.spot_base}/api/v3/klines"
        return await self._get(url, params)

    async def klines_range(self, symbol: str, interval: str, start_ms: int, end_ms: int | None = None,
                           market: str = "spot"):
        """Yields batches of *closed* klines from start_ms up to end_ms (or now)."""
        step = interval_ms(interval)
        cursor = start_ms
        while True:
            batch = await self.klines(symbol, interval, cursor, end_ms, market=market)
            if not batch:
                return
            now = now_ms()
            closed = [k for k in batch if k[6] < now]
            if closed:
                yield closed
            last_open = batch[-1][0]
            if len(batch) < 1000 or len(closed) < len(batch) or (end_ms is not None and last_open + step > end_ms):
                return
            cursor = last_open + step

    # ---------- futures ----------

    async def funding_rate(self, symbol: str, start_ms: int, limit: int = 1000) -> list[dict]:
        return await self._get(
            f"{self.futures_base}/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": max(start_ms, 1), "limit": limit},
        )

    async def premium_index(self, symbol: str) -> dict:
        return await self._get(f"{self.futures_base}/fapi/v1/premiumIndex", {"symbol": symbol})

    async def premium_klines(self, symbol: str, interval: str, start_ms: int, limit: int = 500) -> list[list]:
        """Premium index klines (same row format as klines)."""
        return await self._get(f"{self.futures_base}/fapi/v1/premiumIndexKlines",
                               {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit})

    async def open_interest(self, symbol: str) -> dict:
        """Current open interest: {"symbol", "openInterest", "time"}."""
        return await self._get(f"{self.futures_base}/fapi/v1/openInterest", {"symbol": symbol})

    async def futures_stats_range(self, path: str, symbol: str, start_ms: int, limit: int = 500):
        """Yields batches from /futures/data/{path} (5m period) from start_ms to now.

        With both startTime and endTime set Binance returns the newest `limit` rows of the window,
        so we walk fixed windows of exactly `limit` periods."""
        now = now_ms()
        start = max(start_ms, now - FUTURES_STATS_MAX_AGE_MS)
        window = FUTURES_STATS_PERIOD_MS * limit
        while start < now:
            end = min(start + window - 1, now)
            batch = await self._get(
                f"{self.futures_base}/futures/data/{path}",
                {"symbol": symbol, "period": "5m", "startTime": start, "endTime": end, "limit": limit},
            )
            if batch:
                yield batch
            start = end + 1
