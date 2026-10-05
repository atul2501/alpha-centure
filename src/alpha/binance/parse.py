"""Pure parsing helpers: Binance payloads -> DB row tuples. No I/O, easy to unit test."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 3_600_000,
    "2h": 2 * 3_600_000,
    "4h": 4 * 3_600_000,
    "6h": 6 * 3_600_000,
    "12h": 12 * 3_600_000,
    "1d": 86_400_000,
    "1w": 7 * 86_400_000,
}

CANDLE_COLS = (
    "symbol", "interval", "open_time", "close_time", "open", "high", "low", "close",
    "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "source",
)


def interval_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(f"unsupported interval {interval!r}") from None


def interval_td(interval: str) -> timedelta:
    return timedelta(milliseconds=interval_ms(interval))


def ms_to_dt(ms: int | str) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def now_ms() -> int:
    return dt_to_ms(datetime.now(timezone.utc))


def rest_kline_to_row(symbol: str, interval: str, k: list) -> tuple:
    """REST /api/v3/klines entry:
    [openTime, open, high, low, close, volume, closeTime, quoteVol, trades, takerBase, takerQuote, ignore]"""
    return (
        symbol, interval, ms_to_dt(k[0]), ms_to_dt(k[6]),
        Decimal(k[1]), Decimal(k[2]), Decimal(k[3]), Decimal(k[4]),
        Decimal(k[5]), Decimal(k[7]), int(k[8]), Decimal(k[9]), Decimal(k[10]), "rest",
    )


def ws_kline_to_row(k: dict, symbol: str | None = None) -> tuple:
    """Websocket kline payload `data.k`. `symbol` overrides the stored name (e.g. BTCUSDT.P for perps)."""
    return (
        symbol or k["s"], k["i"], ms_to_dt(k["t"]), ms_to_dt(k["T"]),
        Decimal(k["o"]), Decimal(k["h"]), Decimal(k["l"]), Decimal(k["c"]),
        Decimal(k["v"]), Decimal(k["q"]), int(k["n"]), Decimal(k["V"]), Decimal(k["Q"]), "ws",
    )


def orderbook_row(symbol: str, ts: datetime, bids: list, asks: list) -> tuple | None:
    """Top-of-book stats from a depth20 snapshot. bids/asks are [[price, qty], ...] strings."""
    if not bids or not asks:
        return None
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    bid_qty = sum(float(q) for _, q in bids)
    ask_qty = sum(float(q) for _, q in asks)
    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid * 10_000 if mid else 0.0
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total else 0.0
    return (symbol, ts, best_bid, best_ask, spread_bps, bid_qty, ask_qty, imbalance, bids, asks)


def liquidation_row(o: dict) -> tuple:
    """Futures forceOrder payload `data.o`."""
    return (
        o["s"], ms_to_dt(o["T"]), o["S"], float(o["p"]), float(o["ap"]),
        float(o["q"]), float(o["z"]), o["X"],
    )


class FlowBucket:
    """Accumulates aggTrades for one symbol-minute."""

    __slots__ = ("minute_ms", "buy_vol", "sell_vol", "buy_quote", "sell_quote", "trades", "max_trade_quote", "complete")

    def __init__(self, minute_ms: int, complete: bool):
        self.minute_ms = minute_ms
        self.buy_vol = self.sell_vol = self.buy_quote = self.sell_quote = self.max_trade_quote = 0.0
        self.trades = 0
        self.complete = complete

    def add(self, price: float, qty: float, buyer_is_maker: bool) -> None:
        quote = price * qty
        # buyer is maker => the seller crossed the spread => aggressive sell
        if buyer_is_maker:
            self.sell_vol += qty
            self.sell_quote += quote
        else:
            self.buy_vol += qty
            self.buy_quote += quote
        self.trades += 1
        if quote > self.max_trade_quote:
            self.max_trade_quote = quote

    def row(self, symbol: str) -> tuple:
        return (
            symbol, ms_to_dt(self.minute_ms), self.buy_vol, self.sell_vol, self.buy_vol - self.sell_vol,
            self.buy_quote, self.sell_quote, self.trades, self.max_trade_quote, self.complete,
        )


class FlowAggregator:
    """Turns a stream of aggTrades into closed 1-minute flow rows.

    A minute's bucket is emitted when the first trade of a later minute arrives. The first bucket
    per symbol after (re)connect is marked incomplete because earlier trades in that minute were missed.
    """

    def __init__(self) -> None:
        self.buckets: dict[str, FlowBucket] = {}
        self.seen: set[str] = set()  # symbols that already had a full minute since last reset

    def reset(self) -> list[tuple]:
        """Call on (re)connect. Returns open buckets as incomplete rows."""
        out = []
        for sym, b in self.buckets.items():
            b.complete = False
            out.append(b.row(sym))
        self.buckets.clear()
        self.seen.clear()
        return out

    def add_trade(self, symbol: str, trade_time_ms: int, price: float, qty: float, buyer_is_maker: bool) -> tuple | None:
        minute = trade_time_ms - trade_time_ms % 60_000
        closed = None
        b = self.buckets.get(symbol)
        if b is not None and minute > b.minute_ms:
            closed = b.row(symbol)
            b = None
        if b is None:
            b = self.buckets[symbol] = FlowBucket(minute, complete=symbol in self.seen)
            self.seen.add(symbol)
        if minute == b.minute_ms:  # late trades from an already-emitted minute are dropped
            b.add(price, qty, buyer_is_maker)
        return closed


def book_tick_row(symbol: str, ts: datetime, bids: list, asks: list) -> tuple | None:
    """Top of book + visible depth from a depth20 payload (perp: data['b'] / data['a'])."""
    if not bids or not asks:
        return None
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    mid = (best_bid + best_ask) / 2
    if mid <= 0:
        return None
    bid_quote = sum(float(p) * float(q) for p, q in bids)
    ask_quote = sum(float(p) * float(q) for p, q in asks)
    bid_reach = (mid - float(bids[-1][0])) / mid * 10_000
    ask_reach = (float(asks[-1][0]) - mid) / mid * 10_000
    return (symbol, ts, best_bid, best_ask, float(bids[0][1]), float(asks[0][1]), bid_quote, ask_quote,
            bid_reach, ask_reach)


class LatencyStats:
    """Per-minute delivery latency (receive - event time, ms) per stream kind."""

    def __init__(self) -> None:
        self.samples: dict[tuple[int, str], list[float]] = {}

    def add(self, stream: str, event_ms: int, recv_ms: int) -> list[tuple]:
        """Record one sample; returns rows for minutes that are now complete."""
        minute = recv_ms - recv_ms % 60_000
        self.samples.setdefault((minute, stream), []).append(recv_ms - event_ms)
        done = [k for k in self.samples if k[0] < minute]
        return [self._row(k) for k in done]

    def flush(self) -> list[tuple]:
        return [self._row(k) for k in list(self.samples)]

    def _row(self, key: tuple[int, str]) -> tuple:
        xs = sorted(self.samples.pop(key))
        q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
        return (ms_to_dt(key[0]), key[1], len(xs), float(q(0.5)), float(q(0.95)), float(xs[-1]))
