"""Paper execution against the real mainnet book and trade stream. Pure logic (no I/O), unit tested.

Taker: walk the visible top-20 book (as it is after the measured latency) level by level. Size beyond the visible
       book fills at the last level plus BEYOND_BOOK_PENALTY_BPS and is flagged.
Maker: a resting limit order at our price. It fills only from aggressive trades on the other side:
       - a trade strictly through our price -> our whole remainder fills (the level was swept)
       - a trade at our price -> first consumes the queue that was ahead of us when we joined, then fills us
       Partial fills are kept. Unfilled remainder at expiry is handed back to the OMS (it then goes taker).
"""

from dataclasses import dataclass, field

BEYOND_BOOK_PENALTY_BPS = 10.0


@dataclass
class Book:
    symbol: str
    bids: list[tuple[float, float]] = field(default_factory=list)  # best first
    asks: list[tuple[float, float]] = field(default_factory=list)
    event_ms: int = 0
    recv_ms: int = 0

    def update(self, bids: list, asks: list, event_ms: int, recv_ms: int) -> None:
        self.bids = [(float(p), float(q)) for p, q in bids if float(q) > 0]
        self.asks = [(float(p), float(q)) for p, q in asks if float(q) > 0]
        self.event_ms, self.recv_ms = event_ms, recv_ms

    @property
    def ok(self) -> bool:
        return bool(self.bids and self.asks and self.bids[0][0] < self.asks[0][0])

    @property
    def mid(self) -> float:
        return (self.bids[0][0] + self.asks[0][0]) / 2

    def queue_at(self, side: int, price: float) -> float:
        """Displayed size at `price` on our side (side +1 buy -> bids)."""
        levels = self.bids if side > 0 else self.asks
        return sum(q for p, q in levels if p == price)


@dataclass
class Fill:
    price: float
    qty: float
    liquidity: str           # maker | taker
    beyond_book: bool = False


def taker_fill(book: Book, side: int, qty: float) -> Fill:
    """Market order of `qty` (base) walking the book. side +1 buys from asks, -1 sells into bids."""
    levels = book.asks if side > 0 else book.bids
    left, cost = qty, 0.0
    for p, q in levels:
        take = min(left, q)
        cost += take * p
        left -= take
        if left <= 1e-12:
            return Fill(cost / qty, qty, "taker")
    last = levels[-1][0] if levels else book.mid
    px = last * (1 + side * BEYOND_BOOK_PENALTY_BPS / 1e4)
    cost += left * px
    return Fill(cost / qty, qty, "taker", beyond_book=True)


@dataclass
class MakerOrder:
    order_id: int
    symbol: str
    side: int                # +1 buy (rests on bid), -1 sell (rests on ask)
    price: float
    qty: float
    queue_ahead: float
    placed_ms: int
    expires_ms: int
    filled: float = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled)

    def on_trade(self, price: float, qty: float, buyer_is_maker: bool) -> Fill | None:
        """Feed one aggTrade. Returns the fill it caused (if any)."""
        if self.remaining <= 0:
            return None
        aggressive_sell = buyer_is_maker  # buyer resting => seller crossed
        if self.side > 0 and not aggressive_sell:
            return None
        if self.side < 0 and aggressive_sell:
            return None
        through = price < self.price if self.side > 0 else price > self.price
        at = price == self.price
        if through:
            take = self.remaining
        elif at:
            hit = qty
            eaten = min(self.queue_ahead, hit)
            self.queue_ahead -= eaten
            take = min(self.remaining, hit - eaten)
            if take <= 0:
                return None
        else:
            return None
        self.filled += take
        return Fill(self.price, take, "maker")
