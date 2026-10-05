"""Paper perp account (USDT-margined, cross). Pure accounting, rebuilt from the ledger on restart.

cash      = starting equity + realized P&L - fees - funding paid
equity    = cash + sum(qty * (mark - entry))
Realized P&L books on any reduction / flip at the average entry price.
"""

from dataclasses import dataclass, field

FEE_BPS = {"maker": 2.0, "taker": 5.0}  # Binance USD-M VIP0


@dataclass
class Position:
    qty: float = 0.0          # signed base quantity
    entry: float = 0.0        # average entry price


@dataclass
class Account:
    start_equity: float
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    fees: float = 0.0
    funding: float = 0.0       # paid (>0) / received (<0)
    realized: float = 0.0

    def __post_init__(self) -> None:
        if not self.cash:
            self.cash = self.start_equity

    def pos(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position())

    def apply_fill(self, symbol: str, side: int, qty: float, price: float, liquidity: str) -> float:
        """Book a fill; returns the fee charged."""
        p = self.pos(symbol)
        signed = side * qty
        fee = abs(qty * price) * FEE_BPS[liquidity] / 1e4
        if p.qty == 0 or (p.qty > 0) == (signed > 0):          # open / add
            new = p.qty + signed
            p.entry = (p.entry * abs(p.qty) + price * abs(signed)) / abs(new)
            p.qty = new
        else:                                                   # reduce / close / flip
            closing = min(abs(signed), abs(p.qty))
            pnl = closing * (price - p.entry) * (1 if p.qty > 0 else -1)
            self.realized += pnl
            self.cash += pnl
            rest = abs(signed) - closing
            p.qty += signed
            if abs(p.qty) < 1e-12:
                p.qty, p.entry = 0.0, 0.0
            elif rest > 0:                                       # flipped: remainder opens at fill price
                p.entry = price
        self.cash -= fee
        self.fees += fee
        return fee

    def apply_funding(self, symbol: str, mark: float, rate: float) -> float:
        """Settlement: longs pay positive rates. Returns amount paid (>0) or received (<0)."""
        q = self.pos(symbol).qty
        amount = q * mark * rate
        self.cash -= amount
        self.funding += amount
        return amount

    def unrealized(self, marks: dict[str, float]) -> float:
        return sum(p.qty * (marks[s] - p.entry) for s, p in self.positions.items() if p.qty and s in marks)

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.unrealized(marks)

    def notional(self, marks: dict[str, float]) -> dict[str, float]:
        return {s: p.qty * marks[s] for s, p in self.positions.items() if p.qty and s in marks}

    def gross_leverage(self, marks: dict[str, float]) -> float:
        eq = self.equity(marks)
        return sum(abs(v) for v in self.notional(marks).values()) / eq if eq > 0 else float("inf")
