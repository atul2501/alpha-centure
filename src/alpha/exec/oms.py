"""Order planning + risk. Pure functions; the engine does the I/O.

plan_orders: target weights -> per-coin orders against ACTUAL positions, with the strategy's no-trade band
             (skip a coin when |target - current| < band, as in the backtest), rounded to Binance lot steps and
             dropped below the exchange minimum notional.
Risk:        hard caps (gross <= 3x, coin <= 0.5x), kill switches (drawdown from peak -> flatten + halt; daily loss
             -> no new risk today), and a conservative maintenance-margin ratio check.
"""

import math
from dataclasses import dataclass

MAX_GROSS = 3.0
MAX_COIN = 0.5
KILL_DRAWDOWN = 0.30  # catastrophe stop (manual reset). V4_carry's drawdowns reached 14% (DEV) and 18% (VALID-A) in the 2021-26 replay:
                      # a 10% stop would have halted it for good in May 2021. Normal losses are the monitor's job.
DAILY_LOSS_STOP = 0.03
MAINT_MARGIN_RATE = 0.015   # conservative across brackets for these sizes (Binance first brackets are 0.4-1%)
MARGIN_RATIO_ALERT = 0.5


@dataclass(frozen=True)
class LotFilter:
    step: float          # LOT_SIZE / MARKET_LOT_SIZE stepSize
    min_qty: float
    min_notional: float  # MIN_NOTIONAL notional


@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    side: int
    qty: float
    target_w: float
    current_w: float


def round_step(qty: float, step: float) -> float:
    """Floor to the lot step, without float noise (0.1 * 497.3 -> 49.73, not 49.730000000000004)."""
    if step <= 0:
        return qty
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(math.floor(qty / step + 1e-9) * step, decimals)


def cap_targets(target: dict[str, float]) -> dict[str, float]:
    t = {s: max(-MAX_COIN, min(MAX_COIN, w)) for s, w in target.items()}
    gross = sum(abs(w) for w in t.values())
    if gross > MAX_GROSS:
        t = {s: w * MAX_GROSS / gross for s, w in t.items()}
    return t


def plan_orders(target: dict[str, float], positions: dict[str, float], marks: dict[str, float], equity: float,
                filters: dict[str, LotFilter], band: float, reduce_only: bool = False) -> list[PlannedOrder]:
    """positions: signed base qty per coin. reduce_only: daily-loss stop active -> only orders that shrink |position|."""
    out = []
    target = cap_targets(target)
    for s in sorted(set(target) | {k for k, q in positions.items() if q}):
        if s not in marks or equity <= 0:
            continue
        cur_q = positions.get(s, 0.0)
        cur_w = cur_q * marks[s] / equity
        tw = target.get(s, 0.0)
        if abs(tw - cur_w) < band:
            continue
        if reduce_only and abs(tw) > abs(cur_w):
            tw = cur_w if (tw > 0) == (cur_w > 0) else 0.0
            if abs(tw - cur_w) < 1e-12:
                continue
        delta_q = tw * equity / marks[s] - cur_q
        f = filters.get(s, LotFilter(0.0, 0.0, 5.0))
        q = round_step(abs(delta_q), f.step)
        if q < f.min_qty or q * marks[s] < f.min_notional:
            continue
        out.append(PlannedOrder(s, 1 if delta_q > 0 else -1, q, tw, cur_w))
    return out


@dataclass
class RiskState:
    peak_equity: float
    day: str = ""
    day_start_equity: float = 0.0
    halted: bool = False
    reason: str = ""

    def update(self, equity: float, day: str) -> None:
        if day != self.day:
            self.day, self.day_start_equity = day, equity
        self.peak_equity = max(self.peak_equity, equity)
        if not self.halted and equity <= self.peak_equity * (1 - KILL_DRAWDOWN):
            self.halted, self.reason = True, f"drawdown {1 - equity / self.peak_equity:.1%} from peak"

    @property
    def daily_stop(self) -> bool:
        return self.day_start_equity > 0 and self.day_loss >= DAILY_LOSS_STOP

    @property
    def day_loss(self) -> float:
        return 0.0 if not self.day_start_equity else max(0.0, 1 - self._eq / self.day_start_equity)

    _eq: float = 0.0

    def observe(self, equity: float, day: str) -> None:
        self._eq = equity
        self.update(equity, day)


def margin_ratio(notional: dict[str, float], equity: float) -> float:
    return sum(abs(v) for v in notional.values()) * MAINT_MARGIN_RATE / equity if equity > 0 else float("inf")
