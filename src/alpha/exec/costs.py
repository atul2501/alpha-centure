"""One cost model for research, backtest, paper and live. All costs are in basis points of traded notional.

taker fill  = fee_taker + half spread + book impact + latency drift
maker fill  = fee_maker - half spread + adverse selection         (fills are not guaranteed: see fill_prob)
holding     = funding at each settlement (long pays a positive rate)

Book impact walks a linear book: with D quote resting within 1% (100 bps) of mid on the side being hit, an order of
N quote moves the price ~ N / D * 100 bps and fills on average halfway there. Defaults are VIP0 USD-M fees and
conservative stand-ins; calibrate() replaces them from recorded paper fills / live book data.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

LATENCY_DRIFT_FACTOR = 0.4  # E|move| over the delay ~ 0.8 sigma; assume half of it goes against us


@dataclass(frozen=True)
class CostModel:
    fee_maker_bps: float = 2.0          # Binance USD-M VIP0
    fee_taker_bps: float = 5.0
    latency_ms: float = 150.0           # signal -> order at exchange; replace with measured ws_latency p95 + REST RTT
    impact_mult: float = 1.0            # scales the linear-book impact (calibrated from paper fills)
    adverse_selection_bps: float | None = None  # maker: None = lose the whole half spread (no spread capture)
    maker_fill_prob: float = 0.6        # share of passive orders that fill within their life (calibrated)
    cost_mult: float = 1.0              # stress test: every cost x cost_mult

    def stressed(self, k: float) -> "CostModel":
        return replace(self, cost_mult=self.cost_mult * k)

    # ---- per fill (scalars or arrays) ----

    def impact_bps(self, notional, depth_1pct):
        depth = np.where(np.asarray(depth_1pct, float) > 0, depth_1pct, np.nan)
        return self.impact_mult * 0.5 * np.asarray(notional, float) / depth * 100.0

    def latency_bps(self, vol_bps_per_sqrt_s):
        return LATENCY_DRIFT_FACTOR * np.asarray(vol_bps_per_sqrt_s, float) * np.sqrt(self.latency_ms / 1000.0)

    def taker_bps(self, notional, spread_bps, depth_1pct, vol_bps_per_sqrt_s=0.0):
        c = (self.fee_taker_bps + 0.5 * np.asarray(spread_bps, float) + self.impact_bps(notional, depth_1pct)
             + self.latency_bps(vol_bps_per_sqrt_s))
        return self.cost_mult * c

    def maker_bps(self, spread_bps):
        half = 0.5 * np.asarray(spread_bps, float)
        adverse = half if self.adverse_selection_bps is None else self.adverse_selection_bps
        return self.cost_mult * (self.fee_maker_bps - half + adverse)

    # ---- holding ----

    def funding_bps(self, side, rates) -> float:
        """side +1 long / -1 short; rates = funding rates of the settlements crossed while holding."""
        return self.cost_mult * float(side) * float(np.sum(rates)) * 10_000.0


def funding_crossed(rates: pd.Series, entry: pd.Timestamp, exit: pd.Timestamp) -> pd.Series:
    """Settlements with entry < funding_time <= exit (a position open at the settlement instant pays)."""
    return rates[(rates.index > entry) & (rates.index <= exit)]


def gross_edge_clears(gross_bps: float, cost_bps: float, margin: float = 1.5) -> bool:
    """The trade gate: expected gross edge must exceed all-in cost by `margin`."""
    return bool(np.isfinite(gross_bps) and np.isfinite(cost_bps) and gross_bps >= margin * cost_bps)
