"""Per-bar execution cost per side (bps), from the one shared cost model (alpha.exec.costs.CostModel).

    taker side = fee_taker + half spread + linear-book impact(notional, depth within 1%) + latency drift(vol)
    maker side = fee_maker - half spread + adverse selection; with fill probability p the expected side cost is
                 p * maker + (1 - p) * taker (an unfilled passive order is chased with a market order)

Spread: 2 x the live median SOL spread (history was wider than today). Depth: the thinner side of book_depth_5m
within 1% at t, capped at the DEV 5th percentile (conservative); before 2023 (no book history) the 5th percentile.
Fees and slippage are kept apart so stress tests can scale them separately.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from alpha.exec.costs import CostModel

LIVE_SPREAD_BPS = 0.83  # median SOLUSDT perp spread from book_tick (81k samples, 2026-10)


@dataclass(frozen=True)
class CostConfig:
    name: str = "taker"
    execution: str = "taker"            # taker | mixed_maker
    notional: float = 30_000.0          # $30k paper equity at 1x (leverage cap 3x is never reached)
    spread_bps: float = 2 * LIVE_SPREAD_BPS
    depth_floor: float | None = None    # quote depth within 1% (one side); None = DEV 5th percentile of the data
    fee_mult: float = 1.0               # cost stress (fees)
    slip_mult: float = 1.0              # slippage stress (spread + impact + latency)
    funding: bool = True
    model: CostModel = field(default_factory=CostModel)

    def with_(self, **kw) -> "CostConfig":
        d = {**{k: getattr(self, k) for k in self.__dataclass_fields__}, **kw}
        return CostConfig(**d)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SideCosts:
    fee: np.ndarray    # bps per side, per decision bar
    slip: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.fee + self.slip


def side_costs(raw: pd.DataFrame, cfg: CostConfig) -> SideCosts:
    cm = cfg.model
    side_depth = np.fmin(raw["bid_1"].to_numpy(float), raw["ask_1"].to_numpy(float))
    ok = np.isfinite(side_depth).any()
    floor = cfg.depth_floor if cfg.depth_floor is not None else float(np.nanquantile(side_depth, 0.05)) \
        if ok else 1.5e6
    # bad snapshots (a few rows near $0 depth) would blow up impact: bound below by the 1st percentile
    low = min(floor, float(np.nanquantile(side_depth, 0.01))) if ok else floor
    depth = np.where(np.isfinite(side_depth), np.clip(side_depth, low, floor), floor)
    close = raw["sol_close"].astype(float)
    vol_bar = (1e4 * np.log(close / close.shift(1))).rolling(96, min_periods=24).std().bfill().fillna(50.0)
    vol_s = (vol_bar / np.sqrt(900.0)).to_numpy()  # bps per sqrt(second)
    half = 0.5 * cfg.spread_bps
    impact = cm.impact_bps(cfg.notional, depth)
    latency = cm.latency_bps(vol_s)
    taker_fee = np.full(len(raw), cm.fee_taker_bps)
    taker_slip = half + impact + latency
    if cfg.execution == "mixed_maker":
        p = cm.maker_fill_prob
        adverse = half if cm.adverse_selection_bps is None else cm.adverse_selection_bps
        fee = p * cm.fee_maker_bps + (1 - p) * taker_fee
        slip = p * (adverse - half) + (1 - p) * (taker_slip + latency)  # chasing a missed fill costs extra drift
    else:
        fee, slip = taker_fee, taker_slip
    return SideCosts(fee=cfg.fee_mult * cm.cost_mult * np.asarray(fee, float),
                     slip=cfg.slip_mult * cm.cost_mult * np.asarray(slip, float))


def round_trip_bps(raw: pd.DataFrame, cfg: CostConfig = CostConfig()) -> pd.Series:
    return pd.Series(2 * side_costs(raw, cfg).total, index=raw.index)
