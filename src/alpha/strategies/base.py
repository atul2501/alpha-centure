"""Strategy interface. A strategy looks at a feature frame (from alpha.features.build) and emits setups.

Setup contract (one row per signal, index = signal bar open time):
    side      +1 long / -1 short
    stop      stop-loss price
    target    take-profit price (None/NaN = exit on time limit only)
    max_bars  time limit in bars after entry
Signals use only the signal bar and earlier. Entry happens at the NEXT bar's open (see alpha.strategies.labels).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SETUP_COLS = ["side", "stop", "target", "max_bars"]


@dataclass
class Strategy:
    name: str
    timeframes: tuple[str, ...]
    max_bars: int = 48
    cooldown: int = 3  # min bars between setups of the same side, avoids near-duplicate samples
    params: dict = field(default_factory=dict)

    def signals(self, f: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with SETUP_COLS on signal rows only. Implemented by subclasses."""
        raise NotImplementedError

    def setups(self, f: pd.DataFrame) -> pd.DataFrame:
        s = self.signals(f)
        if s.empty:
            return pd.DataFrame(columns=["strategy", *SETUP_COLS])
        s = s.dropna(subset=["side", "stop"])
        s = s[_cooldown_mask(f.index, s, self.cooldown)]
        s["strategy"] = self.name
        s["max_bars"] = s["max_bars"].fillna(self.max_bars).astype(int)
        return s[["strategy", *SETUP_COLS]]


def _cooldown_mask(index: pd.DatetimeIndex, s: pd.DataFrame, n: int) -> np.ndarray:
    if n <= 0 or s.empty:
        return np.ones(len(s), dtype=bool)
    pos = index.get_indexer(s.index)
    keep = np.ones(len(s), dtype=bool)
    last: dict[int, int] = {}
    for i, (p, side) in enumerate(zip(pos, s["side"].to_numpy())):
        if side in last and p - last[side] <= n:
            keep[i] = False
        else:
            last[side] = p
    return keep


def build_setups(side: pd.Series, stop: pd.Series, target: pd.Series, max_bars: int | None = None) -> pd.DataFrame:
    """Helper: keep rows where side != 0."""
    out = pd.DataFrame({"side": side, "stop": stop, "target": target, "max_bars": max_bars})
    return out[out["side"].fillna(0) != 0]


def r_target(entry_ref: pd.Series, stop: pd.Series, side: pd.Series, r: float) -> pd.Series:
    """Target at r times the risk distance, measured from the signal close."""
    return entry_ref + side * r * (entry_ref - stop).abs()


def clamp_stop(close: pd.Series, stop: pd.Series, atr: pd.Series, side: pd.Series,
               lo_atr: float = 0.5, hi_atr: float = 3.0) -> pd.Series:
    """Keep risk between lo_atr and hi_atr ATRs so one setup can't have an absurd stop."""
    dist = ((close - stop) * side).clip(lower=lo_atr * atr, upper=hi_atr * atr)
    return close - side * dist
