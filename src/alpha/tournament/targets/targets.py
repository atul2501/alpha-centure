"""Targets. A decision at the close of bar t fills at open[t+1] and exits at open[t+1+h]:

    fwd_h (bps) = 1e4 * log(open[t+1+h] / open[t+1])

    reg_<h>       regression on fwd_h / (rvol_96 * sqrt(h))  (vol-normalized, clipped at +-5; the model's score is
                  mapped back to bps with the vol known at t)
    cls_<h>_<k>   3-class LONG / SHORT / FLAT: LONG if fwd_h > k x round-trip cost at t, SHORT if < -k x cost,
                  else FLAT. k = 0 is plain up / down.

The label of row t uses bars t+1 .. t+1+h, so training rows must end h + 2 bars before any held-out row (purge).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 4, 8, 16)
MAX_H = max(HORIZONS)
VOL_FLOOR_BPS = 5.0


@dataclass(frozen=True)
class Target:
    kind: str     # reg | cls
    h: int
    k: float = 0.0

    @property
    def name(self) -> str:
        return f"reg_{self.h}" if self.kind == "reg" else f"cls_{self.h}_{self.k:g}"

    @staticmethod
    def parse(s: str) -> "Target":
        p = s.split("_")
        if p[0] == "reg":
            return Target("reg", int(p[1]))
        if p[0] == "cls":
            return Target("cls", int(p[1]), float(p[2]) if len(p) > 2 else 0.0)
        raise ValueError(f"unknown target {s!r}")


def fwd_bps(open_: pd.Series, h: int) -> pd.Series:
    o = open_.astype(float)
    return 1e4 * np.log(o.shift(-(1 + h)) / o.shift(-1))


def bar_vol_bps(close: pd.Series) -> pd.Series:
    """Per-bar return vol (bps) known at t."""
    return (1e4 * np.log(close / close.shift(1))).rolling(96, min_periods=48).std().clip(lower=VOL_FLOOR_BPS)


def make(t: Target, open_: pd.Series, close: pd.Series, rt_cost_bps: pd.Series) -> pd.Series:
    f = fwd_bps(open_, t.h)
    if t.kind == "reg":
        scale = bar_vol_bps(close) * np.sqrt(t.h)
        return (f / scale).clip(-5, 5)
    thr = t.k * rt_cost_bps
    y = pd.Series(np.where(f > thr, 2, np.where(f < -thr, 0, 1)), index=f.index, dtype=float)
    return y.where(f.notna())


def score_scale(t: Target, close: pd.Series) -> pd.Series:
    """Multiply a reg model's prediction by this to get expected bps; 1 for classifiers."""
    if t.kind == "reg":
        return bar_vol_bps(close) * np.sqrt(t.h)
    return pd.Series(1.0, index=close.index)
