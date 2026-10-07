"""The one signal -> position rule every model goes through.

    signal[t] = sign(score[t]) if |score[t]| >= threshold else 0
    position[t] = the most recent non-zero signal within the last `hold` bars (a newer signal overrides an older
                  one, an opposite signal flips), else FLAT

Thresholds are candidates chosen on the validation window only: quantiles of |score| on validation, plus (for
scores in bps) the economic rule |E[gross]| >= round-trip cost. If no candidate makes money on validation the
model abstains (stays FLAT) for that test fold: NO TRADE is always allowed.
"""

import numpy as np

QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.98)


def signal(score: np.ndarray, threshold: float) -> np.ndarray:
    s = np.nan_to_num(np.asarray(score, float))
    return np.where(np.abs(s) >= threshold, np.sign(s), 0.0) * (np.abs(s) > 0)


def hold_positions(sig: np.ndarray, hold: int) -> np.ndarray:
    sig = np.asarray(sig, float)
    n = len(sig)
    if hold <= 1:
        return sig.copy()
    idx = np.where(sig != 0, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    age = np.arange(n) - last
    out = np.where((last >= 0) & (age < hold), sig[np.clip(last, 0, None)], 0.0)
    return out


def positions(score: np.ndarray, threshold: float, hold: int) -> np.ndarray:
    return hold_positions(signal(score, threshold), hold)


def candidate_thresholds(val_score: np.ndarray, rt_cost_bps: float | None = None) -> list[float]:
    a = np.abs(np.asarray(val_score, float))
    a = a[np.isfinite(a) & (a > 0)]
    if len(a) == 0:
        return []
    th = [float(np.quantile(a, q)) for q in QUANTILES]
    if rt_cost_bps is not None:
        th.append(float(rt_cost_bps))
    return sorted(set(th))
