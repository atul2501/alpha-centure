"""Walk-forward folds inside DEV (purged + embargoed), and purged K-fold for search inside a training window.

    test folds  quarterly, 2021-07-01 -> DEV_END (12 folds)
    val         the 90 days before the test fold, minus the gap
    train       everything from the data start up to val start, minus the gap (expanding window)
    gap         MAX_H + 2 bars (a label of row t uses bars up to t + 1 + h) + an embargo of 16 bars

The same folds and gap are used by every model (fair comparison).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.research.splits import DEV_END
from alpha.tournament.targets.targets import MAX_H

BAR = pd.Timedelta(minutes=15)
EMBARGO_BARS = 16
GAP = (MAX_H + 2 + EMBARGO_BARS) * BAR
OOS_START = pd.Timestamp("2021-07-01", tz="UTC")
VAL_DAYS = 90


@dataclass(frozen=True)
class Fold:
    k: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp   # exclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp     # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp    # exclusive

    def masks(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tr = (index >= self.train_start) & (index < self.train_end)
        va = (index >= self.val_start) & (index < self.val_end)
        te = (index >= self.test_start) & (index < self.test_end)
        return tr, va, te


def walk_forward(data_start: pd.Timestamp, oos_start: pd.Timestamp = OOS_START, end: pd.Timestamp = DEV_END,
                 freq: str = "QS", val_days: int = VAL_DAYS, gap: pd.Timedelta = GAP) -> list[Fold]:
    starts = pd.date_range(oos_start, end, freq=freq, tz="UTC")
    starts = [s for s in starts if s < end]
    folds = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else end
        val_start = s - pd.Timedelta(days=val_days)
        folds.append(Fold(k, data_start, val_start - gap, val_start, s - gap, s, e))
    return folds


def purged_kfold(index: pd.DatetimeIndex, n_splits: int = 5, gap: pd.Timedelta = GAP):
    """Contiguous K-fold for hyper-parameter search inside one training window: rows within `gap` of the
    validation block on either side are dropped from training (purge before, embargo after)."""
    n = len(index)
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    for i in range(n_splits):
        va = np.zeros(n, bool)
        va[bounds[i]:bounds[i + 1]] = True
        lo, hi = index[bounds[i]], index[bounds[i + 1] - 1]
        tr = ~va & ((index < lo - gap) | (index > hi + gap))
        yield tr, va
