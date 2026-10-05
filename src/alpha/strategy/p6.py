"""P6: ridge forecast book (the finalist that passed VALID-A and failed VALID-B; run in PAPER mode only).

One implementation for research replay and the live/paper engine, built from the research functions themselves:
    features   alpha.research.models.feature_frame (momentum, trend, carry, basis, OI, positioning, flow, vol)
    forecast   ridge (alpha=100) on standardized features -> 72h vol-normalized return, trained on every row whose
               target window closed before train_end (hours % 4 == 0 sample), refit monthly
    weights    inverse-vol, gross 1 -> 20% vol target, 3x gross cap, 0.5x per coin (alpha.research.phase4)
    schedule   decision at the close of bars whose open hour (since epoch) % 72 == 0
The 1% no-trade band is applied by the OMS against the account's ACTUAL positions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from sklearn.linear_model import Ridge

from alpha.research import signals as sig
from alpha.research.models import feature_frame, target
from alpha.research.panel import build_panel, wide
from alpha.research.phase4 import vol_target
from alpha.research.portfolio_sim import to_weights

NAME = "P6_m1_band_maker"
H = 72
RIDGE_ALPHA = 100.0
HISTORY_DAYS = 200      # live window: longest lookback (1320h) + 60-day vol estimate, with margin
BAND = 0.01
MAKER_SHARE_ASSUMED = 0.6


@dataclass
class RidgeBundle:
    cols: list[str]
    mu: pd.Series
    sd: pd.Series
    model: Ridge
    train_end: pd.Timestamp
    fitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def save(self, models_dir: str) -> Path:
        path = Path(models_dir) / f"p6_ridge_{self.train_end:%Y%m%d}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def latest(models_dir: str) -> "RidgeBundle | None":
        files = sorted(Path(models_dir).glob("p6_ridge_*.joblib"))
        return joblib.load(files[-1]) if files else None


def fit(X: pd.DataFrame, y: pd.Series, train_end: pd.Timestamp, h: int = H) -> RidgeBundle:
    """Same rows / scaling / model as research walk_forward('M1')."""
    times = X.index.get_level_values("time")
    mask = (times + pd.Timedelta(hours=2 * h) <= train_end) & (times.hour % 4 == 0)
    tr = X[mask].assign(y=y[mask]).dropna(subset=["y"])
    cols = list(X.columns)
    Xtr = tr[cols].fillna(0.0)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    return RidgeBundle(cols, mu, sd, Ridge(alpha=RIDGE_ALPHA).fit((Xtr - mu) / sd, tr["y"]), train_end)


def score(b: RidgeBundle, X: pd.DataFrame) -> pd.DataFrame:
    p = b.model.predict((X[b.cols].fillna(0.0) - b.mu) / b.sd)
    return pd.Series(p, index=X.index).unstack("symbol")


def target_weights(panel: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Vol-targeted weights (time x symbol) before the no-trade band."""
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    el = wide(panel, "eligible").fillna(False).astype(bool)
    t = ret.index
    w = to_weights(s.reindex(t).reindex(columns=ret.columns), vol, el, "ts", every=H)
    return vol_target(w, ret)


def is_rebalance(bar_open: pd.Timestamp) -> bool:
    hours = (bar_open - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
    return hours % H == 0


def train(conn: psycopg.Connection, symbols: list[str], train_end: pd.Timestamp) -> RidgeBundle:
    panel = build_panel(conn, symbols, end=train_end)
    X, vol = feature_frame(panel, sig.compute(panel))
    return fit(X, target(panel, vol, H), train_end)


def live_targets(conn: psycopg.Connection, symbols: list[str], b: RidgeBundle,
                 now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series, pd.DataFrame]:
    """(last closed bar open time, target weights for it, recent panel) using data known at `now`."""
    start = (now - pd.Timedelta(days=HISTORY_DAYS)).floor("h")
    panel = build_panel(conn, symbols, start=start, end=now)
    X, _ = feature_frame(panel, sig.compute(panel))
    w = target_weights(panel, score(b, X))
    last = w.index[w.index + pd.Timedelta(hours=1) <= now][-1]
    return last, w.loc[last].fillna(0.0), panel
