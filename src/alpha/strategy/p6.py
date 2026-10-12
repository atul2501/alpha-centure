"""V4_carry: ridge-forecast book + momentum rules + funding carry, scaled to the 20% vol target (PAPER mode only).

One implementation for research replay and the live/paper engine, built from the research functions themselves:
    features   alpha.research.models.feature_frame (momentum, trend, carry, basis, OI, positioning, flow, vol)
    forecast   ridge (alpha=100) on standardized features -> 72h vol-normalized return, trained on every row whose
               target window closed before train_end (hours % 4 == 0 sample), refit monthly
    ridge half inverse-vol on the forecast, gross 1 -> 20% vol target
    rules half cross-sectional momentum (1/2/4-week rank, dollar-neutral) + time-series momentum (2/4-week trend,
               20-day breakout), 50/50, inverse-vol -> 20% vol target. No trained model.
    carry part cross-sectional funding carry (short coins whose 24h-average funding is high, long low), dollar-neutral,
               inverse-vol -> 20% vol target
    book       0.5 ridge + 0.25 rules + 0.25 carry, then scaled back to the 20% vol target, 3x gross cap, 0.5x per
               coin (alpha.research.phase4)
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
from alpha.research.phase4 import MAX_COIN, MAX_GROSS, TARGET_VOL, momentum_scores, vol_target
from alpha.research.portfolio_sim import to_weights
from alpha.strategy.trace import BLEND, BOOK, FINAL, SIGNAL, Layer, StrategyTrace

NAME = "V4_carry"
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


def rules_weights(panel: pd.DataFrame) -> pd.DataFrame:
    """Momentum-rules half: xs + ts momentum, 50/50, inverse-vol, vol-targeted (time x symbol)."""
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    el = wide(panel, "eligible").fillna(False).astype(bool)
    xs, ts = momentum_scores(panel, 1.0)
    w = (to_weights(xs.reindex(columns=ret.columns), vol, el, "xs", every=H) +
         to_weights(ts.reindex(columns=ret.columns), vol, el, "ts", every=H)) / 2
    return vol_target(w, ret)


def carry_weights(panel: pd.DataFrame) -> pd.DataFrame:
    """Funding-carry part: short coins whose longs pay high funding, long coins with low funding (time x symbol)."""
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    el = wide(panel, "eligible").fillna(False).astype(bool)
    carry = sig.compute(panel)["xscarry"].reindex(columns=ret.columns)
    return vol_target(to_weights(carry, vol, el, "xs", every=H), ret)


MIX = {"ridge": 0.5, "rules": 0.25, "carry": 0.25}


def book_parts(panel: pd.DataFrame, s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every step of the book (time x symbol): the three sub-books, their blend and the final re-scaled book."""
    base = target_weights(panel, s)
    parts = {"ridge": base,
             "rules": rules_weights(panel).reindex_like(base).fillna(0.0),
             "carry": carry_weights(panel).reindex_like(base).fillna(0.0)}
    parts["blend"] = sum(MIX[k] * parts[k] for k in MIX)
    parts["final"] = vol_target(parts["blend"], wide(panel, "ret"))
    return parts


def book_weights(panel: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """The traded book before the band: 0.5 ridge + 0.25 rules + 0.25 carry, re-scaled to the 20% vol target."""
    return book_parts(panel, s)["final"]


def is_rebalance(bar_open: pd.Timestamp) -> bool:
    hours = (bar_open - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
    return hours % H == 0


def train(conn: psycopg.Connection, symbols: list[str], train_end: pd.Timestamp) -> RidgeBundle:
    panel = build_panel(conn, symbols, end=train_end)
    X, vol = feature_frame(panel, sig.compute(panel))
    return fit(X, target(panel, vol, H), train_end)


def _row(df: pd.DataFrame, t: pd.Timestamp, nd: int = 5) -> dict[str, float]:
    r = df.loc[t] if t in df.index else pd.Series(dtype=float)
    return {str(k): round(float(v), nd) for k, v in r.items() if pd.notna(v)}


def explain(panel: pd.DataFrame, scores: dict[str, pd.DataFrame], X: pd.DataFrame, s: pd.DataFrame,
            parts: dict[str, pd.DataFrame], b: RidgeBundle, last: pd.Timestamp) -> StrategyTrace:
    """The decision at bar `last`, step by step, in the strategy-agnostic trace format."""
    xs, ts = momentum_scores(panel, 1.0)
    vol = wide(panel, "ret").rolling(sig.VOL_WINDOW, min_periods=48).std() * np.sqrt(24 * 365)
    feats = None
    if last in X.index.get_level_values("time"):
        z = (X.xs(last, level="time")[b.cols].fillna(0.0) - b.mu) / b.sd
        contrib = z.mul(pd.Series(b.model.coef_, index=b.cols), axis=1)
        feats = {str(sym): {k: round(float(v), 4) for k, v in row.items()} for sym, row in contrib.iterrows()}
    layers = [
        Layer("forecast", "Ridge forecast", SIGNAL, "expected 72h return / vol, from 19 features", _row(s, last, 4),
              features=feats),
        Layer("xsmom", "Momentum vs other coins", SIGNAL, "1/2/4-week return rank across coins", _row(xs, last, 4)),
        Layer("tsmom", "Own trend", SIGNAL, "2/4-week trend + 20-day breakout", _row(ts, last, 4)),
        Layer("funding", "Funding carry", SIGNAL, "low 24h funding = long, high = short",
              _row(scores["xscarry"], last, 4)),
        Layer("vol", "Volatility (annual)", SIGNAL, "7-day realized; bigger = smaller position", _row(vol, last, 3)),
        Layer("ridge", "Ridge book", BOOK, "inverse-vol on the forecast, 20% vol", _row(parts["ridge"], last),
              blend_weight=MIX["ridge"], inputs=["forecast", "vol"]),
        Layer("rules", "Momentum rules book", BOOK, "cross-sectional + time-series momentum, 20% vol",
              _row(parts["rules"], last), blend_weight=MIX["rules"], inputs=["xsmom", "tsmom", "vol"]),
        Layer("carry", "Funding carry book", BOOK, "short high funding, long low funding, 20% vol",
              _row(parts["carry"], last), blend_weight=MIX["carry"], inputs=["funding", "vol"]),
        Layer("blend", "Blend 50/25/25", BLEND, "0.5 ridge + 0.25 rules + 0.25 carry", _row(parts["blend"], last)),
        Layer("final", "Vol target + caps", FINAL, f"scaled to {TARGET_VOL:.0%} vol, max {MAX_GROSS:g}x gross, "
              f"{MAX_COIN:g}x per coin", _row(parts["final"], last)),
    ]
    return StrategyTrace(NAME, last.isoformat(), f"every {H}h", {
        "vol target": TARGET_VOL, "max gross": MAX_GROSS, "max per coin": MAX_COIN, "no-trade band": BAND,
        "model trained up to": str(b.train_end.date())}, layers, _row(parts["final"], last))


def live_targets_explained(conn: psycopg.Connection, symbols: list[str], b: RidgeBundle, now: pd.Timestamp
                           ) -> tuple[pd.Timestamp, pd.Series, pd.DataFrame, StrategyTrace]:
    """live_targets plus the step-by-step trace of how the targets were built (same panel, no extra queries)."""
    start = (now - pd.Timedelta(days=HISTORY_DAYS)).floor("h")
    panel = build_panel(conn, symbols, start=start, end=now)
    scores = sig.compute(panel)
    X, _ = feature_frame(panel, scores)
    s = score(b, X)
    parts = book_parts(panel, s)
    w = parts["final"]
    last = w.index[w.index + pd.Timedelta(hours=1) <= now][-1]
    return last, w.loc[last].fillna(0.0), panel, explain(panel, scores, X, s, parts, b, last)


def live_targets(conn: psycopg.Connection, symbols: list[str], b: RidgeBundle,
                 now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series, pd.DataFrame]:
    """(last closed bar open time, target weights for it, recent panel) using data known at `now`."""
    last, w, panel, _ = live_targets_explained(conn, symbols, b, now)
    return last, w, panel
