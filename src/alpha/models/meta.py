"""Meta model: given a strategy setup and its context, estimate P(win) and the expected R after costs.

ev_mode:
  avg    EV = P(win) * avg_win_r + (1 - P(win)) * avg_loss_r, averages per (strategy, tf, exit_policy)
  setup  EV = P(win) * (gross_win_r - cost_r) + (1 - P(win)) * (gross_loss_r - cost_r), where cost_r is THIS
         setup's round-trip cost in R (fees are a much bigger share of R on tight stops)
calibrate: isotonic map from raw to calibrated P(win), fitted on time-ordered out-of-fold predictions.
One pooled LightGBM across strategies/symbols/timeframes (setups are sparse; pooling gives more examples).
"""

from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from alpha.features.build import CORE_COLS
from alpha.regime.hmm import REGIMES

GEOMETRY_COLS = ["risk_atr", "rr", "dist_vwap_atr", "dist_ema50_atr", "trend_align", "risk"]
REGIME_COLS = [f"p_{r}" for r in REGIMES] + ["p_switch", "regime_dur"]
CAT_COLS = ["strategy", "tf", "symbol", "regime"]
NUM_COLS = CORE_COLS + GEOMETRY_COLS + REGIME_COLS + ["side"]
PAYOFF_KEYS = ["strategy", "tf", "exit_policy"]
ROUND_TRIP = 0.0012  # taker fees + slippage, both sides (alpha.strategies.labels.Costs().round_trip)

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, n_estimators=300, num_leaves=15, min_child_samples=80,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1)


def signal_cost_r(df: pd.DataFrame, round_trip: float = ROUND_TRIP) -> np.ndarray:
    """Round-trip cost in R from signal-time geometry (known before entry, so usable live)."""
    risk = np.abs(df["close_px"].to_numpy(float) - df["stop"].to_numpy(float)) / df["close_px"].to_numpy(float)
    return round_trip / np.where(risk > 0, risk, np.nan)


@dataclass
class MetaModel:
    model: lgb.LGBMClassifier
    categories: dict[str, list[str]]
    payoff: pd.DataFrame
    features: list[str] = field(default_factory=list)
    ev_mode: str = "avg"
    calibrator: IsotonicRegression | None = None

    @classmethod
    def fit(cls, train: pd.DataFrame, seed: int = 0, extra: list[str] | None = None, ev_mode: str = "avg",
            calibrate: bool = False, weights: np.ndarray | None = None) -> "MetaModel":
        train = _with_exit(train)
        feats = NUM_COLS + (extra or []) + CAT_COLS
        cats = {c: sorted(train[c].dropna().astype(str).unique()) for c in CAT_COLS}
        y = train["win"].astype(int).to_numpy()
        m = _fit_lgb(_prep(train, feats, cats), y, weights, seed)
        calibrator = _fit_calibrator(train, feats, cats, y, weights, seed) if calibrate else None
        return cls(m, cats, _payoff(train), feats, ev_mode, calibrator)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _with_exit(df)
        p = self.model.predict_proba(_prep(df, self.features, self.categories))[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        keys = PAYOFF_KEYS if self.payoff.index.nlevels == len(PAYOFF_KEYS) else ["strategy", "tf"]  # older models
        pay = self.payoff.reindex(pd.MultiIndex.from_frame(df[keys].astype(str)))
        if self.ev_mode == "setup":
            cost = signal_cost_r(df)
            gw = pay["gross_win_r"].fillna(self.payoff["gross_win_r"].mean()).to_numpy()
            gl = pay["gross_loss_r"].fillna(self.payoff["gross_loss_r"].mean()).to_numpy()
            ev = p * (gw - cost) + (1 - p) * (gl - cost)
        else:
            win_r = pay["avg_win_r"].fillna(self.payoff["avg_win_r"].mean()).to_numpy()
            loss_r = pay["avg_loss_r"].fillna(self.payoff["avg_loss_r"].mean()).to_numpy()
            ev = p * win_r + (1 - p) * loss_r
        return pd.DataFrame({"p_win": p, "ev_r": ev}, index=df.index)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """df plus p_win / ev_r columns (positional; never join on the repeating signal-time index)."""
        pred = self.predict(df)
        return df.assign(p_win=pred["p_win"].to_numpy(), ev_r=pred["ev_r"].to_numpy())

    def importance(self, top: int = 20) -> pd.Series:
        return pd.Series(self.model.booster_.feature_importance("gain"), index=self.features).nlargest(top)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "MetaModel":
        m = joblib.load(path)
        # models saved before ev_mode/calibration existed
        m.__dict__.setdefault("ev_mode", "avg")
        m.__dict__.setdefault("calibrator", None)
        return m


def recency_weights(index: pd.DatetimeIndex, as_of: pd.Timestamp, half_life_days: float | None) -> np.ndarray | None:
    if not half_life_days:
        return None
    age = (as_of - index).total_seconds().to_numpy() / 86400
    return np.power(0.5, np.clip(age, 0, None) / half_life_days)


def _with_exit(df: pd.DataFrame) -> pd.DataFrame:
    return df if "exit_policy" in df else df.assign(exit_policy="fixed")


def _fit_lgb(X, y, weights, seed):
    m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    m.fit(X, y, sample_weight=weights)
    return m


def _fit_calibrator(train, feats, cats, y, weights, seed, n_folds: int = 3):
    """Expanding time folds: train on the first k blocks, predict block k+1. Isotonic fit on those OOF preds."""
    n = len(train)
    edges = np.linspace(0, n, n_folds + 2).astype(int)
    order = np.argsort(train.index.to_numpy(), kind="stable")
    raw, ys, ws = [], [], []
    for k in range(1, n_folds + 1):
        tr, te = order[: edges[k]], order[edges[k]: edges[k + 1]]
        if len(tr) < 500 or len(te) < 100 or len(np.unique(y[tr])) < 2:
            continue
        m = _fit_lgb(_prep(train.iloc[tr], feats, cats), y[tr], None if weights is None else weights[tr], seed)
        raw.append(m.predict_proba(_prep(train.iloc[te], feats, cats))[:, 1])
        ys.append(y[te])
        ws.append(None if weights is None else weights[te])
    if not raw:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    w = None if weights is None else np.concatenate(ws)
    iso.fit(np.concatenate(raw), np.concatenate(ys), sample_weight=w)
    return iso


def _prep(df: pd.DataFrame, feats: list[str], cats: dict[str, list[str]]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in feats:
        if c in cats:
            X[c] = pd.Categorical(df[c].astype(str) if c in df else None, categories=cats[c])
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df else np.nan
    return X


def _payoff(train: pd.DataFrame) -> pd.DataFrame:
    t = train.assign(gross_r=train["gross"] / train["risk"])
    g = t.groupby(PAYOFF_KEYS)
    return pd.DataFrame({
        "avg_win_r": g["r"].apply(lambda r: r[r > 0].mean() if (r > 0).any() else 0.0),
        "avg_loss_r": g["r"].apply(lambda r: r[r <= 0].mean() if (r <= 0).any() else -1.0),
        "gross_win_r": t[t["win"]].groupby(PAYOFF_KEYS)["gross_r"].mean(),
        "gross_loss_r": t[~t["win"]].groupby(PAYOFF_KEYS)["gross_r"].mean(),
    })
