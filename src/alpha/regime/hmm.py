"""Market regime model: Gaussian HMM on 4h features, pooled across symbols, with causal (forward-only) filtering.

Why a custom filter: hmmlearn's predict/predict_proba run forward-backward over the whole sequence, so the
probability at bar t would use bars after t. Here P(state_t | x_1..x_t) is computed with the forward
recursion only, so it is exactly what a live system could have known at t.
"""

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import psycopg
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from alpha.features.build import base_symbol, load_candles
from alpha.features.indicators import bollinger, ema

REGIME_TF = "4h"
FEATURES = ["ret_6", "rvol", "slope", "bb_width"]
REGIMES = ["trend_up", "trend_down", "range", "squeeze", "extreme"]
MIN_FIT_ROWS = 50


def regime_features(c: pd.DataFrame) -> pd.DataFrame:
    """Per-bar 4h features (causal). Index = bar open time; keeps close_time for joining."""
    lr = np.log(c["close"]).diff()
    out = pd.DataFrame({
        "ret_6": np.log(c["close"] / c["close"].shift(6)),
        "rvol": np.log(lr.rolling(20).std(ddof=0)),
        "slope": ema(c["close"], 50).pct_change(5),
        "bb_width": np.log(bollinger(c["close"])["bb_width"]),
        "close_time": c["close_time"],
    }, index=c.index)
    return out.dropna()


@dataclass
class RegimeModel:
    hmm: GaussianHMM
    center: np.ndarray
    scale: np.ndarray
    names: list[str]  # names[state_index] -> regime name
    meta: dict = field(default_factory=dict)

    # ---------- fitting ----------
    @classmethod
    def fit(cls, frames: dict[str, pd.DataFrame], n_states: int = 5, seed: int = 0, n_init: int = 5) -> "RegimeModel":
        # symbols listed after the training cut-off (e.g. SUI before 2023) have no rows yet: skip them
        frames = {k: f for k, f in frames.items() if len(f) >= MIN_FIT_ROWS}
        if not frames:
            raise ValueError("not enough 4h history to fit the regime model")
        X_raw = np.vstack([f[FEATURES].to_numpy() for f in frames.values()])
        lengths = [len(f) for f in frames.values()]
        center = np.median(X_raw, axis=0)
        scale = np.subtract(*np.percentile(X_raw, [75, 25], axis=0)) / 1.349
        X = (X_raw - center) / scale
        best, best_ll = None, -np.inf
        for k in range(n_init):  # EM finds local optima: keep the best of several starts
            m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, tol=1e-4,
                            random_state=seed + k, min_covar=1e-3)
            m.fit(X, lengths)
            ll = m.score(X, lengths)
            if ll > best_ll:
                best, best_ll = m, ll
        names = _name_states(best.means_ * scale + center, n_states)
        return cls(best, center, scale, names, {"loglik": float(best_ll), "n_obs": int(len(X)),
                                                "bic": float(-2 * best_ll + _n_params(n_states, len(FEATURES)) * np.log(len(X)))})

    # ---------- causal filtering ----------
    def filter(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = (feats[FEATURES].to_numpy() - self.center) / self.scale
        m = self.hmm
        log_b = np.column_stack([multivariate_normal.logpdf(X, m.means_[k], m.covars_[k], allow_singular=True)
                                 for k in range(m.n_components)])
        log_A = np.log(m.transmat_ + 1e-300)
        alpha = np.empty_like(log_b)
        prev = np.log(m.startprob_ + 1e-300) + log_b[0]
        alpha[0] = prev - logsumexp(prev)
        for t in range(1, len(X)):
            prev = logsumexp(alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]
            alpha[t] = prev - logsumexp(prev)
        p = np.exp(alpha)
        out = pd.DataFrame(p, index=feats.index, columns=[f"p_{self.names[k]}" for k in range(m.n_components)])
        # collapse duplicates if two states got the same name (e.g. 4-state model)
        out = out.T.groupby(level=0).sum().T
        for name in REGIMES:
            if f"p_{name}" not in out:
                out[f"p_{name}"] = 0.0
        state = p.argmax(axis=1)
        out["regime"] = [self.names[s] for s in state]
        out["p_switch"] = 1 - (p * np.diag(m.transmat_)).sum(axis=1)
        changed = out["regime"].ne(out["regime"].shift())
        out["regime_dur"] = changed.groupby(changed.cumsum()).cumcount() + 1
        out["close_time"] = feats["close_time"].to_numpy()
        return out

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RegimeModel":
        return joblib.load(path)


def _n_params(k: int, d: int) -> int:
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def _name_states(means: np.ndarray, k: int) -> list[str]:
    """Deterministic names from state means (in original feature units), so IDs are stable across refits."""
    ret, vol, slope, width = (means[:, FEATURES.index(c)] for c in FEATURES)
    names: list[str | None] = [None] * k
    left = set(range(k))

    def take(idx, name):
        names[idx] = name
        left.discard(idx)

    if k >= 5:
        take(max(left, key=lambda i: vol[i]), "extreme")
    if k >= 4:
        take(min(left, key=lambda i: width[i]), "squeeze")
    take(max(left, key=lambda i: slope[i] + ret[i]), "trend_up")
    take(min(left, key=lambda i: slope[i] + ret[i]), "trend_down")
    for i in list(left):
        take(i, "range")
    return names


def load_regime_frames(conn: psycopg.Connection, symbols: list[str], start=None, end=None) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        c = load_candles(conn, base_symbol(s), REGIME_TF, start, end)
        if len(c) > 100:
            out[base_symbol(s)] = regime_features(c)
    return out


def attach_regime(ds: pd.DataFrame, regimes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join filtered regime probabilities onto rows (needs 'symbol' and 'close_time'), as known at close_time."""
    parts = []
    for sym, rows in ds.groupby("symbol", sort=False):
        r = regimes.get(base_symbol(sym))
        if r is None:
            parts.append(rows)
            continue
        r = r.sort_values("close_time")
        left = rows.reset_index().sort_values("close_time")
        merged = pd.merge_asof(left, r.rename(columns={"close_time": "_rt"}), left_on="close_time", right_on="_rt",
                               direction="backward").drop(columns="_rt")
        parts.append(merged.set_index(rows.index.name or "index"))
    return pd.concat(parts).sort_index()
