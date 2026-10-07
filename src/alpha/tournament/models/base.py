"""BaseTradingModel: the one interface every tournament entry implements.

Contract (enforced by tests/tournament/test_model_contract.py):
    fit(X, y, train, val)    X: full feature frame up to the end of the test fold, y: target aligned to X,
                             train / val: boolean masks. The model may read X rows outside `train` only to build
                             causal inputs (sequence windows, filters); it must fit parameters on `train` rows
                             (and `val` for early stopping / calibration) only.
    predict(X, rows)         raw model output for the requested rows (NaN elsewhere). For regression targets this
                             is the vol-normalized expected forward return; for classifiers p_long - p_short.
                             The value at row t must not change if rows after t are removed (causality test).
    predict_proba(X, rows)   (n, 3) = [p_short, p_flat, p_long]
    generate_signal(X, rows, threshold, hold)   positions via the shared policy
    save(path) / load(path)

Class attributes: family, complexity (0 rules, 1 linear/classical, 2 tree/regime, 3 deep/hybrid/meta,
4 transformer/SSM/foundation/RL/ensemble), input_kind, target_kinds, experimental.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from alpha.tournament.backtesting import policy
from alpha.tournament.targets.targets import Target


class BaseTradingModel:
    name: str = "base"
    family: str = "base"
    complexity: int = 1
    input_kind: str = "tabular"       # tabular | sequence | series | members
    target_kinds: tuple[str, ...] = ("reg", "cls")
    experimental: bool = False
    description: str = ""

    def __init__(self, target: Target, params: dict | None = None, seed: int = 0, columns: list[str] | None = None):
        self.target = target
        self.params = dict(params or {})
        self.seed = seed
        self.columns = columns
        self._calib: tuple[float, float] | None = None

    # ---- to implement ----
    def fit(self, X: pd.DataFrame, y: pd.Series, train: np.ndarray, val: np.ndarray) -> "BaseTradingModel":
        raise NotImplementedError

    def _raw(self, X: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        """Model output for X[rows] (length rows.sum()): regression value, or (n, 3) class probabilities."""
        raise NotImplementedError

    # ---- shared ----
    def _cols(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns] if self.columns is not None else X

    def predict(self, X: pd.DataFrame, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.ones(len(X), bool) if rows is None else rows
        out = np.full(len(X), np.nan)
        if rows.any():
            r = np.asarray(self._raw(X, rows), float)
            out[rows] = r[:, 2] - r[:, 0] if r.ndim == 2 else r
        return out

    def predict_proba(self, X: pd.DataFrame, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.ones(len(X), bool) if rows is None else rows
        out = np.full((len(X), 3), np.nan)
        if not rows.any():
            return out
        r = np.asarray(self._raw(X, rows), float)
        if r.ndim == 2:
            out[rows] = r
        else:  # regression: P(up) from a logistic calibration of the sign, fit on validation
            a, b = self._calib or (1.0, 0.0)
            p = 1 / (1 + np.exp(-(a * r + b)))
            out[rows] = np.column_stack([1 - p, np.zeros_like(p), p])
        return out

    def calibrate(self, X: pd.DataFrame, y: pd.Series, val: np.ndarray) -> None:
        """Fit the regression -> P(up) map on validation rows (classifiers are already probabilistic)."""
        if self.target.kind != "reg" or val.sum() < 50:
            return
        from sklearn.linear_model import LogisticRegression

        r = self.predict(X, val)[val]
        yy = y.to_numpy()[val]
        ok = np.isfinite(r) & np.isfinite(yy) & (yy != 0)
        if ok.sum() < 50 or np.nanstd(r[ok]) == 0:
            return
        lr = LogisticRegression(C=1.0).fit(r[ok, None], (yy[ok] > 0).astype(int))
        self._calib = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def generate_signal(self, X: pd.DataFrame, rows: np.ndarray | None, threshold: float, hold: int) -> np.ndarray:
        return policy.positions(self.predict(X, rows), threshold, hold)

    def feature_importance(self) -> dict[str, float] | None:
        return None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mode-2 features for hybrids. Default: the model's own causal score (regime models add more)."""
        return pd.DataFrame({f"{self.name}_score": self.predict(X)}, index=X.index)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("context", None)  # the shared data context is never pickled with a model
        return d

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "BaseTradingModel":
        return joblib.load(path)


def train_rows(train: np.ndarray, y: pd.Series, X: pd.DataFrame | None = None, stride: int = 1,
               max_rows: int | None = None) -> np.ndarray:
    """Indices of usable training rows (label known), optionally strided / capped to the most recent max_rows."""
    ok = train & y.notna().to_numpy()
    idx = np.flatnonzero(ok)
    if stride > 1:
        idx = idx[::stride]
    if max_rows is not None and len(idx) > max_rows:
        idx = idx[-max_rows:]
    return idx


class Preprocessor:
    """Median impute + standardize + clip, fit on training rows only (records what it was fit on)."""

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.fit_range: tuple[pd.Timestamp, pd.Timestamp] | None = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        a = X.to_numpy(np.float64)
        self.med = np.nanmedian(a, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        a = np.where(np.isfinite(a), a, self.med)
        self.mu = a.mean(axis=0)
        self.sd = a.std(axis=0)
        self.sd = np.where(self.sd > 1e-12, self.sd, 1.0)
        self.fit_range = (X.index.min(), X.index.max())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        a = X.to_numpy(np.float64)
        a = np.where(np.isfinite(a), a, self.med)
        return np.clip((a - self.mu) / self.sd, -self.clip, self.clip).astype(np.float32)
