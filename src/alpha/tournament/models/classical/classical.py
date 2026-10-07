"""Classical statistical baselines on the SOL return series (bps). Parameters are estimated on the most recent
training rows only; forecasts for later rows come from running the fitted model's filter forward (causal: the
value at t uses observations <= t), never from re-estimating on held-out data.

    AR / MA / ARMA     SARIMAX on 15m log returns; score = E[sum of the next h returns]
    ARIMA              SARIMAX(p, 1, q) on log price; score = E[price_{t+h}] - price_t
    VAR                SOL / BTC / ETH returns, OLS; h-step cumulative forecast of SOL
    GARCH / EGARCH / GJR-GARCH   AR(1) mean with the vol model; score = AR forecast / GARCH vol (t-stat-like)
    Kalman             local linear trend on log price with fixed noise ratio (grid); score = slope * h
    State space        statsmodels UnobservedComponents local linear trend (MLE); score = slope * h
"""

import numpy as np
import pandas as pd
from numba import njit

from alpha.tournament.models.base import BaseTradingModel

N_FIT = 20_000  # most recent training rows used for estimation (~7 months of 15m bars)


class _Series(BaseTradingModel):
    family = "classical"
    complexity = 1
    input_kind = "series"
    target_kinds = ("reg",)
    output_units = "bps"
    min_val_trades = 10

    def _returns(self, X: pd.DataFrame, col: str = "ret_1") -> np.ndarray:
        return np.nan_to_num(X[col].to_numpy(np.float64) * 1e4)

    def _fit_slice(self, train: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        idx = np.flatnonzero(train)
        idx = idx[-N_FIT:]
        # winsorization bounds come from training rows only (no held-out statistics)
        self.clip_ = {c: WINSOR_K * float(np.std(self._returns(X, c)[idx]))
                      for c in ("ret_1", "btc_ret_1", "eth_ret_1") if c in X}
        return idx

    def _w(self, X: pd.DataFrame, col: str = "ret_1") -> np.ndarray:
        k = self.clip_[col]
        return np.clip(self._returns(X, col), -k, k)


WINSOR_K = 8.0


class _Sarimax(_Series):
    order = (1, 0, 0)
    level = False

    def fit(self, X, y, train, val):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        order = tuple(self.params.get("order", self.order))
        idx = self._fit_slice(train, X)
        r = self._w(X)
        self.mu = float(r[idx].mean())
        series = np.cumsum(r - self.mu) if self.level else r - self.mu
        self.order_ = order
        self.res = SARIMAX(series[idx], order=order, trend="n").fit(disp=False, maxiter=200)
        return self

    def _raw(self, X, rows):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        r = self._w(X)
        series = np.cumsum(r - self.mu) if self.level else r - self.mu
        res = SARIMAX(series, order=self.order_, trend="n").filter(self.res.params)
        fr = res.filter_results
        Z, T = fr.design[:, :, 0], fr.transition[:, :, 0]
        a = fr.predicted_state[:, 1:]  # a_{t+1|t} for t = 0..n-1
        h = self.target.h
        if self.level:
            M = Z @ np.linalg.matrix_power(T, h - 1)
            fc = (M @ a)[0] - series  # E[level_{t+h}] - level_t
        else:
            M = sum(Z @ np.linalg.matrix_power(T, j) for j in range(h))
            fc = (M @ a)[0]
        return (fc + self.mu * h)[rows]


class AR(_Sarimax):
    name, order = "ar", (2, 0, 0)


class MA(_Sarimax):
    name, order = "ma", (0, 0, 2)


class ARMA(_Sarimax):
    name, order = "arma", (1, 0, 1)


class ARIMA(_Sarimax):
    name, order, level = "arima", (1, 1, 1), True


class VAR(_Series):
    name = "var"

    def fit(self, X, y, train, val):
        from statsmodels.tsa.api import VAR as smVAR

        idx = self._fit_slice(train, X)
        Y = self._Y(X)
        self.p = int(self.params.get("lags", 2))
        res = smVAR(Y[idx]).fit(self.p, trend="c")
        self.c = res.params[0]                      # (k,)
        self.A = res.params[1:].reshape(self.p, Y.shape[1], Y.shape[1]).transpose(0, 2, 1)  # A[l] @ y_{t-l}
        return self

    def _Y(self, X):
        return np.column_stack([self._w(X, c) for c in ("ret_1", "btc_ret_1", "eth_ret_1")])

    def _raw(self, X, rows):
        Y = self._Y(X)
        n, k = Y.shape
        lags = [np.vstack([np.zeros((l, k)), Y[: n - l]]) for l in range(self.p)]  # y_t, y_{t-1}, ...
        total = np.zeros(n)
        for _ in range(self.target.h):
            nxt = self.c + sum(lags[l] @ self.A[l].T for l in range(self.p))
            total += nxt[:, 0]
            lags = [nxt] + lags[:-1]
        return total[rows]


class _Garch(_Series):
    vol = "GARCH"
    o = 0
    output_units = "target"  # score = forecast / GARCH vol, then x realized vol like any reg model

    def fit(self, X, y, train, val):
        from arch import arch_model

        idx = self._fit_slice(train, X)
        r = self._w(X) / 100.0  # percent of a percent: keeps arch's optimizer well scaled
        self.am_kw = dict(mean="AR", lags=1, vol=self.vol, p=1, o=self.o, q=1, dist="t", rescale=False)
        self.res = arch_model(r[idx], **self.am_kw).fit(disp="off", show_warning=False)
        return self

    def _raw(self, X, rows):
        from arch import arch_model

        r = self._w(X) / 100.0
        fixed = arch_model(r, **self.am_kw).fix(self.res.params)
        sig = np.asarray(fixed.conditional_volatility)  # sigma_t given info < t; one step ahead is ~ the same
        mu, phi = self.res.params.iloc[0], self.res.params.iloc[1]
        h = self.target.h
        phi = float(np.clip(phi, -0.99, 0.99))
        fc = mu * h + phi * (1 - phi**h) / (1 - phi) * (r - mu)
        out = fc / (np.maximum(sig, 1e-6) * np.sqrt(h))
        return np.nan_to_num(out)[rows]


class GARCH(_Garch):
    name = "garch"


class EGARCH(_Garch):
    name, vol = "egarch", "EGARCH"


class GJRGARCH(_Garch):
    name, o = "gjr_garch", 1


@njit(cache=True)
def _llt_filter(y, K0, K1):
    """Steady-state Kalman filter for a local linear trend: returns the slope estimate after observing y_t."""
    n = len(y)
    lvl, slope = y[0], 0.0
    out = np.zeros(n)
    for t in range(n):
        pl, ps = lvl + slope, slope   # predict
        e = y[t] - pl
        lvl, slope = pl + K0 * e, ps + K1 * e
        out[t] = slope
    return out


def _steady_gain(q_level: float, q_slope: float, r: float = 1.0) -> tuple[float, float]:
    T = np.array([[1.0, 1.0], [0.0, 1.0]])
    Z = np.array([[1.0, 0.0]])
    Q = np.diag([q_level, q_slope])
    P = np.eye(2)
    for _ in range(20_000):
        Pp = T @ P @ T.T + Q
        S = (Z @ Pp @ Z.T)[0, 0] + r
        K = Pp @ Z.T / S
        Pn = (np.eye(2) - K @ Z) @ Pp
        if np.abs(Pn - P).max() < 1e-12:
            break
        P = Pn
    return float(K[0, 0]), float(K[1, 0])


class KalmanTrend(_Series):
    name = "kalman"
    description = "local linear trend Kalman filter on log price; noise ratios from the grid (chosen on validation)"

    def fit(self, X, y, train, val):
        self._fit_slice(train, X)
        q = float(self.params.get("q_slope", 1e-4))
        self.K = _steady_gain(float(self.params.get("q_level", 0.1)), q)
        return self

    def _raw(self, X, rows):
        lp = np.cumsum(self._w(X))
        slope = _llt_filter(lp, *self.K)
        return (slope * self.target.h)[rows]


class StateSpaceTrend(_Series):
    name = "state_space"
    description = "statsmodels UnobservedComponents local linear trend fit by MLE on recent training bars"

    def fit(self, X, y, train, val):
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        idx = self._fit_slice(train, X)[-8000:]
        lp = np.cumsum(self._w(X))
        self.res = UnobservedComponents(lp[idx], level="lltrend").fit(disp=False, maxiter=100)
        return self

    def _raw(self, X, rows):
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        lp = np.cumsum(self._w(X))
        res = UnobservedComponents(lp, level="lltrend").filter(self.res.params)
        slope = res.filter_results.filtered_state[1]
        return (slope * self.target.h)[rows]
