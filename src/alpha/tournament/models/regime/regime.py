"""Regime models. All parameters are fit on training rows; regime probabilities for any later bar come from
forward filtering only (P(state_t | x_1..x_t)), never forward-backward smoothing, so they are what a live system
would have known.

Mode 1 (direct):  score_t = sum_k P(state_{t+1} = k | x_<=t) * m_k, with m_k the training mean of the
                  (vol-normalized) target in state k.
Mode 2 (features): transform() -> p_regime_k, regime (argmax), duration (bars in the current argmax regime),
                  p_transition (P(next state differs)), expected_remaining (HSMM only), plus model-specific state.

    GaussianHMM          1-D Gaussian HMM on 4-bar returns (hmmlearn EM)
    HMM                  2-D (16-bar return, log 16-bar vol), full covariance
    MultivariateHMM      7 standardized features, diagonal covariance
    MarkovSwitching      statsmodels MarkovRegression on returns (switching mean + variance), Hamilton filter
    HSMM / EDHMM         explicit-duration (residual-time) hidden semi-Markov model, own EM + forward filter;
                         non-parametric duration pmf up to D_MAX bars (EDHMM: 4 regimes, finer emissions)
    SLDS / SwitchingKalman / SwitchingSSM   switching linear dynamical system on returns: latent drift with
                         regime-specific noise, IMM (interacting multiple model) filter, parameters by maximum
                         likelihood. Variants differ in regime count and whether the AR coefficient is fitted.
"""

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp

from alpha.tournament.models.base import BaseTradingModel, train_rows

N_FIT = 40_000
D_MAX = 192  # HSMM max regime duration (2 days of 15m bars)


# ---------------------------------------------------------------- emissions / filters
def _gauss_loglik(Z: np.ndarray, means: np.ndarray, covs: np.ndarray, diag: bool) -> np.ndarray:
    n, K = len(Z), len(means)
    out = np.empty((n, K))
    for k in range(K):
        d = Z - means[k]
        if diag:
            v = covs[k]
            out[:, k] = -0.5 * (np.sum(d * d / v, axis=1) + np.sum(np.log(2 * np.pi * v)))
        else:
            C = covs[k]
            Ci = np.linalg.inv(C)
            _, logdet = np.linalg.slogdet(C)
            out[:, k] = -0.5 * (np.einsum("ij,jk,ik->i", d, Ci, d) + logdet + Z.shape[1] * np.log(2 * np.pi))
    return out


@njit(cache=True)
def _hmm_forward(logb, logA, logpi):
    n, K = logb.shape
    out = np.empty((n, K))
    la = logpi + logb[0]
    m = la.max()
    la = la - (m + np.log(np.exp(la - m).sum()))
    out[0] = la
    for t in range(1, n):
        new = np.empty(K)
        for k in range(K):
            v = la + logA[:, k]
            mv = v.max()
            new[k] = mv + np.log(np.exp(v - mv).sum()) + logb[t, k]
        m = new.max()
        la = new - (m + np.log(np.exp(new - m).sum()))
        out[t] = la
    return np.exp(out)


def _run_length(state: np.ndarray) -> np.ndarray:
    n = len(state)
    out = np.ones(n)
    for t in range(1, n):
        out[t] = out[t - 1] + 1 if state[t] == state[t - 1] else 1
    return out


class _Regime(BaseTradingModel):
    family = "regime"
    complexity = 2
    input_kind = "series"
    target_kinds = ("reg",)
    K = 3
    feature_cols: list[str] = ["ret_16"]

    # -- feature matrix with training standardization --
    def _Z(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.feature_cols].to_numpy(np.float64).copy()
        for j, c in enumerate(self.feature_cols):
            if c.startswith("rvol"):
                A[:, j] = np.log(np.maximum(A[:, j], 1e-6))
        A = np.where(np.isfinite(A), A, self.mu_ if hasattr(self, "mu_") else 0.0)
        if not hasattr(self, "mu_"):
            return A
        return np.clip((A - self.mu_) / self.sd_, -6, 6)

    def _standardize_fit(self, X, idx):
        A = self._Z(X)[idx]
        self.mu_ = np.nanmean(A, axis=0)
        self.sd_ = np.nanstd(A, axis=0) + 1e-9

    def fit(self, X, y, train, val):
        idx = train_rows(train, y)[-N_FIT:]
        self._standardize_fit(X, idx)
        self._fit_regime(self._Z(X)[idx])
        P = self.filter_probs(X)
        Pn = self.next_probs(P)
        yy = y.to_numpy()[idx]
        w = Pn[idx]
        self.m_ = (w * yy[:, None]).sum(axis=0) / (w.sum(axis=0) + 1e-9)
        return self

    def _raw(self, X, rows):
        Pn = self.next_probs(self.filter_probs(X))
        return (Pn @ self.m_)[rows]

    def next_probs(self, P):
        return P @ self.A_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        P = self.filter_probs(X)
        Pn = self.next_probs(P)
        st = P.argmax(axis=1)
        out = {f"{self.name}_p{k}": P[:, k] for k in range(P.shape[1])}
        out[f"{self.name}_regime"] = st.astype(float)
        out[f"{self.name}_duration"] = np.log1p(_run_length(st))
        out[f"{self.name}_p_transition"] = 1 - Pn[np.arange(len(st)), st]
        out[f"{self.name}_score"] = Pn @ self.m_ if hasattr(self, "m_") else 0.0
        return pd.DataFrame(out, index=X.index)

    def _fit_regime(self, Z):
        raise NotImplementedError

    def filter_probs(self, X) -> np.ndarray:
        raise NotImplementedError


class _HMM(_Regime):
    diag = False

    def _fit_regime(self, Z):
        from hmmlearn.hmm import GaussianHMM

        K = int(self.params.get("K", self.K))
        m = GaussianHMM(n_components=K, covariance_type="diag" if self.diag else "full", n_iter=60,
                        random_state=self.seed, min_covar=1e-3)
        m.fit(Z)
        self.A_ = m.transmat_
        self.pi_ = m.startprob_
        self.means_ = m.means_
        self.covs_ = np.array([np.diag(c) for c in m.covars_]) if self.diag else m.covars_

    def filter_probs(self, X):
        logb = _gauss_loglik(self._Z(X), self.means_, self.covs_, self.diag)
        return _hmm_forward(logb, np.log(self.A_ + 1e-300), np.log(self.pi_ + 1e-300))


class GaussianHMMDirect(_HMM):
    name = "gaussian_hmm"
    feature_cols = ["ret_4"]


class HMMDirect(_HMM):
    name = "hmm"
    feature_cols = ["ret_16", "rvol_16"]


class MultiHMMDirect(_HMM):
    name = "multivariate_hmm"
    diag = True
    K = 4
    feature_cols = ["ret_4", "ret_16", "rvol_16", "rvol_ratio_16_96", "flow_60", "vol_z_96", "btc_ret_16"]


class MSDirect(_Regime):
    name = "markov_switching"
    feature_cols = ["ret_4"]
    K = 2

    def _fit_regime(self, Z):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        K = int(self.params.get("K", self.K))
        z = Z[-15_000:, 0]
        self.ms_kw = dict(k_regimes=K, trend="c", switching_variance=True)
        self.res = MarkovRegression(z, **self.ms_kw).fit(disp=False, maxiter=200, search_reps=3)
        P = np.asarray(self.res.model.regime_transition_matrix(self.res.params))[:, :, 0]  # P[i, j] = P(i | j)
        self.A_ = P.T

    def filter_probs(self, X):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        z = self._Z(X)[:, 0]
        r = MarkovRegression(z, **self.ms_kw).filter(self.res.params)
        return np.asarray(r.filtered_marginal_probabilities)


# ---------------------------------------------------------------- HSMM (residual-time explicit duration)
@njit(cache=True)
def _hsmm_forward(b, A, P, pi):
    """b: (n, K) emission likelihoods (scaled per row), A: (K, K) zero-diagonal, P: (K, D) duration pmf.
    Returns alpha (n, K, D) normalized per t (filtered P(k, remaining d | x_<=t)) and the log scales."""
    n, K = b.shape
    D = P.shape[1]
    alpha = np.zeros((n, K, D))
    logc = np.zeros(n)
    for k in range(K):
        for d in range(D):
            alpha[0, k, d] = pi[k] * P[k, d] * b[0, k]
    s = alpha[0].sum()
    alpha[0] /= s
    logc[0] = np.log(s)
    for t in range(1, n):
        end = alpha[t - 1, :, 0]
        for k in range(K):
            enter = 0.0
            for j in range(K):
                enter += end[j] * A[j, k]
            for d in range(D):
                stay = alpha[t - 1, k, d + 1] if d + 1 < D else 0.0
                alpha[t, k, d] = (stay + enter * P[k, d]) * b[t, k]
        s = alpha[t].sum()
        if s <= 0:
            s = 1e-300
        alpha[t] /= s
        logc[t] = np.log(s)
    return alpha, logc


@njit(cache=True)
def _hsmm_filter_summary(b, A, P, pi):
    """Forward filter without storing alpha: per bar P(k), P(k, remaining = 1), E[remaining | k]."""
    n, K = b.shape
    D = P.shape[1]
    a = np.zeros((K, D))
    pk = np.zeros((n, K))
    pend = np.zeros((n, K))
    erem = np.zeros((n, K))
    for t in range(n):
        new = np.zeros((K, D))
        if t == 0:
            for k in range(K):
                for d in range(D):
                    new[k, d] = pi[k] * P[k, d] * b[0, k]
        else:
            for k in range(K):
                enter = 0.0
                for j in range(K):
                    enter += a[j, 0] * A[j, k]
                for d in range(D):
                    stay = a[k, d + 1] if d + 1 < D else 0.0
                    new[k, d] = (stay + enter * P[k, d]) * b[t, k]
        s = new.sum()
        if s <= 0:
            s = 1e-300
        a = new / s
        for k in range(K):
            tot = 0.0
            ex = 0.0
            for d in range(D):
                tot += a[k, d]
                ex += a[k, d] * (d + 1)
            pk[t, k] = tot
            pend[t, k] = a[k, 0]
            erem[t, k] = ex / tot if tot > 0 else 0.0
    return pk, pend, erem


@njit(cache=True)
def _hsmm_backward_stats(b, A, P, alpha, logc):
    """Backward pass; returns regime posteriors gamma (n, K), transition counts (K, K), duration counts (K, D)."""
    n, K = b.shape
    D = P.shape[1]
    beta = np.ones((K, D))
    gamma = np.zeros((n, K))
    trans = np.zeros((K, K))
    dur = np.zeros((K, D))
    for k in range(K):
        gamma[n - 1, k] = alpha[n - 1, k].sum()
    for t in range(n - 2, -1, -1):
        c = np.exp(logc[t + 1])
        # u[j] = sum_d' P[j, d'] b[t+1, j] beta_{t+1}(j, d')
        u = np.zeros(K)
        for j in range(K):
            for d in range(D):
                u[j] += P[j, d] * b[t + 1, j] * beta[j, d]
        new = np.zeros((K, D))
        for k in range(K):
            for d in range(1, D):
                new[k, d] = b[t + 1, k] * beta[k, d - 1] / c
            s = 0.0
            for j in range(K):
                s += A[k, j] * u[j]
            new[k, 0] = s / c
        # expected segment boundaries t -> t+1
        for k in range(K):
            ak = alpha[t, k, 0]
            if ak <= 0:
                continue
            for j in range(K):
                if A[k, j] <= 0:
                    continue
                for d in range(D):
                    v = ak * A[k, j] * P[j, d] * b[t + 1, j] * beta[j, d] / c
                    trans[k, j] += v
                    dur[j, d] += v
        beta = new
        for k in range(K):
            g = 0.0
            for d in range(D):
                g += alpha[t, k, d] * beta[k, d]
            gamma[t, k] = g
    return gamma, trans, dur


class HSMMDirect(_Regime):
    name = "hsmm"
    feature_cols = ["ret_16", "rvol_16", "flow_60"]
    K = 3
    n_iter = 12

    def _emis(self, Z):
        ll = _gauss_loglik(Z, self.means_, self.vars_, True)
        return np.exp(ll - ll.max(axis=1, keepdims=True))

    def _fit_regime(self, Z):
        K = int(self.params.get("K", self.K))
        D = int(self.params.get("D", D_MAX))
        Z = Z[-int(self.params.get("n_fit", 20_000)):]
        rng = np.random.default_rng(self.seed)
        # init: k-means-ish on quantiles of the first feature
        q = np.quantile(Z[:, 0], np.linspace(0, 1, K + 1))
        lab = np.clip(np.searchsorted(q, Z[:, 0]) - 1, 0, K - 1)
        self.means_ = np.array([Z[lab == k].mean(axis=0) if (lab == k).any() else rng.standard_normal(Z.shape[1])
                                for k in range(K)])
        self.vars_ = np.array([Z[lab == k].var(axis=0) + 0.05 if (lab == k).sum() > 2 else np.ones(Z.shape[1])
                               for k in range(K)])
        A = np.full((K, K), 1.0 / (K - 1))
        np.fill_diagonal(A, 0.0)
        mean_d = float(self.params.get("init_duration", 32))
        d = np.arange(1, D + 1)
        P = np.tile(np.exp(-d / mean_d), (K, 1))
        P /= P.sum(axis=1, keepdims=True)
        pi = np.full(K, 1.0 / K)
        for _ in range(self.n_iter):
            b = self._emis(Z)
            alpha, logc = _hsmm_forward(b, A, P, pi)
            gamma, trans, dur = _hsmm_backward_stats(b, A, P, alpha, logc)
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            A = trans + 1e-6
            np.fill_diagonal(A, 0.0)
            A /= A.sum(axis=1, keepdims=True)
            P = dur + 1e-4
            # light smoothing of the duration pmf (5-bar moving average) keeps EM from overfitting spikes
            ker = np.ones(5) / 5
            P = np.array([np.convolve(p, ker, mode="same") for p in P]) + 1e-8
            P /= P.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0) + 1e-9
            self.means_ = (gamma.T @ Z) / w[:, None]
            self.vars_ = np.maximum((gamma.T @ (Z**2)) / w[:, None] - self.means_**2, 0.02)
            pi = gamma[0] + 1e-6
            pi /= pi.sum()
        self.A_hsmm_, self.P_, self.pi_ = A, P, pi

    def filter_probs(self, X):
        b = self._emis(self._Z(X))
        pk, pend, erem = _hsmm_filter_summary(b, self.A_hsmm_, self.P_, self.pi_)
        self._last = (pk, pend, erem)
        return pk

    def next_probs(self, P):
        last = getattr(self, "_last", None)
        if last is None or len(last[0]) != len(P):
            return P
        pk, pend, _ = last
        return (pk - pend) + pend @ self.A_hsmm_

    def transform(self, X):
        out = super().transform(X)
        pk, pend, erem = self._last
        st = pk.argmax(axis=1)
        out[f"{self.name}_expected_remaining"] = np.log1p(erem[np.arange(len(st)), st])
        out[f"{self.name}_p_end"] = pend.sum(axis=1)
        return out


class EDHMMDirect(HSMMDirect):
    name = "explicit_duration_hmm"
    feature_cols = ["ret_4", "ret_16", "rvol_16", "rvol_ratio_16_96", "flow_60"]
    K = 4


# ---------------------------------------------------------------- SLDS / switching Kalman (IMM)
@njit(cache=True)
def _imm(y, phi, q, r, Pst):
    """IMM filter. Model per regime k: x_t = phi_k x_{t-1} + w (var q_k); y_t = x_t + v (var r_k).
    Pst: (K, K) regime transition. Returns (mu (n, K) filtered regime probs, xhat (n,), loglik)."""
    n = len(y)
    K = len(q)
    mu = np.full(K, 1.0 / K)
    x = np.zeros(K)
    P = np.ones(K) * 10.0
    MU = np.zeros((n, K))
    XH = np.zeros(n)
    ll = 0.0
    for t in range(n):
        # mixing
        c = Pst.T @ mu  # c[j] = sum_i P[i, j] mu[i]
        x0 = np.zeros(K)
        P0 = np.zeros(K)
        for j in range(K):
            if c[j] <= 0:
                c[j] = 1e-300
            for i in range(K):
                w = Pst[i, j] * mu[i] / c[j]
                x0[j] += w * x[i]
            for i in range(K):
                w = Pst[i, j] * mu[i] / c[j]
                P0[j] += w * (P[i] + (x[i] - x0[j]) ** 2)
        lik = np.zeros(K)
        for j in range(K):
            xp = phi[j] * x0[j]
            Pp = phi[j] ** 2 * P0[j] + q[j]
            S = Pp + r[j]
            e = y[t] - xp
            K_ = Pp / S
            x[j] = xp + K_ * e
            P[j] = (1 - K_) * Pp
            lik[j] = np.exp(-0.5 * e * e / S) / np.sqrt(2 * np.pi * S)
        tot = 0.0
        for j in range(K):
            mu[j] = c[j] * lik[j]
            tot += mu[j]
        if tot <= 0:
            tot = 1e-300
        mu /= tot
        ll += np.log(tot)
        MU[t] = mu
        XH[t] = (mu * x).sum()
    return MU, XH, ll


class SLDSDirect(_Regime):
    name = "slds"
    output_units = "bps"
    K = 2
    fit_phi = False

    def _y(self, X):
        y = np.nan_to_num(X["ret_1"].to_numpy(np.float64) * 1e4)
        return np.clip(y, -self.clip_, self.clip_)

    def _unpack(self, th, K):
        q = np.exp(th[:K])
        r = np.exp(th[K:2 * K])
        stay = 1 / (1 + np.exp(-th[2 * K]))
        Pst = np.full((K, K), (1 - stay) / (K - 1))
        np.fill_diagonal(Pst, stay)
        phi = np.full(K, 1 / (1 + np.exp(-th[2 * K + 1])) if self.fit_phi else 0.98)
        return phi, q, r, Pst

    def fit(self, X, y, train, val):
        K = int(self.params.get("K", self.K))
        idx = train_rows(train, y)[-20_000:]
        raw = np.nan_to_num(X["ret_1"].to_numpy(np.float64) * 1e4)
        self.clip_ = 8 * float(np.std(raw[idx]))
        yy = self._y(X)[idx]
        v = float(np.var(yy))
        th0 = np.concatenate([np.log(v * np.geomspace(1e-4, 1e-2, K)), np.log(v * np.linspace(0.5, 1.5, K)), [4.0],
                              [3.0] if self.fit_phi else []])
        f = lambda th: -_imm(yy, *self._unpack(th, K))[2]
        self.th_ = minimize(f, th0, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-2}).x
        self.K_ = K
        self.A_ = self._unpack(self.th_, K)[3]
        MU, XH, _ = _imm(self._y(X), *self._unpack(self.th_, K))
        self._cache = (len(X), MU, XH)
        return self

    def _run(self, X):
        if getattr(self, "_cache", (None,))[0] == len(X):
            return self._cache[1], self._cache[2]
        MU, XH, _ = _imm(self._y(X), *self._unpack(self.th_, self.K_))
        self._cache = (len(X), MU, XH)
        return MU, XH

    def filter_probs(self, X):
        return self._run(X)[0]

    def _raw(self, X, rows):
        _, XH = self._run(X)
        phi = self._unpack(self.th_, self.K_)[0].mean()
        h = self.target.h
        return (XH * phi * (1 - phi**h) / (1 - phi))[rows]

    def transform(self, X):
        MU, XH = self._run(X)
        st = MU.argmax(axis=1)
        Pn = MU @ self.A_
        out = {f"{self.name}_p{k}": MU[:, k] for k in range(MU.shape[1])}
        out[f"{self.name}_regime"] = st.astype(float)
        out[f"{self.name}_duration"] = np.log1p(_run_length(st))
        out[f"{self.name}_p_transition"] = 1 - Pn[np.arange(len(st)), st]
        out[f"{self.name}_drift"] = XH
        return pd.DataFrame(out, index=X.index)


class SwitchingKalmanDirect(SLDSDirect):
    name = "switching_kalman"
    K = 3


class SwitchingSSMDirect(SLDSDirect):
    name = "switching_state_space"
    fit_phi = True
