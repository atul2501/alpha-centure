"""Statistical significance with multiple-testing awareness.

    stationary bootstrap (Politis-Romano)   CIs for total net P&L and annualized Sharpe of daily P&L
    deflated Sharpe ratio                   alpha.research.scorecard.deflated_sharpe with N = experiments run
    PBO                                     alpha.research.scorecard.pbo_cscv over all experiments' daily P&L
    Hansen SPA / White Reality Check        is the best of the whole tournament better than FLAT (zero)?
"""

import numpy as np
import pandas as pd

from alpha.research.scorecard import deflated_sharpe, pbo_cscv

MEAN_BLOCK_DAYS = 10


def stationary_indices(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    jump = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_ci(daily: pd.Series, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    x = daily.to_numpy(float)
    if len(x) < 30:
        return {}
    rng = np.random.default_rng(seed)
    idx = stationary_indices(len(x), n_boot, MEAN_BLOCK_DAYS, rng)
    b = x[idx]
    tot = b.sum(axis=1)
    sd = b.std(axis=1, ddof=1)
    sh = np.where(sd > 0, b.mean(axis=1) / sd * np.sqrt(365), np.nan)
    lo, hi = alpha / 2, 1 - alpha / 2
    return {"net_ci_lo": float(np.quantile(tot, lo)), "net_ci_hi": float(np.quantile(tot, hi)),
            "sharpe_ci_lo": float(np.nanquantile(sh, lo)), "sharpe_ci_hi": float(np.nanquantile(sh, hi)),
            "p_net_le_0": float((tot <= 0).mean())}


def dsr(daily: pd.Series, n_trials: int) -> float:
    return deflated_sharpe(daily, n_trials)


def pbo(matrix: pd.DataFrame, n_blocks: int = 10) -> float:
    return pbo_cscv(matrix, n_blocks)


def spa(matrix: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Hansen (2005) SPA (consistent) and White (2000) Reality Check p-values for H0: no model beats a zero
    benchmark. matrix: days x models of daily net P&L (benchmark FLAT = 0, so losses are the model's own)."""
    d = matrix.fillna(0.0).to_numpy(float)
    n, m = d.shape
    if n < 30 or m == 0:
        return {"spa_p": np.nan, "rc_p": np.nan}
    rng = np.random.default_rng(seed)
    mean = d.mean(axis=0)
    idx = stationary_indices(n, n_boot, MEAN_BLOCK_DAYS, rng)
    boot_means = np.stack([d[i].mean(axis=0) for i in idx])          # n_boot x m
    omega = np.sqrt(n) * boot_means.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    t_stat = np.max(np.sqrt(n) * mean / omega)
    rc_stat = np.max(np.sqrt(n) * mean)
    # SPA consistent: recentre only models that are not too poor
    thr = -np.sqrt(2 * np.log(np.log(n))) * omega / np.sqrt(n)
    mu_c = np.where(mean >= thr, mean, 0.0)
    cent = boot_means - mu_c
    t_boot = np.max(np.sqrt(n) * cent / omega, axis=1)
    rc_boot = np.max(np.sqrt(n) * (boot_means - mean), axis=1)
    return {"spa_p": float((t_boot >= t_stat).mean()), "rc_p": float((rc_boot >= rc_stat).mean()),
            "spa_stat": float(t_stat), "n_models": int(m), "n_days": int(n)}
