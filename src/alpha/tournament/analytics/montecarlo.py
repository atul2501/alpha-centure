"""Monte Carlo on a candidate's DEV-OOS trades. Each simulation draws, independently:

    missed trades      drop a U(5%, 20%) random share of trades
    cost perturbation  fees x U(0.9, 1.5)
    slippage           slippage x lognormal(0, 0.35)
    return noise       gross + N(0, 10% of the trades' gross std)
    reshuffle          random trade order (path-dependent drawdown / ruin)

Execution delay is deterministic and is run separately (positions shifted one bar, see robustness.delay_variant).
Ruin = the cumulative P&L reaches -30% of equity (the live kill switch).
"""

import numpy as np
import pandas as pd

RUIN = -0.30


def simulate(trades: pd.DataFrame, n: int = 5000, seed: int = 0) -> dict:
    if trades.empty or len(trades) < 20:
        return {}
    rng = np.random.default_rng(seed)
    g = trades["gross_bps"].to_numpy(float)
    fee = trades["fee_bps"].to_numpy(float)
    slip = trades["slip_bps"].to_numpy(float)
    fund = trades["funding_bps"].to_numpy(float)
    m = len(g)
    noise_sd = 0.1 * g.std()
    tot = np.empty(n)
    dd = np.empty(n)
    ruin = np.empty(n, bool)
    for i in range(n):
        keep = rng.random(m) >= rng.uniform(0.05, 0.20)
        net = (g + rng.normal(0, noise_sd, m) - fee * rng.uniform(0.9, 1.5)
               - slip * rng.lognormal(0, 0.35) - fund)[keep] / 1e4
        net = rng.permutation(net)
        eq = np.cumsum(net)
        tot[i] = eq[-1] if len(eq) else 0.0
        peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))
        dd[i] = (peak[1:] - eq).max() if len(eq) else 0.0
        ruin[i] = (eq <= RUIN).any() if len(eq) else False
    return {"mc_p_profit": float((tot > 0).mean()), "mc_p_ruin": float(ruin.mean()),
            "mc_expected_dd": float(dd.mean()), "mc_worst_dd": float(dd.max()), "mc_dd_p95": float(np.quantile(dd, 0.95)),
            "mc_pnl_p05": float(np.quantile(tot, 0.05)), "mc_pnl_p50": float(np.median(tot)),
            "mc_pnl_p95": float(np.quantile(tot, 0.95)), "mc_n": n}
