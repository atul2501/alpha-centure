"""Robustness of a finished experiment, from its stored OOS predictions (no refit):

    threshold perturbation  every fold's chosen threshold x {0.8, 0.9, 1.1, 1.2}
    hold perturbation       hold h -> {h/2, 2h} (bars, at least 1)
    execution delay         positions shifted one bar later (the fill one bar late)
Cost and slippage stress are part of every experiment run (runner variants cost_x*, slip_x*).
A candidate is 'parameter stable' when every perturbation stays net positive and keeps >= 50% of the base net.
"""

import numpy as np
import pandas as pd

from alpha.tournament.backtesting import engine, policy

TH_MULTS = (0.8, 0.9, 1.1, 1.2)


def _positions(pred: pd.DataFrame, idx: pd.DatetimeIndex, th_mult: float = 1.0, hold_mult: float = 1.0,
               abstain: dict | None = None) -> np.ndarray:
    pos = np.zeros(len(idx))
    for k, g in pred.groupby("fold"):
        if abstain and abstain.get(int(k), False):
            continue
        h = max(1, int(round(g["target_h"].iloc[0] * hold_mult)))
        p = policy.positions(g["score"].to_numpy(), float(g["threshold"].iloc[0]) * th_mult, h)
        loc = idx.get_indexer(g.index)
        ok = loc >= 0
        pos[loc[ok]] = p[ok]
    return pos


def perturb(pred: pd.DataFrame, market: engine.Market, abstain: dict) -> dict:
    """pred: index bar_ts, columns fold, score, threshold, target_h. market: sliced to the OOS index."""
    idx = market.index
    base = engine.run(_positions(pred, idx, abstain=abstain), market).bar_pnl.sum()
    out = {"base_net": float(base)}
    for m in TH_MULTS:
        out[f"th_x{m:g}"] = float(engine.run(_positions(pred, idx, th_mult=m, abstain=abstain), market).bar_pnl.sum())
    for hm in (0.5, 2.0):
        out[f"hold_x{hm:g}"] = float(engine.run(_positions(pred, idx, hold_mult=hm, abstain=abstain), market)
                                     .bar_pnl.sum())
    p = _positions(pred, idx, abstain=abstain)
    out["delay_1bar"] = float(engine.run(np.concatenate([[0.0], p[:-1]]), market).bar_pnl.sum())
    vals = [v for k, v in out.items() if k != "base_net"]
    out["param_stable"] = bool(base > 0 and all(v > 0 and v >= 0.5 * base for v in vals))
    return out
