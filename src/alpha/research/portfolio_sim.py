"""Vectorized hourly portfolio simulation: target weights -> gross, funding, cost, net P&L.

Weights W (time x symbol, fraction of equity) are decided at the close of bar t and held over bar t+1:
    gross_{t+1}   = sum_i W_{t,i} * (exp(ret_{t+1,i}) - 1)
    funding_{t+1} = sum_i W_{t,i} * funding_rate_{t+1,i}        (long pays a positive rate; > 0 = paid)
    cost_t        = sum_i |W_{t,i} - W_{t-1,i}| * cost_bps_i / 1e4
net = gross - funding - cost. Weights are only ever non-zero on eligible coins.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def to_weights(score: pd.DataFrame, vol: pd.DataFrame, eligible: pd.DataFrame, kind: str, every: int = 1,
               gross: float = 1.0) -> pd.DataFrame:
    """Score -> weights with total gross exposure `gross`. 'ts': inverse-vol risk parity on the score;
    'xs': dollar-neutral (score already demeaned). Rebalanced every `every` hours (held in between)."""
    s = score.where(eligible).astype(float)
    raw = s / vol if kind == "ts" else s
    if kind == "xs":
        raw = raw.sub(raw.mean(axis=1), axis=0)
    raw = raw.where(eligible).fillna(0.0)
    tot = raw.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(tot, axis=0).fillna(0.0) * gross
    if every > 1:
        hours = (w.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)  # unit-safe (ns or us index)
        on = np.asarray(hours % every == 0)  # rebalance hours, UTC-aligned
        w = w.copy()
        w.iloc[~on] = np.nan
        w = w.ffill().fillna(0.0)
    return w.where(eligible, 0.0)


@dataclass
class SimResult:
    hourly: pd.DataFrame          # gross, funding, cost, net, turnover
    by_symbol: pd.DataFrame       # gross, funding, cost, net per symbol

    def daily(self) -> pd.DataFrame:
        return self.hourly.groupby(self.hourly.index.floor("D")).sum()


def simulate(w: pd.DataFrame, ret: pd.DataFrame, funding: pd.DataFrame, cost_bps: pd.Series) -> SimResult:
    w = w.fillna(0.0)
    held = w.shift(1).fillna(0.0)
    simple = np.expm1(ret.reindex_like(w).fillna(0.0))
    g = held * simple
    f = held * funding.reindex_like(w).fillna(0.0)
    dw = (w - held).abs()
    c = dw * (cost_bps.reindex(w.columns).to_numpy() / 1e4)
    hourly = pd.DataFrame({"gross": g.sum(axis=1), "funding": f.sum(axis=1), "cost": c.sum(axis=1),
                           "turnover": dw.sum(axis=1)})
    hourly["net"] = hourly["gross"] - hourly["funding"] - hourly["cost"]
    by_symbol = pd.DataFrame({"gross": g.sum(), "funding": f.sum(), "cost": c.sum()})
    by_symbol["net"] = by_symbol["gross"] - by_symbol["funding"] - by_symbol["cost"]
    return SimResult(hourly, by_symbol)


def newey_west_t(x: pd.Series, lags: int) -> float:
    """t-stat of the mean with Newey-West (Bartlett) standard errors: overlapping holdings autocorrelate P&L."""
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 10:
        return np.nan
    e = v - v.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(v.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def summarize(res: SimResult, every: int) -> dict:
    h, d = res.hourly, res.daily()
    turn = h["turnover"].sum()
    years = d["net"].groupby(d.index.year).sum()
    sym = res.by_symbol["net"]
    pos = sym.clip(lower=0)
    eq = d["net"].cumsum()
    return {
        "edge_bps": h["gross"].sum() / turn * 1e4 if turn else np.nan,     # gross P&L per unit traded
        "cost_bps": h["cost"].sum() / turn * 1e4 if turn else np.nan,      # execution cost per unit traded
        "funding_bps": h["funding"].sum() / turn * 1e4 if turn else np.nan,
        "turnover_per_day": turn / max(1, len(d)),
        "gross_ann": d["gross"].mean() * 365, "net_ann": d["net"].mean() * 365,
        "sharpe": d["net"].mean() / d["net"].std() * np.sqrt(365) if d["net"].std() > 0 else np.nan,
        "t_nw": newey_west_t(d["net"], max(1, every // 24 + 1) * 5),
        "max_dd": float((eq.cummax() - eq).max()),
        "years_pos": float((years > 0).mean()), "years": {int(k): round(float(v), 3) for k, v in years.items()},
        "tokens_pos": float((sym > 0).mean()),
        "max_token_share": float(pos.max() / pos.sum()) if pos.sum() > 0 else 1.0,
    }
