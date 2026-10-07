"""Metrics on top of the project scorecard (alpha.research.scorecard): one Card per run plus breakdowns by fold,
regime, side, month, week and year, and the composite display score (never used for selection)."""

import numpy as np
import pandas as pd

from alpha.research.scorecard import Card, _pf, daily_pnl, max_drawdown, score
from alpha.tournament.analytics.regimes import DIMENSIONS

EQUITY = 30_000.0


def card(name: str, trades: pd.DataFrame, complexity: int) -> Card:
    t = trades.copy()
    if "regime" not in t and "trend" in t:
        t["regime"] = t["trend"]
    return score(name, t, n_trials=1, complexity=complexity)


def breakdown(trades: pd.DataFrame, key) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["trades", "net", "gross", "pf", "win_rate", "max_dd", "net_usd"])
    pnl = trades["net_bps"] / 1e4
    g = pnl.groupby(key)

    def dd(x):
        return max_drawdown(x.groupby(x.index.floor("D")).sum())

    out = pd.DataFrame({"trades": g.size(), "net": g.sum(),
                        "gross": (trades["gross_bps"] / 1e4).groupby(key).sum(),
                        "pf": g.apply(_pf), "win_rate": g.apply(lambda x: float((x > 0).mean())),
                        "max_dd": g.apply(dd)})
    out["net_usd"] = out["net"] * EQUITY
    return out


def all_breakdowns(trades: pd.DataFrame, fold_of: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {}
    idx = pd.DatetimeIndex(trades.index)
    out = {"side": breakdown(trades, trades["side"].map({1: "long", -1: "short"}).to_numpy()),
           "month": breakdown(trades, idx.strftime("%Y-%m")),
           "week": breakdown(trades, idx.strftime("%G-W%V")),
           "year": breakdown(trades, idx.year.astype(str))}
    if "fold" in trades:
        out["fold"] = breakdown(trades, trades["fold"].astype(str).to_numpy())
    for d in DIMENSIONS:
        if d in trades:
            out[f"regime_{d}"] = breakdown(trades, trades[d].fillna("unknown").to_numpy())
    return out


def equity_curve(bar_pnl: pd.Series) -> pd.DataFrame:
    eq = bar_pnl.cumsum()
    return pd.DataFrame({"equity": eq, "drawdown": eq - eq.cummax().clip(lower=0)})


def calmar(c: Card, days: int) -> float:
    ann = c.net / max(days, 1) * 365
    return float(ann / c.max_dd) if c.max_dd > 0 else np.nan


def composite(rows: pd.DataFrame) -> pd.Series:
    """Display-only 0-100 score: percentile ranks within the leaderboard, weighted 35/20/15/10/8/5/5/2."""
    if rows.empty:
        return pd.Series(dtype=float)
    r = lambda s, asc=True: s.rank(pct=True, ascending=asc).fillna(0)
    s = (0.35 * r(rows["net"]) + 0.20 * r(rows.get("folds_positive", pd.Series(0, index=rows.index)))
         + 0.15 * r(rows["pf"]) + 0.10 * r(rows["max_dd"], asc=False) + 0.08 * r(rows["sharpe"])
         + 0.05 * r(rows["trades"]) + 0.05 * r(rows.get("regimes_positive", pd.Series(0, index=rows.index)))
         + 0.02 * r(rows["complexity"], asc=False))
    return 100 * s


def summary_row(c: Card, trades: pd.DataFrame, oos_days: int) -> dict:
    s = c.summary()
    s.update({"net_usd": c.net * EQUITY, "fees": float(trades["fee_bps"].sum() / 1e4) if len(trades) else 0.0,
              "slippage": float(trades["slip_bps"].sum() / 1e4) if len(trades) else 0.0,
              "calmar": calmar(c, oos_days), "win_rate": c.hit})
    if len(trades) and "fold" in trades:
        f = (trades["net_bps"]).groupby(trades["fold"]).sum()
        s["folds_positive"] = float((f > 0).mean())
        s["folds_traded"] = int(len(f))
    if len(trades) and "trend" in trades:
        reg = []
        for d in DIMENSIONS:
            g = trades.groupby(d)["net_bps"]
            ok = g.size() >= 30
            reg += list((g.sum()[ok] > 0).to_numpy())
        s["regimes_positive"] = float(np.mean(reg)) if reg else np.nan
    if len(trades):
        m = (trades["net_bps"] / 1e4).groupby(pd.DatetimeIndex(trades.index).strftime("%Y-%m")).sum()
        pos = m.clip(lower=0)
        s["max_month_share"] = float(pos.max() / pos.sum()) if pos.sum() > 0 else 1.0
    return s
