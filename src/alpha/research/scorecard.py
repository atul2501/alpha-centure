"""Scorecard, acceptance gates and ranking for every candidate system.

Input: one row per closed trade (DatetimeIndex = exit time) with columns
    symbol, side (+1/-1), weight (notional / equity at entry), gross_bps, cost_bps (fees + spread + impact + latency),
    funding_bps (paid > 0), optional regime.
net_bps = gross_bps - cost_bps - funding_bps. Account P&L of a trade = weight * net_bps / 1e4.

Ranking follows the stated priority (net P&L -> profit factor -> expectancy -> max drawdown -> Sharpe/Sortino),
but only among candidates that pass every gate, and with the one-standard-error rule: candidates whose net P&L is
within one standard error of the best are treated as tied and the simplest one wins. A spectacular-but-fragile
backtest therefore loses to a slightly weaker robust one.
"""

from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649
STRESS = (1.5, 2.0)


def with_net(trades: pd.DataFrame, cost_mult: float = 1.0) -> pd.DataFrame:
    t = trades.copy()
    if "funding_bps" not in t:
        t["funding_bps"] = 0.0
    if "weight" not in t:
        t["weight"] = 1.0
    t["net_bps"] = t["gross_bps"] - cost_mult * t["cost_bps"] - t["funding_bps"]
    t["pnl"] = t["weight"] * t["net_bps"] / 1e4
    t["gross_pnl"] = t["weight"] * t["gross_bps"] / 1e4
    return t


def _pf(pnl: pd.Series) -> float:
    loss = -pnl[pnl < 0].sum()
    return float(pnl[pnl > 0].sum() / loss) if loss > 0 else (np.inf if (pnl > 0).any() else np.nan)


def _group(t: pd.DataFrame, key) -> pd.DataFrame:
    g = t.groupby(key)
    return pd.DataFrame({"trades": g.size(), "net": g["pnl"].sum(), "gross": g["gross_pnl"].sum(),
                         "pf": g["pnl"].apply(_pf), "hit": g["pnl"].apply(lambda x: (x > 0).mean())})


def daily_pnl(t: pd.DataFrame) -> pd.Series:
    if t.empty:
        return pd.Series(dtype=float)
    d = t["pnl"].groupby(t.index.floor("D")).sum()
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"), fill_value=0.0)


def max_drawdown(pnl_daily: pd.Series) -> float:
    eq = pnl_daily.cumsum()
    return float((eq.cummax().clip(lower=0) - eq).max()) if len(eq) else 0.0


def deflated_sharpe(daily: pd.Series, n_trials: int) -> float:
    """Probability that the true Sharpe > 0 after selecting the best of n_trials (Bailey & Lopez de Prado 2014).
    Uses per-day Sharpe; the cross-trial variance of Sharpe is approximated by its sampling variance."""
    x = daily.to_numpy()
    T = len(x)
    if T < 30 or x.std(ddof=1) == 0:
        return np.nan
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    var_sr = (1 + 0.5 * sr**2) / T
    n = max(1, n_trials)
    sr0 = 0.0 if n == 1 else np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e)))
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 10) -> float:
    """Probability of backtest overfitting via combinatorially symmetric cross-validation.
    returns: period x candidate P&L (e.g. daily). Share of splits where the in-sample best is below the
    out-of-sample median."""
    r = returns.dropna(how="all").fillna(0.0).to_numpy()
    if r.shape[1] < 2 or len(r) < n_blocks * 2:
        return np.nan
    blocks = np.array_split(np.arange(len(r)), n_blocks)
    sums = np.array([r[b].sum(axis=0) for b in blocks])  # block x candidate
    logits = []
    for ins in combinations(range(n_blocks), n_blocks // 2):
        is_mask = np.zeros(n_blocks, bool)
        is_mask[list(ins)] = True
        best = sums[is_mask].sum(axis=0).argmax()
        oos = sums[~is_mask].sum(axis=0)
        w = (oos < oos[best]).sum() / (len(oos) - 1)  # relative rank of the IS winner out of sample
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    return float(np.mean(np.array(logits) <= 0))


@dataclass
class Card:
    name: str
    trades: int
    net: float                 # account fraction, sum over trades
    gross: float
    cost: float                # execution costs (account fraction)
    funding: float
    net_bps: float             # mean per trade
    gross_bps: float
    cost_bps: float
    funding_bps: float         # mean per trade (paid > 0)
    net_se_bps: float
    t_stat: float
    pf: float
    expectancy: float          # mean account P&L per trade
    hit: float
    max_dd: float
    sharpe: float              # annualized, daily
    sortino: float
    trades_per_day: float
    years_positive: float
    tokens_positive: float
    max_token_share: float     # largest single-token share of total positive P&L
    stress: dict[float, float] = field(default_factory=dict)  # cost multiplier -> net
    by_symbol: pd.DataFrame | None = None
    by_side: pd.DataFrame | None = None
    by_regime: pd.DataFrame | None = None
    by_year: pd.DataFrame | None = None
    dsr: float = np.nan
    pbo: float = np.nan
    shuffled_t: float = np.nan  # t-stat of the shuffled-label control (should be ~0)
    complexity: int = 1        # 0 rules, 1 linear, 2 GBDT, 3 sequence / deep
    param_stable: bool | None = None

    def summary(self) -> dict:
        keep = ["name", "trades", "net", "gross", "cost", "funding", "net_bps", "gross_bps", "cost_bps", "funding_bps",
                "t_stat", "pf", "expectancy", "hit", "max_dd", "sharpe", "sortino", "trades_per_day", "years_positive",
                "tokens_positive", "max_token_share", "dsr", "pbo", "complexity"]
        out = {k: getattr(self, k) for k in keep}
        out.update({f"net_x{k}": v for k, v in self.stress.items()})
        return out


def score(name: str, trades: pd.DataFrame, n_trials: int = 1, complexity: int = 1) -> Card:
    t = with_net(trades)
    if t.empty:
        nan = np.nan
        return Card(name=name, trades=0, net=0.0, gross=0.0, cost=0.0, funding=0.0, net_bps=nan, gross_bps=nan,
                    cost_bps=nan, funding_bps=nan, net_se_bps=nan, t_stat=nan, pf=nan, expectancy=nan, hit=nan,
                    max_dd=0.0, sharpe=nan, sortino=nan, trades_per_day=0.0, years_positive=0.0,
                    tokens_positive=0.0, max_token_share=1.0, complexity=complexity)
    n = len(t)
    d = daily_pnl(t)
    days = max(1, len(d))
    sd = d.std(ddof=1) if len(d) > 1 else np.nan
    down = d[d < 0].std(ddof=1) if (d < 0).sum() > 1 else np.nan
    se = t["net_bps"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    by_symbol = _group(t, "symbol")
    pos = by_symbol["net"].clip(lower=0)
    by_year = _group(t, t.index.year)
    return Card(
        name=name, trades=n, net=float(t["pnl"].sum()), gross=float(t["gross_pnl"].sum()),
        cost=float((t["weight"] * t["cost_bps"] / 1e4).sum()), funding=float((t["weight"] * t["funding_bps"] / 1e4).sum()),
        net_bps=float(t["net_bps"].mean()), gross_bps=float(t["gross_bps"].mean()), cost_bps=float(t["cost_bps"].mean()),
        funding_bps=float(t["funding_bps"].mean()), net_se_bps=float(se), t_stat=float(t["net_bps"].mean() / se) if se and se > 0 else np.nan,
        pf=_pf(t["pnl"]), expectancy=float(t["pnl"].mean()), hit=float((t["pnl"] > 0).mean()),
        max_dd=max_drawdown(d),
        sharpe=float(d.mean() / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
        sortino=float(d.mean() / down * np.sqrt(365)) if down and down > 0 else np.nan,
        trades_per_day=n / days, years_positive=float((by_year["net"] > 0).mean()),
        tokens_positive=float((by_symbol["net"] > 0).mean()),
        max_token_share=float(pos.max() / pos.sum()) if pos.sum() > 0 else 1.0,
        stress={k: float(with_net(trades, k)["pnl"].sum()) for k in STRESS},
        by_symbol=by_symbol, by_side=_group(t, "side"),
        by_regime=_group(t, "regime") if "regime" in t else None, by_year=by_year,
        dsr=deflated_sharpe(d, n_trials), complexity=complexity,
    )


@dataclass(frozen=True)
class Gates:
    edge_margin: float = 1.5        # mean gross_bps >= margin * mean (cost + funding)
    min_t: float = 2.0
    stress_mult: float = 1.5        # still net positive with costs x stress_mult
    min_years_positive: float = 0.6
    min_tokens_positive: float = 0.6
    max_token_share: float = 0.4
    max_pbo: float = 0.2
    min_dsr: float = 0.95
    max_shuffled_abs_t: float = 2.0
    min_trades: int = 100


def verdict(c: Card, g: Gates = Gates()) -> tuple[bool, list[str]]:
    """(passes every gate, failed gates). Gates whose input was not computed (NaN) are reported as failures
    so a candidate cannot pass by skipping a check."""
    all_in = c.cost_bps + c.funding_bps
    checks = {
        f"trades >= {g.min_trades}": c.trades >= g.min_trades,
        f"gross edge >= {g.edge_margin}x costs": bool(c.gross_bps >= g.edge_margin * all_in),
        f"net t-stat >= {g.min_t}": c.t_stat >= g.min_t,
        f"net > 0 at costs x{g.stress_mult}": c.stress.get(g.stress_mult, -np.inf) > 0,
        f"years positive >= {g.min_years_positive:.0%}": c.years_positive >= g.min_years_positive,
        f"tokens positive >= {g.min_tokens_positive:.0%}": c.tokens_positive >= g.min_tokens_positive,
        f"no token > {g.max_token_share:.0%} of profit": c.max_token_share <= g.max_token_share,
        f"PBO <= {g.max_pbo}": c.pbo <= g.max_pbo,
        f"deflated Sharpe >= {g.min_dsr}": c.dsr >= g.min_dsr,
        f"shuffled control |t| < {g.max_shuffled_abs_t}": abs(c.shuffled_t) < g.max_shuffled_abs_t,
        "stable under +-20% parameters": c.param_stable is True,
    }
    fails = [k for k, ok in checks.items() if not ok]
    return not fails, fails


def rank(cards: list[Card], g: Gates = Gates()) -> pd.DataFrame:
    """Passing candidates first. Among them, those within one SE (account terms) of the best net P&L are tied and
    ordered by complexity, then profit factor, expectancy, drawdown, Sharpe, Sortino."""
    rows = []
    for c in cards:
        ok, fails = verdict(c, g)
        se_net = c.net_se_bps * c.trades / 1e4 if np.isfinite(c.net_se_bps) else np.inf  # SE of the sum (weight ~1)
        rows.append({**c.summary(), "passes": ok, "failed": "; ".join(fails), "_se": se_net})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    passing = df[df["passes"]]
    tied = pd.Series(False, index=df.index)
    if not passing.empty:
        best = passing.loc[passing["net"].idxmax()]
        tied[passing.index] = passing["net"] >= best["net"] - best["_se"]
    df["tier"] = np.where(tied, 0, np.where(df["passes"], 1, 2))
    df["_dd"] = df["max_dd"]
    df = df.sort_values(["tier", "complexity", "net", "pf", "expectancy", "_dd", "sharpe", "sortino"],
                        ascending=[True, True, False, False, False, True, False, False])
    # outside the tie, complexity must not outrank P&L
    head, rest = df[df["tier"] == 0], df[df["tier"] > 0].sort_values(["tier", "net"], ascending=[True, False])
    return pd.concat([head, rest]).drop(columns=["_se", "_dd"]).reset_index(drop=True)


def n_combos(n_blocks: int) -> int:
    return comb(n_blocks, n_blocks // 2)
