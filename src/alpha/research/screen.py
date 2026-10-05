"""Phase 2: screen every signal family on DEV only.

    uv run python -m alpha.research.screen

For each (signal, rebalance horizon) it reports gross edge per unit traded vs all-in cost per unit traded, net
Sharpe, Newey-West t, drawdown, years / tokens positive. A family survives only if

    edge >= 1.5 x (execution cost + funding)  AND  net NW-t >= 2  AND  >= 60% of DEV years positive

with taker execution. The number of (signal, horizon) pairs tried is recorded as the trial count for the deflated
Sharpe ratio of anything built from the survivors later.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.exec.costs import CostModel
from alpha.research import signals as sig
from alpha.research.panel import cached_panel, wide
from alpha.research.portfolio_sim import simulate, summarize, to_weights
from alpha.research.splits import DEV

HORIZONS = (1, 4, 12, 24, 72)
POSITION_NOTIONAL = 15_000.0  # $50k equity x up to 3x over ~10 positions
OUT = Path("data/experiments")
FALLBACK_SPREAD_BPS = 1.0


def symbol_costs(conn, panel: pd.DataFrame, cm: CostModel = CostModel()) -> pd.DataFrame:
    """Per-symbol taker / maker cost per side (bps). Spread: 2 x live median (history was wider than today);
    depth: 5th percentile of DEV-period +-1% depth (2023-24), or the thinnest coin's when missing."""
    live = dict(conn.execute("""SELECT split_part(symbol, '.', 1),
            percentile_cont(0.5) WITHIN GROUP (ORDER BY (best_ask - best_bid) / ((best_ask + best_bid) / 2) * 1e4)
            FROM book_tick GROUP BY 1""").fetchall())
    dev = panel[DEV.mask(panel.index.get_level_values("time"))]
    depth = dev["depth_1pct"].groupby(level="symbol").quantile(0.05)
    floor = depth.min()
    ret = dev["ret"].groupby(level="symbol").std() * 1e4 / np.sqrt(3600)  # bps per sqrt(second)
    rows = {}
    for s in panel.index.get_level_values("symbol").unique():
        spread = 2 * live.get(s, FALLBACK_SPREAD_BPS)
        d = depth.get(s, np.nan)
        d = floor if not np.isfinite(d) else d
        rows[s] = {"spread_bps": spread, "depth_p05": d,
                   "taker_bps": float(cm.taker_bps(POSITION_NOTIONAL, spread, d, ret.get(s, 1.0))),
                   "maker_bps": float(cm.maker_bps(spread))}
    return pd.DataFrame(rows).T


def run(panel: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    dev = panel[DEV.mask(panel.index.get_level_values("time"))]
    scores = sig.compute(panel)  # computed on the full panel (each row uses only past data), then cut to DEV
    t_dev = wide(dev, "ret").index
    ret, fund = wide(panel, "ret").loc[t_dev], wide(panel, "funding_rate").loc[t_dev]
    el = wide(panel, "eligible").fillna(False).astype(bool).loc[t_dev]
    vol = wide(panel, "ret").rolling(sig.VOL_WINDOW, min_periods=48).std().loc[t_dev]
    rows = []
    for name, score in scores.items():
        score = score.reindex(t_dev)
        for h in HORIZONS:
            w = to_weights(score, vol, el, sig.kind(name), every=h)
            res = simulate(w, ret, fund, costs["taker_bps"])
            s = summarize(res, h)
            all_in = s["cost_bps"] + max(s["funding_bps"], 0.0)  # funding received is not netted against costs
            s.update(signal=name, kind=sig.kind(name), horizon_h=h,
                     edge_to_cost=s["edge_bps"] / all_in if all_in > 0 else np.nan,
                     maker_cost_bps=float((w.diff().abs() * costs["maker_bps"].reindex(w.columns).to_numpy() / 1e4)
                                          .sum().sum() / max(res.hourly["turnover"].sum(), 1e-12) * 1e4))
            s["passes"] = bool(s["edge_bps"] >= 1.5 * all_in and s["t_nw"] >= 2 and s["years_pos"] >= 0.6)
            rows.append(s)
    return pd.DataFrame(rows)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        panel = cached_panel(conn, s.symbols)
        costs = symbol_costs(conn, panel)
    pd.set_option("display.width", 250)
    print("cost per side (bps):\n", costs.round(2).to_string())
    df = run(panel, costs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "phase2_screen.csv", index=False)
    (OUT / "phase2_trials.json").write_text(json.dumps({"trials": len(df)}))
    cols = ["signal", "horizon_h", "edge_bps", "cost_bps", "funding_bps", "maker_cost_bps", "edge_to_cost",
            "turnover_per_day", "net_ann", "sharpe", "t_nw", "max_dd", "years_pos", "tokens_pos", "passes"]
    print(df[cols].sort_values("edge_to_cost", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n{len(df)} (signal, horizon) trials; {int(df['passes'].sum())} pass the Phase 2 gate")


if __name__ == "__main__":
    main()
