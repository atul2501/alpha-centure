"""Held-out evaluation of the pre-registered finalists (data/experiments/valid_a_preregistration.json).

    uv run python -m alpha.research.validate --split VALID-A
    uv run python -m alpha.research.validate --split VALID-B --finalists P3_xs_ts_band_maker

Each (finalist, split) passes through the lockbox once. Data after the split's end is never loaded. A 90-day
warm-up before the split start feeds the trailing vol estimate; P&L is counted from the split start only.
The ridge book keeps retraining quarterly on data known at each fold, exactly as it would live.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.research import signals as sig
from alpha.research.models import feature_frame, target, walk_forward
from alpha.research.panel import cached_panel
from alpha.research.phase4 import CONFIGS, card, run_config
from alpha.research.screen import symbol_costs
from alpha.research.splits import SPLITS, open_lockbox

OUT = Path("data/experiments")
WARMUP = pd.Timedelta(days=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["VALID-A", "VALID-B"])
    ap.add_argument("--finalists", default=None)
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    prereg = json.loads((OUT / "valid_a_preregistration.json").read_text())
    names = a.finalists.split(",") if a.finalists else prereg["finalists"]
    split = SPLITS[a.split]
    end = split.end or pd.Timestamp.now(tz="UTC").floor("h")
    for n in names:
        open_lockbox(n, a.split)  # refuses a second look
    s = get_settings()
    dev_rank = pd.read_csv(OUT / "phase4_rank.csv").set_index("name")
    cfgs = {c.name: c for c in CONFIGS}
    with psycopg.connect(s.database_url) as conn:
        panel = cached_panel(conn, s.symbols)
        costs = symbol_costs(conn, panel)
        panel = panel[panel.index.get_level_values("time") < end]
        m1 = None
        if any("m1" in cfgs[n].books for n in names):
            scores = sig.compute(panel)
            X, vol = feature_frame(panel, scores)
            first_fold = (split.start - WARMUP).to_period("Q").start_time.tz_localize("UTC")
            m1 = walk_forward(conn, panel, X, target(panel, vol, 72), 72, "M1", start=first_fold, end=end)
    rows = []
    for n in names:
        cfg = cfgs[n]
        d, info = run_config(cfg, panel, costs, m1, start=split.start - WARMUP, end=end)
        d = d[d.index >= split.start]
        d15, _ = run_config(cfg, panel, costs, m1, cost_mult=1.5, start=split.start - WARMUP, end=end)
        d15 = d15[d15.index >= split.start]
        info["weights"] = info["weights"][info["weights"].index >= split.start]
        c = card(n, d, info, {1.5: float(d15["net"].sum())}, 1, 0)
        dev_dd = float(dev_rank.loc[n, "max_dd"])
        checks = {"net > 0": c.net > 0, "net at costs x1.5 > 0": c.stress[1.5] > 0, "sharpe >= 0.5": c.sharpe >= 0.5,
                  f"max_dd <= 1.5 x DEV ({1.5 * dev_dd:.3f})": c.max_dd <= 1.5 * dev_dd,
                  "tokens positive >= 50%": c.tokens_positive >= 0.5}
        rows.append({"finalist": n, "days": len(d), "net": c.net, "gross": c.gross, "cost": c.cost,
                     "funding": c.funding, "net_x1.5": c.stress[1.5], "pf": c.pf, "sharpe": c.sharpe,
                     "sortino": c.sortino, "max_dd": c.max_dd, "hit_days": c.hit, "tokens_pos": c.tokens_positive,
                     "turnover/day": c.trades_per_day, "passes": all(checks.values()),
                     "failed": "; ".join(k for k, ok in checks.items() if not ok),
                     "by_symbol": {k: round(v, 4) for k, v in info["by_symbol"]["net"].items()},
                     "monthly": {str(k.date()): round(v, 4) for k, v in d["net"].resample("MS").sum().items()}})
        d.to_parquet(OUT / f"{a.split}_{n}_daily.parquet")
    df = pd.DataFrame(rows)
    df.to_json(OUT / f"{a.split}_results.json", orient="records", indent=2)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 400)
    print(df.drop(columns=["by_symbol", "monthly"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for r in rows:
        print(f"\n{r['finalist']} monthly net: {r['monthly']}\n by symbol: {r['by_symbol']}")


if __name__ == "__main__":
    main()
