"""P6 vs the top-4 SOL-tournament models on the 23-coin universe, 2025-10-01 -> 2026-09-30 (descriptive only).

    uv run python -m alpha.tournament compare

The window is VALID-B, already seen by P6: this comparison describes, it does not select. Each model's look is
recorded in the lockbox (split COMPARE-2025-10). Window end = 2026-09-30 because the 1m-flow features end there.

Tournament models (random_forest, decision_tree, linear_svm, buy_hold) run per coin with the tournament procedure
unchanged: quarterly test folds from 2025-10-01, expanding training on all earlier bars, 90-day validation, purge /
embargo, the same grid and per-fold config / threshold choice on validation only. Each coin trades a $30k / 23 slice
(costs computed on that notional with the coin's own spread and depth); the book is the sum over coins, so gross
exposure is at most 1x. P6 runs through its own research path (alpha.research.overnight.book, quarterly ridge
retraining, 20% vol target, 3x cap, 1% band) with 60% maker (as in paper) and taker-only execution.
All P&L is additive on a $30k start (not compounded) so every book is measured the same way.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha.config import get_settings
from alpha.research.splits import LockboxError, open_lockbox
from alpha.tournament.database.repo import Repo, clean_json
from alpha.tournament.execution.fills import CostConfig
from alpha.tournament.experiments.matrix import load_specs
from alpha.tournament.training import runner

START = pd.Timestamp("2025-10-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")  # exclusive
OUT = Path("data/experiments/compare_2025_10_v2")  # v1 = before the depth-floor fix (bad book snapshots)
EQUITY = 30_000.0
MODELS = {"random_forest": "configs/tournament/stage_3.yaml", "decision_tree": "configs/tournament/stage_3.yaml",
          "linear_svm": "configs/tournament/stage_2.yaml", "buy_hold": "configs/tournament/stage_2.yaml"}
STAGE = "compare_2025_10"


def live_spreads(conn) -> dict[str, float]:
    rows = conn.execute("""SELECT split_part(symbol, '.', 1), percentile_cont(0.5) WITHIN GROUP
            (ORDER BY (best_ask - best_bid) / ((best_ask + best_bid) / 2) * 1e4) FROM book_tick GROUP BY 1""").fetchall()
    return {s: float(v) for s, v in rows}


def _spec(model: str):
    _, specs = load_specs(MODELS[model])
    s = next(x for x in specs if x.entry == model)
    s.stage = STAGE
    return s


def run_models(conn, coins: list[str]) -> None:
    """Per (model, coin): daily net (account fraction of the coin's slice), trades, variant totals -> OUT."""
    repo = Repo(conn)
    OUT.mkdir(parents=True, exist_ok=True)
    spreads = live_spreads(conn)
    worst = max(spreads.values())
    for m in MODELS:
        try:
            open_lockbox(f"{m}@23", "COMPARE-2025-10")
        except LockboxError:
            logger.warning("{}@23 already opened on COMPARE-2025-10: continuing the same comparison run", m)
    n = len(get_settings().symbols)  # slices are always $30k / 23, whichever subset runs now
    for coin in coins:
        todo = [m for m in MODELS if not (OUT / f"{m}__{coin}.json").exists()]
        if not todo:
            continue
        cost = CostConfig(spread_bps=2 * spreads.get(coin, worst), notional=EQUITY / n)
        ctx = runner.build_context("real", conn, cost=cost, symbol=coin, forward=(START, END))
        logger.info("{}: {} bars, {} folds, round trip median {:.1f} bps", coin, len(ctx.raw), len(ctx.folds),
                    float(ctx.rt_cost[ctx.rt_cost.index >= START].median()))
        for m in todo:
            spec = _spec(m)
            spec.entry = f"{m}__{coin}"
            out = runner.run_experiment(ctx, spec, repo)
            res = {"model": m, "coin": coin, "status": out.status, "reason": out.reason,
                   "summary": out.summary, "folds": out.folds}
            if out.status == "done":
                for v in ("policy", "forced"):
                    bp = out.bar_pnl[v]
                    bp[bp.index >= START].groupby(bp.index[bp.index >= START].floor("D")).sum() \
                        .rename("net").to_frame().to_parquet(OUT / f"{m}__{coin}__{v}_daily.parquet")
                    t = out.trades[v]
                    if len(t):
                        t[["entry_ts", "exit_ts", "side", "bars", "gross_bps", "fee_bps", "slip_bps", "funding_bps",
                           "net_bps"]].to_parquet(OUT / f"{m}__{coin}__{v}_trades.parquet")
            (OUT / f"{m}__{coin}.json").write_text(json.dumps(clean_json(res), default=str))
            s = out.summary.get("policy", {})
            logger.info("  {} {}: {} trades net {:.3f} ({})", coin, m, s.get("trades"), s.get("net", 0) or 0,
                        out.status)
        del ctx


def run_p6(conn) -> dict[str, pd.Series]:
    """P6 daily net (fraction of $30k) on the 23 coins for the window: as in paper (60% maker) and taker-only,
    plus cost stress."""
    from alpha.research.overnight import book, costs_wide, features, subset, walk_scores
    from alpha.research.panel import cached_panel

    s = get_settings()
    core = s.symbols
    from alpha.research.round2 import FRESH

    wide_p = cached_panel(conn, core + FRESH, tag="panel_1h_round2_v1")
    costs = costs_wide(conn, wide_p)
    p23 = subset(wide_p, core)
    X, y = features(p23)
    score = walk_scores(X, y, X)
    out = {}
    for name, kw in {"P6 (60% maker, as in paper)": dict(maker=0.6), "P6 taker-only": dict(maker=0.0),
                     "P6 x1.5 cost": dict(maker=0.6, cost_mult=1.5), "P6 x2 cost": dict(maker=0.6, cost_mult=2.0)}.items():
        # run continuously from 2021 (as P6 runs live) and cut the window: starting flat on START would skip the
        # vol-target warm-up and overstate October's leverage (-24.8% vs -8.1% that month)
        res, w = book(p23, score, costs, **kw)
        w = w[(w.index >= START) & (w.index < END)]
        d = res.daily()
        d = d[(d.index >= START) & (d.index < END)]
        out[name] = d
        if name.startswith("P6 (60%"):
            ch = (w.diff().abs() > 1e-9)
            out["_p6_changes"] = int(ch.sum().sum())
            out["_p6_by_symbol"] = res.by_symbol["net"]
            out["_p6_gross_lev"] = float(w.abs().sum(axis=1).mean())
    return out


def stats(daily: pd.Series) -> dict:
    x = daily.fillna(0.0)
    eq = x.cumsum()
    dd = float((eq.cummax().clip(lower=0) - eq).max())
    sd = x.std()
    down = x[x < 0].std()
    m = x.groupby(x.index.strftime("%Y-%m")).sum()
    net = float(x.sum())
    return {"net": net, "net_usd": net * EQUITY, "sharpe": float(x.mean() / sd * np.sqrt(365)) if sd > 0 else np.nan,
            "sortino": float(x.mean() / down * np.sqrt(365)) if down > 0 else np.nan, "max_dd": dd,
            "calmar": net / dd if dd > 0 else np.nan, "vol_ann": float(sd * np.sqrt(365)),
            "pf_days": float(x[x > 0].sum() / -x[x < 0].sum()) if (x < 0).any() else np.nan,
            "win_days": float((x > 0).mean()), "months_pos": int((m > 0).sum()), "months": int(len(m)),
            "worst_month": float(m.min()), "best_month": float(m.max()), "monthly": m.round(5).to_dict()}


def books(coins: list[str]) -> dict:
    """Combine per-coin results into 23-coin books (equal $30k / 23 slices)."""
    n = len(coins)
    idx = pd.date_range(START, END - pd.Timedelta(days=1), freq="D", tz="UTC")
    out = {}
    for m in MODELS:
        for v in ("policy", "forced"):
            tot = pd.Series(0.0, index=idx)
            tr, per_coin, var = [], {}, {}
            for c in coins:
                f = OUT / f"{m}__{c}__{v}_daily.parquet"
                j = json.loads((OUT / f"{m}__{c}.json").read_text())
                for k, s in (j.get("summary") or {}).items():
                    var[k] = var.get(k, 0.0) + (s.get("net") or 0.0) / n
                if f.exists():
                    d = pd.read_parquet(f)["net"].reindex(idx, fill_value=0.0)
                    tot += d / n
                    per_coin[c] = float(d.sum() / n)
                else:
                    per_coin[c] = 0.0
                tf = OUT / f"{m}__{c}__{v}_trades.parquet"
                if tf.exists():
                    t = pd.read_parquet(tf)
                    t["coin"] = c
                    tr.append(t)
            t = pd.concat(tr) if tr else pd.DataFrame()
            name = {"policy": m, "forced": f"{m} (always trading)"}[v]
            if m == "buy_hold" and v == "forced":
                name = "hold all 23 coins (market)"
            st = stats(tot)
            st.update(trades=int(len(t)), fees=float(t["fee_bps"].sum() / 1e4 / n) if len(t) else 0.0,
                      slippage=float(t["slip_bps"].sum() / 1e4 / n) if len(t) else 0.0,
                      funding=float(t["funding_bps"].sum() / 1e4 / n) if len(t) else 0.0,
                      long=float(t.loc[t["side"] > 0, "net_bps"].sum() / 1e4 / n) if len(t) else 0.0,
                      short=float(t.loc[t["side"] < 0, "net_bps"].sum() / 1e4 / n) if len(t) else 0.0,
                      coins_pos=int(sum(v_ > 0 for v_ in per_coin.values())), per_coin=per_coin,
                      net_x1_5=var.get("cost_x1.5") if v == "policy" else None,
                      net_x2=var.get("cost_x2") if v == "policy" else None, daily=tot)
            out[name] = st
    return out


def main(conn) -> dict:
    coins = get_settings().symbols
    run_models(conn, coins)
    res = books(coins)
    p6 = run_p6(conn)
    for name in ("P6 (60% maker, as in paper)", "P6 taker-only"):
        d = p6[name]
        st = stats(d["net"])
        st.update(trades=p6["_p6_changes"], fees=float(d["cost"].sum()), slippage=0.0,
                  funding=float(d["funding"].sum()), long=None, short=None,
                  net_x1_5=float(p6["P6 x1.5 cost"]["net"].sum()) if name.startswith("P6 (60%") else None,
                  net_x2=float(p6["P6 x2 cost"]["net"].sum()) if name.startswith("P6 (60%") else None,
                  gross_lev=p6["_p6_gross_lev"], daily=d["net"])
        res[name] = st
    # volatility-matched net: each book scaled to P6's realized vol (display only)
    ref = res["P6 (60% maker, as in paper)"]["vol_ann"]
    for k, v in res.items():
        v["vol_matched_net"] = v["net"] * ref / v["vol_ann"] if v["vol_ann"] > 0 else np.nan
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: v["daily"] for k, v in res.items()}).to_parquet(OUT / "books_daily.parquet")
    (OUT / "books.json").write_text(json.dumps(clean_json({k: {kk: vv for kk, vv in v.items() if kk != "daily"}
                                                            for k, v in res.items()}), indent=1, default=str))
    return res
