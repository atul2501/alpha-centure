"""Round 2 improvement research (plan of 2026-10-06; rules in data/experiments/round2_preregistration.json).

    uv run python -m alpha.research.round2

Baseline: P6 on the 23 coins exactly as in paper. Candidates R1-R7 run through alpha.research.overnight.book (same
costs, funding, caps, vol target) with quarterly walk-forward. Judged on:
    DEV        2021-01 -> 2024-07 (gates)
    INFO       2024-07 -> today (already seen; information only)
    UNSEEN-14  the 14 fresh coins (data/experiments/round2_fresh_coins.json), models trained on the 23 only; one look
               per candidate, recorded in data/experiments/lockbox_unseen14.jsonl
"""

import json
import sys
from pathlib import Path

import lightgbm  # noqa: F401
import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.research import signals as sig
from alpha.research.models import feature_frame, target
from alpha.research.overnight import (LAST12, book, costs_wide, dev_gate, features, money, rules_weights, subset,
                                      summarize, walk_scores)
from alpha.research.panel import cached_panel, wide
from alpha.research.phase4 import vol_target
from alpha.research.portfolio_sim import to_weights
from alpha.research.scorecard import pbo_cscv
from alpha.research.splits import DEV_END, open_lockbox
from alpha.strategy import p6

OUT = Path("data/experiments")
LOCKBOX = OUT / "lockbox_unseen14.jsonl"
FRESH = json.loads((OUT / "round2_fresh_coins.json").read_text())["coins"]


# ---------------------------------------------------------------------------------------------------------------
# candidate building blocks

def multi_horizon_score(X_tr, panel_tr, X_ap, vol_tr) -> pd.DataFrame:
    """R2: average of 24h / 72h / 168h ridge forecasts, each z-scored across coins at each time."""
    parts = []
    for h in (24, 72, 168):
        sc = walk_scores(X_tr, target(panel_tr, vol_tr, h), X_ap, h=h)
        z = sc.sub(sc.mean(axis=1), axis=0).div(sc.std(axis=1).replace(0, np.nan), axis=0)
        parts.append(z)
    return sum(parts) / len(parts)


def btc_beta(panel) -> pd.DataFrame:
    ret = wide(panel, "ret")
    btc = ret["BTCUSDT"]
    n = 2160
    cov = ret.rolling(n, min_periods=n // 2).cov(btc)
    return cov.div(btc.rolling(n, min_periods=n // 2).var(), axis=0).shift(1)


def beta_neutral(panel):
    """R4: remove BTC-beta exposure: w - (w.b / b.b) b, then re-cap each coin at 0.5x."""
    beta = btc_beta(panel)

    def post(w, t):
        b = beta.reindex(t).reindex(columns=w.columns).fillna(1.0)
        k = (w * b).sum(axis=1) / (b * b).sum(axis=1).replace(0, np.nan)
        return (w - b.mul(k.fillna(0.0), axis=0)).clip(-0.5, 0.5)
    return post


def cost_band(score: pd.DataFrame, costs: pd.DataFrame, maker: float = p6.MAKER_SHARE_ASSUMED):
    """R5: at a rebalance, change a coin only if the expected gain of the change beats 1.5x its round-trip cost.
    Gain = |dw| x |expected 72h return| (forecast x vol x sqrt(72)); cost = |dw| x round trip, so the test is
    |expected return| > 1.5 x round-trip cost. Weaker forecasts keep the current position."""
    rt = 2 * (maker * costs["maker_bps"] + (1 - maker) * costs["taker_bps"]) / 1e4

    def band_fn(w, ret, vol, h):
        exp = (score.reindex_like(w).abs() * vol.reindex_like(w) * np.sqrt(h)).to_numpy()
        thr = 1.5 * rt.reindex(w.columns).fillna(rt.max()).to_numpy()
        hours = ((w.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).to_numpy()
        W, out, prev = w.to_numpy(), np.zeros_like(w.to_numpy()), np.zeros(w.shape[1])
        for i in range(len(W)):
            if hours[i] % h == 0:
                go = (np.nan_to_num(exp[i]) > thr) | (W[i] == 0)       # always allowed to close
                go &= np.abs(W[i] - prev) > 1e-3
                prev = np.where(go, W[i], prev)
            out[i] = prev
        return pd.DataFrame(out, index=w.index, columns=w.columns)
    return band_fn


def carry_weights(panel, start) -> pd.DataFrame:
    """R6 sleeve: cross-sectional carry (short high 24h-average funding, long low), vol-targeted."""
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    el = wide(panel, "eligible").fillna(False).astype(bool)
    t = ret.index[ret.index >= start]
    sc = sig.compute(panel)["xscarry"]
    return vol_target(to_weights(sc.loc[t], vol.loc[t], el.loc[t], "xs", every=p6.H), ret.loc[t])


# ---------------------------------------------------------------------------------------------------------------

def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    s = get_settings()
    core = s.symbols
    pool = core + FRESH
    prior = json.loads((OUT / "overnight_trials.json").read_text())["trials"]
    with psycopg.connect(s.database_url) as conn:
        wide_p = cached_panel(conn, pool, tag="panel_1h_round2_v1")
        costs = costs_wide(conn, wide_p)
    p23 = subset(wide_p, core)
    p37 = wide_p
    pf = subset(wide_p, FRESH + ["BTCUSDT"], untradable=("BTCUSDT",))
    logger.info("panels: 23 {} rows, 37 {} rows, fresh {} rows", len(p23), len(p37), len(pf))

    X23, y23 = features(p23)
    X37, y37 = features(p37)
    Xf, _ = features(pf)
    Xf = Xf[Xf.index.get_level_values("symbol").isin(FRESH)]
    vol23 = wide(p23, "ret").rolling(sig.VOL_WINDOW, min_periods=48).std()
    logger.info("features ready")

    sc = {"b23": walk_scores(X23, y23, X23), "bf": walk_scores(X23, y23, Xf),
          "mh23": multi_horizon_score(X23, p23, X23, vol23), "mhf": multi_horizon_score(X23, p23, Xf, vol23),
          "r2y23": walk_scores(X23, y23, X23, window_days=730), "r2yf": walk_scores(X23, y23, Xf, window_days=730),
          "w37": walk_scores(X37, y37, X37)}
    logger.info("scores ready")
    rules23, rulesf = rules_weights(p23), rules_weights(pf)
    carry23, carryf = carry_weights(p23, rules23.index.min()), carry_weights(pf, rulesf.index.min())

    # name -> (kwargs on 23 coins, kwargs on the fresh 14 or None = contribution inside the wide book)
    C = {
        "CURRENT_P6_23": (dict(panel=p23, score=sc["b23"]), dict(panel=pf, score=sc["bf"])),
        "R1_ensemble": (dict(panel=p23, score=sc["b23"], blend=rules23), dict(panel=pf, score=sc["bf"], blend=rulesf)),
        "R2_multi_horizon": (dict(panel=p23, score=sc["mh23"]), dict(panel=pf, score=sc["mhf"])),
        "R3_rolling_2y": (dict(panel=p23, score=sc["r2y23"]), dict(panel=pf, score=sc["r2yf"])),
        "R4_beta_neutral": (dict(panel=p23, score=sc["b23"], post=beta_neutral(p23)),
                            dict(panel=pf, score=sc["bf"], post=beta_neutral(pf))),
        "R5_cost_band": (dict(panel=p23, score=sc["b23"], band_fn=cost_band(sc["b23"], costs)),
                         dict(panel=pf, score=sc["bf"], band_fn=cost_band(sc["bf"], costs))),
        "R6_carry_sleeve": (dict(panel=p23, score=sc["b23"], blend=carry23, blend_w=0.25),
                            dict(panel=pf, score=sc["bf"], blend=carryf, blend_w=0.25)),
        "R7_wide_37": (dict(panel=p37, score=sc["w37"]), None),
    }
    results = run_candidates(C, costs, prior + len(C) - 1, core, pool)
    base = results["CURRENT_P6_23"]
    winners = [n for n, r in results.items() if r["better_than_current"]]
    logger.info("better than current: {}", winners or "none")

    # The pre-registered combination (stack the winners' changes into one book) is built in a follow-up run once
    # the winners are known: see combine_winners().
    del base

    df = pd.DataFrame(results.values())
    df.to_json(OUT / "round2_results.json", orient="records", indent=2)
    (OUT / "round2_trials.json").write_text(json.dumps({"trials": prior + len(C) - 1, "winners": winners}))
    pd.set_option("display.width", 250)
    cols = ["name", "dev_30k", "dev_sharpe", "dev_max_dd", "dev_gate", "info_30k", "last12_30k", "unseen_30k",
            "unseen_net_x15", "unseen_pass", "better_than_current", "trades_per_month", "costs_per_month_30k"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


DAILY: dict[str, pd.Series] = {}


def run_candidates(C, costs, n_trials, core, pool, base=None) -> dict:
    out = {}
    for name, (kw, kwf) in C.items():
        res, w = book(costs=costs, **kw)
        res15, _ = book(costs=costs, cost_mult=1.5, **kw)
        coins = pool if kw["panel"].index.get_level_values("symbol").nunique() > len(core) else core
        r = summarize(name, res, w, res15.daily()["net"], n_trials, coins=coins)
        ok, fails = dev_gate(r)
        r.update(dev_gate=ok, dev_fails="; ".join(fails))
        open_lockbox(name, "UNSEEN-14", LOCKBOX)
        if kwf is None:
            fresh_in = [c for c in FRESH if c in res15.by_symbol.index]
            u, u30 = float(res15.by_symbol.loc[fresh_in, "net"].sum()), float("nan")
        else:
            u = float(book(costs=costs, cost_mult=1.5, **kwf)[0].daily()["net"].sum())
            u30 = money(book(costs=costs, **kwf)[0].daily()["net"])
        r.update(unseen_net_x15=u, unseen_30k=u30, unseen_pass=u > 0)
        DAILY[name] = res.daily()["net"]
        out[name] = r
        logger.info("{}: DEV $30k->{:,.0f} sharpe {:.2f} | last12 $30k->{:,.0f} | fresh {:+.3f}", name, r["dev_30k"],
                    r["dev_sharpe"], r["last12_30k"], u)
    b = base or out["CURRENT_P6_23"]
    for name, r in out.items():
        r["better_than_current"] = bool(name != "CURRENT_P6_23" and r["dev_gate"] and r["unseen_pass"]
                                        and r["dev_net_pct"] > b["dev_net_pct"] and r["dev_sharpe"] > b["dev_sharpe"])
    dev = pd.DataFrame({k: v[v.index < DEV_END] for k, v in DAILY.items()}).fillna(0.0)
    pbo = pbo_cscv(dev, n_blocks=10) if dev.shape[1] > 1 else float("nan")
    for r in out.values():
        r["pbo_all_candidates"] = pbo
    pd.DataFrame(DAILY).to_parquet(OUT / "round2_daily.parquet")
    return out


if __name__ == "__main__":
    main()
