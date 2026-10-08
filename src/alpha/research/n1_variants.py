"""N1 improvement test: six pre-registered variants of N1, replayed Jan 2021 -> today with the live pipeline.

    uv run python -m alpha.research.n1_variants            # replay (resumable per year) + summary

Every book uses the same quarterly ridge models (data/research/hist_models/), 72h rebalance, 3x gross / 0.5x per
coin caps, the frozen per-coin cost table (data/research/league_costs.csv), 60% maker fills and funding.

PRE-REGISTERED (2026-10-09, before the run):
    N1   vol_target(0.5 ridge + 0.5 rules), 1% band                      reference
    V1   vol_target(0.3 ridge + 0.7 rules)                               rules-heavy
    V2   vol_target(0.7 ridge + 0.3 rules)                               ridge-heavy
    V3   N1 weights; a coin changes only if |expected 72h move| > 1.5x round-trip cost
    V4   vol_target(0.5 ridge + 0.25 rules + 0.25 cross-sectional carry)
    V5   N1 with a 2% no-trade band
    V6   0.75 x N1 (15% vol target): a sizing choice, never the winner
    rule a variant (V1-V5) improves N1 only if on DEV (2021-01 -> 2024-06) its net > N1's, its Sharpe > N1's and its
         max drawdown <= N1's + 3 points; best DEV Sharpe among those wins. Then: deflated Sharpe with 326 trials,
         PBO across N1 + V1-V5 on DEV, Newey-West t of the daily difference vs N1. VALID-A / last 12 months = info.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.live.report import p6_scores
from alpha.research import signals as sig
from alpha.research.models import feature_frame
from alpha.research.panel import build_panel, wide
from alpha.research.phase4 import apply_band_and_stop, vol_target
from alpha.research.portfolio_sim import newey_west_t, simulate, to_weights
from alpha.research.scorecard import deflated_sharpe, pbo_cscv
from alpha.strategy import p6

MODELS = "data/research/hist_models"
COSTS = Path("data/research/league_costs.csv")
OUT = Path("data/research/n1_variants")
END = pd.Timestamp("2026-10-08", tz="UTC")
PERIODS = {"DEV": ("2021-01-01", "2024-06-30"), "VALID-A": ("2024-07-01", "2025-09-30"),
           "LAST12": ("2025-10-01", "2026-10-07"), "ALL": ("2021-01-01", "2026-10-07")}
CANDIDATES = ["V1", "V2", "V3", "V4", "V5"]
N_TRIALS = 326
DD_SLACK = 0.03


def cost_band(score: pd.DataFrame, costs: pd.DataFrame):
    """At a rebalance a coin changes only if |forecast x vol x sqrt(72)| > 1.5x its round trip; closing is allowed."""
    m = p6.MAKER_SHARE_ASSUMED
    rt = 2 * (m * costs["maker_bps"] + (1 - m) * costs["taker_bps"]) / 1e4

    def band(w, ret, vol):
        exp = (score.reindex_like(w).abs() * vol.reindex_like(w) * np.sqrt(p6.H)).to_numpy()
        thr = 1.5 * rt.reindex(w.columns).fillna(rt.max()).to_numpy()
        hours = ((w.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).to_numpy()
        W, out, prev = w.to_numpy(), np.zeros_like(w.to_numpy()), np.zeros(w.shape[1])
        for i in range(len(W)):
            if hours[i] % p6.H == 0:
                go = ((np.nan_to_num(exp[i]) > thr) | (W[i] == 0)) & (np.abs(W[i] - prev) > 1e-3)
                prev = np.where(go, W[i], prev)
            out[i] = prev
        return pd.DataFrame(out, index=w.index, columns=w.columns)
    return band


def books(panel, scores, sc, costs) -> dict:
    """name -> (pre-band weights, band function(w, ret, vol))."""
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    el = wide(panel, "eligible").fillna(False).astype(bool)
    base = p6.target_weights(panel, sc)
    rules = p6.rules_weights(panel).reindex_like(base).fillna(0.0)
    carry = vol_target(to_weights(scores["xscarry"].reindex(columns=ret.columns), vol, el, "xs", every=p6.H), ret)
    carry = carry.reindex_like(base).fillna(0.0)
    n1 = vol_target(0.5 * base + 0.5 * rules, ret)
    band = lambda b: lambda w, r, v: apply_band_and_stop(w, r, v, b, None, p6.H)  # noqa: E731
    return {
        "N1": (n1, band(p6.BAND)),
        "V1": (vol_target(0.3 * base + 0.7 * rules, ret), band(p6.BAND)),
        "V2": (vol_target(0.7 * base + 0.3 * rules, ret), band(p6.BAND)),
        "V3": (n1, cost_band(sc, costs)),
        "V4": (vol_target(0.5 * base + 0.25 * rules + 0.25 * carry, ret), band(p6.BAND)),
        "V5": (n1, band(0.02)),
        "V6": (0.75 * n1, band(p6.BAND)),
        "P6": (base, band(p6.BAND)),
        "RULES": (rules, band(p6.BAND)),
    }


def year(conn, symbols, costs, since, end) -> pd.DataFrame:
    panel = build_panel(conn, symbols, start=since - pd.Timedelta(days=p6.HISTORY_DAYS), end=end)
    scores = sig.compute(panel)
    X, _ = feature_frame(panel, scores)
    sc = p6_scores(X, end, MODELS)
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    fund = wide(panel, "funding_rate")
    per_side = p6.MAKER_SHARE_ASSUMED * costs["maker_bps"] + (1 - p6.MAKER_SHARE_ASSUMED) * costs["taker_bps"]
    out = []
    for name, (w, band) in books(panel, scores, sc, costs).items():
        w = band(w, ret, vol)
        t = w.index[(w.index >= since) & (w.index < end)]
        d = simulate(w.loc[t], ret.loc[t], fund.loc[t], per_side).daily()
        out.append(d.assign(book=name).rename_axis("day").reset_index())
    return pd.concat(out, ignore_index=True)


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod() * 30_000
    mo = (1 + r).resample("ME").prod() - 1
    return {"final": eq.iloc[-1], "sharpe": r.mean() / r.std() * np.sqrt(365), "vol": r.std() * np.sqrt(365),
            "maxdd": float(((eq.cummax() - eq) / eq.cummax()).max()), "worst_month": mo.min(), "pos_months": (mo > 0).mean()}


def summarize(df: pd.DataFrame) -> dict:
    net = df.pivot(index="day", columns="book", values="net")
    cost = df.pivot(index="day", columns="book", values="cost")
    turn = df.pivot(index="day", columns="book", values="turnover")
    res = {}
    for p, (a, b) in PERIODS.items():
        x = net.loc[a:b]
        res[p] = {k: {**stats(x[k]), "cost": cost[k].loc[a:b].sum(), "turn": turn[k].loc[a:b].sum() / len(x) * 365,
                      "t_vs_n1": newey_west_t(x[k] - x["N1"], 5) if k != "N1" else np.nan} for k in net.columns}
    dev, ref = res["DEV"], res["DEV"]["N1"]
    passed = [k for k in CANDIDATES if dev[k]["final"] > ref["final"] and dev[k]["sharpe"] > ref["sharpe"]
              and dev[k]["maxdd"] <= ref["maxdd"] + DD_SLACK]
    winner = max(passed, key=lambda k: dev[k]["sharpe"]) if passed else None
    xdev = net.loc[PERIODS["DEV"][0]:PERIODS["DEV"][1]]
    return {"periods": res, "passed": passed, "winner": winner,
            "dsr_winner": deflated_sharpe(xdev[winner], N_TRIALS) if winner else None,
            "dsr_n1": deflated_sharpe(xdev["N1"], N_TRIALS),
            "pbo": pbo_cscv(xdev[["N1", *CANDIDATES]]),
            "years": {k: {int(y): float(v) for y, v in ((1 + net[k]).groupby(net.index.year).prod() - 1).items()}
                      for k in net.columns}}


def main() -> None:
    s = get_settings()
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(COSTS, index_col=0)
    costs = c.reindex(s.symbols).fillna(c.max())
    with psycopg.connect(s.database_url) as conn:
        for y in range(2021, 2027):
            f = OUT / f"daily_{y}.parquet"
            if f.exists():
                continue
            since, end = pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), END)
            t0 = time.time()
            year(conn, s.symbols, costs, since, end).to_parquet(f)
            logger.info("year {} in {:.0f}s", y, time.time() - t0)
            gc.collect()
    df = pd.concat(pd.read_parquet(f) for f in sorted(OUT.glob("daily_*.parquet")))
    df["day"] = pd.to_datetime(df["day"])
    out = summarize(df)
    (OUT / "summary.json").write_text(json.dumps(out, default=float, indent=1))
    full = pd.DataFrame(out["periods"]["ALL"]).T[["final", "sharpe", "maxdd", "worst_month", "t_vs_n1"]]
    dev = pd.DataFrame(out["periods"]["DEV"]).T[["final", "sharpe", "maxdd"]]
    print("DEV\n", dev.round(3).to_string(), "\n\nALL\n", full.round(3).to_string())
    print(f"\npassed: {out['passed']}  winner: {out['winner']}  DSR winner: {out['dsr_winner']}  "
          f"DSR N1: {out['dsr_n1']:.3f}  PBO: {out['pbo']:.2f}")


if __name__ == "__main__":
    main()
