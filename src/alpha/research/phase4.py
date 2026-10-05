"""Phase 4: portfolio construction on the Phase 3 survivors, DEV out-of-sample (2021-01 -> 2024-06).

    uv run python -m alpha.research.phase4

Pre-declared configurations (every one is a counted trial):
    P1  XS momentum only (market-neutral), 72h, taker
    P2  XS + TS (50/50 risk), 72h, taker
    P3  P2 + 1% no-trade band + 60% maker fills
    P4  P3 + catastrophe stop (3 daily sigmas against the position -> flat until the next rebalance)
    P5  XS only + band + maker (P1 with P3's execution)
    P6  ridge (M1, 72h) combined book with P3's execution
All: portfolio vol target 20% a year (scale from the trailing 60-day realized vol of the unscaled book, lagged),
gross leverage cap 3x, per-coin cap 0.5x equity. Results are in equity terms.
Robustness for the ranked winner: lookbacks and rebalance period perturbed +-20%, costs x1.5 / x2, shuffled
control, PBO across all candidates, deflated Sharpe with every trial so far.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.research import signals as sig
from alpha.research.models import OOS_START, feature_frame, target, walk_forward
from alpha.research.panel import cached_panel, wide
from alpha.research.portfolio_sim import newey_west_t, simulate, to_weights
from alpha.research.scorecard import Card, deflated_sharpe, pbo_cscv, rank, verdict
from alpha.research.screen import symbol_costs
from alpha.research.splits import DEV_END

OUT = Path("data/experiments")
TARGET_VOL = 0.20
MAX_GROSS = 3.0
MAX_COIN = 0.5
MAKER_FILL = 0.6


@dataclass(frozen=True)
class Config:
    name: str
    books: tuple[str, ...]          # "xs", "ts", "m1"
    h: int = 72
    band: float = 0.0               # skip coin trades smaller than this (equity fraction)
    maker: float = 0.0              # share of turnover filled passively
    stop_sigmas: float | None = None
    lookback_mult: float = 1.0      # robustness: scale momentum lookbacks
    h_mult: float = 1.0


CONFIGS = [
    Config("P1_xs", ("xs",)),
    Config("P2_xs_ts", ("xs", "ts")),
    Config("P3_xs_ts_band_maker", ("xs", "ts"), band=0.01, maker=MAKER_FILL),
    Config("P4_xs_ts_band_maker_stop", ("xs", "ts"), band=0.01, maker=MAKER_FILL, stop_sigmas=3.0),
    Config("P5_xs_band_maker", ("xs",), band=0.01, maker=MAKER_FILL),
    Config("P6_m1_band_maker", ("m1",), band=0.01, maker=MAKER_FILL),
]


def momentum_scores(panel: pd.DataFrame, mult: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(xs score, ts score) with lookbacks scaled by mult (1.0 = the screened 168/336/720 and 480h breakout)."""
    ret = wide(panel, "ret")
    close = wide(panel, "close")
    el = wide(panel, "eligible").fillna(False).astype(bool)
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    cum = ret.fillna(0).cumsum()
    xs, ts = [], []
    for L in (168, 336, 720):
        L2 = int(round(L * mult))
        scaled = (cum - cum.shift(L2)) / (vol * np.sqrt(L2))
        xs.append(sig._xs_rank(scaled, el))
        if L >= 336:
            ts.append(scaled.clip(-3, 3) / 3)
    ts.append(sig._donchian(close, int(round(480 * mult))))
    return sum(xs) / len(xs), sum(ts) / len(ts)


def apply_band_and_stop(w: pd.DataFrame, ret: pd.DataFrame, vol: pd.DataFrame, band: float,
                        stop_sigmas: float | None, every: int) -> pd.DataFrame:
    """Sequential pass: no-trade band at rebalances; catastrophe stop between them."""
    W = w.to_numpy()
    R = ret.reindex_like(w).fillna(0.0).to_numpy()
    V = (vol.reindex_like(w).to_numpy() * np.sqrt(24))  # daily sigma
    hours = ((w.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).to_numpy()
    out = np.zeros_like(W)
    prev = np.zeros(W.shape[1])
    moved = np.zeros(W.shape[1])   # log move since entry
    stopped = np.zeros(W.shape[1], bool)
    for t in range(len(W)):
        if hours[t] % every == 0:
            target = W[t]
            small = np.abs(target - prev) < band
            cur = np.where(small & ~stopped, prev, target)
            moved[~small | stopped] = 0.0
            stopped[:] = False
        else:
            cur = prev.copy()
            if stop_sigmas is not None:
                moved += R[t]
                hit = (np.sign(cur) * moved < -stop_sigmas * np.nan_to_num(V[t], nan=np.inf)) & (cur != 0)
                cur[hit] = 0.0
                stopped |= hit
        out[t] = cur
        prev = cur
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def vol_target(w: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    """Scale the gross-1 book to TARGET_VOL using its trailing 60-day realized vol (known before each day)."""
    unit = (w.shift(1).fillna(0) * np.expm1(ret.reindex_like(w).fillna(0))).sum(axis=1)
    daily = unit.groupby(unit.index.floor("D")).sum()
    rv = daily.rolling(60, min_periods=20).std().shift(1) * np.sqrt(365)  # yesterday's estimate
    scale = (TARGET_VOL / rv).clip(upper=MAX_GROSS).reindex(w.index.floor("D")).to_numpy()
    scaled = w.mul(np.nan_to_num(scale, nan=1.0), axis=0)
    gross = scaled.abs().sum(axis=1)
    scaled = scaled.div(np.maximum(gross / MAX_GROSS, 1.0), axis=0)
    return scaled.clip(-MAX_COIN, MAX_COIN)


def run_config(cfg: Config, panel, costs, m1_score: pd.DataFrame | None = None,
               cost_mult: float = 1.0, shuffle_seed: int | None = None,
               start: pd.Timestamp = OOS_START, end: pd.Timestamp = DEV_END) -> tuple[pd.DataFrame, dict]:
    t_all = wide(panel, "ret").index
    t = t_all[(t_all >= start) & (t_all < end)]
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    el = wide(panel, "eligible").fillna(False).astype(bool)
    h = int(round(cfg.h * cfg.h_mult))
    xs, ts = momentum_scores(panel, cfg.lookback_mult)
    if shuffle_seed is not None:  # control: same structure, coin labels shuffled each day -> no information
        rng = np.random.default_rng(shuffle_seed)
        xs = xs.apply(lambda r: pd.Series(rng.permutation(r.to_numpy()), index=r.index), axis=1)
        ts = ts.apply(lambda r: pd.Series(rng.permutation(r.to_numpy()), index=r.index), axis=1)
    books = []
    for b in cfg.books:
        if b == "xs":
            books.append(to_weights(xs.loc[t], vol.loc[t], el.loc[t], "xs", every=h))
        elif b == "ts":
            books.append(to_weights(ts.loc[t], vol.loc[t], el.loc[t], "ts", every=h))
        elif b == "m1":
            books.append(to_weights(m1_score.reindex(t).reindex(columns=ret.columns), vol.loc[t], el.loc[t], "ts", every=h))
    w = sum(books) / len(books)
    w = vol_target(w, ret.loc[t])
    w = apply_band_and_stop(w, ret.loc[t], vol.loc[t], cfg.band, cfg.stop_sigmas, h)
    per_side = cost_mult * (cfg.maker * costs["maker_bps"] + (1 - cfg.maker) * costs["taker_bps"])
    res = simulate(w, ret.loc[t], wide(panel, "funding_rate").loc[t], per_side)
    d = res.daily()
    lev = w.abs().sum(axis=1)
    info = {"avg_gross": float(lev.mean()), "max_gross": float(lev.max()),
            "net_beta_avg": float(w.sum(axis=1).mean()), "turnover_per_day": float(d["turnover"].mean()),
            "by_symbol": res.by_symbol, "weights": w}
    return d, info


def card(name: str, d: pd.DataFrame, info: dict, stress: dict, n_trials: int, complexity: int) -> Card:
    net, gross, cost, fund = d["net"], d["gross"], d["cost"], d["funding"]
    turn = d["turnover"].sum()
    years = net.groupby(net.index.year).sum()
    sym = info["by_symbol"]["net"]
    pos = sym.clip(lower=0)
    eq = net.cumsum()
    sd, down = net.std(), net[net < 0].std()
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    return Card(
        name=name, trades=int((info["weights"].diff().abs() > 1e-9).sum().sum()), net=float(net.sum()),
        gross=float(gross.sum()), cost=float(cost.sum()), funding=float(fund.sum()),
        net_bps=float(net.sum() / turn * 1e4), gross_bps=float(gross.sum() / turn * 1e4),
        cost_bps=float(cost.sum() / turn * 1e4), funding_bps=float(max(fund.sum(), 0) / turn * 1e4),
        net_se_bps=float(net.std() * np.sqrt(len(net)) / turn * 1e4), t_stat=newey_west_t(net, 5),
        pf=float(wins / losses) if losses > 0 else np.inf, expectancy=float(net.mean()), hit=float((net > 0).mean()),
        max_dd=float((eq.cummax() - eq).max()), sharpe=float(net.mean() / sd * np.sqrt(365)),
        sortino=float(net.mean() / down * np.sqrt(365)), trades_per_day=float(d["turnover"].mean()),
        years_positive=float((years > 0).mean()), tokens_positive=float((sym > 0).mean()),
        max_token_share=float(pos.max() / pos.sum()) if pos.sum() > 0 else 1.0, stress=stress,
        by_symbol=info["by_symbol"], by_year=years.to_frame("net"), dsr=deflated_sharpe(net, n_trials),
        complexity=complexity,
    )


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        panel = cached_panel(conn, s.symbols)
        costs = symbol_costs(conn, panel)
        panel = panel[panel.index.get_level_values("time") < DEV_END]
        scores = sig.compute(panel)
        X, vol = feature_frame(panel, scores)
        m1 = walk_forward(conn, panel, X, target(panel, vol, 72), 72, "M1")
    prior = json.loads((OUT / "phase2_trials.json").read_text())["trials"] + 8 + 7  # screen + phase 3 + 3b
    n_trials = prior + len(CONFIGS) + 4  # + robustness perturbations below
    results, daily = {}, {}
    for cfg in CONFIGS:
        d, info = run_config(cfg, panel, costs, m1_score=m1)
        stress = {k: float(run_config(cfg, panel, costs, m1, cost_mult=k)[0]["net"].sum()) for k in (1.5, 2.0)}
        c = card(cfg.name, d, info, stress, n_trials, complexity=1 if "m1" in cfg.books else 0)
        sh, _ = run_config(cfg, panel, costs, m1, shuffle_seed=7) if "m1" not in cfg.books else (None, None)
        if sh is not None:
            c.shuffled_t = newey_west_t(sh["net"], 5)
        # +-20% robustness on lookbacks and rebalance period
        perturbed = [run_config(Config(cfg.name, cfg.books, cfg.h, cfg.band, cfg.maker, cfg.stop_sigmas, lm, hm),
                                panel, costs, m1)[0]["net"].sum()
                     for lm, hm in ((0.8, 1.0), (1.2, 1.0), (1.0, 0.8), (1.0, 1.2))]
        c.param_stable = bool(min(perturbed) > 0)
        results[cfg.name] = (c, info, perturbed)
        daily[cfg.name] = d["net"]
        logger.info("{} done: net {:.2f}, sharpe {:.2f}", cfg.name, c.net, c.sharpe)
    D = pd.DataFrame(daily).fillna(0.0)
    pbo = pbo_cscv(D, n_blocks=10)
    for c, _, _ in results.values():
        if c.complexity == 1 and np.isnan(c.shuffled_t):
            c.shuffled_t = results["P1_xs"][0].shuffled_t  # ridge book: no row shuffle; use the rules' control
        c.pbo = pbo
    table = rank([c for c, _, _ in results.values()])
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 200)
    cols = ["name", "passes", "net", "gross", "cost", "funding", "pf", "max_dd", "sharpe", "sortino", "t_stat",
            "years_positive", "tokens_positive", "max_token_share", "net_x1.5", "net_x2.0", "dsr", "pbo", "failed"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    extra = pd.DataFrame({n: {"avg_gross": i["avg_gross"], "max_gross": i["max_gross"], "net_beta": i["net_beta_avg"],
                              "turnover/day": i["turnover_per_day"], "perturbed_min": min(p),
                              **{f"y{k}": v for k, v in c.by_year["net"].items()}}
                          for n, (c, i, p) in results.items()}).T
    print("\n", extra.to_string(float_format=lambda x: f"{x:.3f}"))
    table.to_csv(OUT / "phase4_rank.csv", index=False)
    extra.to_csv(OUT / "phase4_extra.csv")
    D.to_parquet(OUT / "phase4_daily.parquet")
    (OUT / "phase4_trials.json").write_text(json.dumps({"trials": n_trials, "pbo": pbo}))


if __name__ == "__main__":
    main()
