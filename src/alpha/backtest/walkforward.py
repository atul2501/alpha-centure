"""Walk-forward backtest of the full decision stack.

For each test fold, using ONLY data whose outcome was known before the fold started:
  1. fit the regime HMM on 4h data before the fold, filter causally over everything
  2. fit_fold(): choose exit policies, gates (EV threshold, max cost) on inner time splits, learn the policy
     (enabled strategy x tf x regime cells) and fit the meta model, all per the WFConfig
  3. trade the fold with decide(): one position per symbol, best EV wins, PASS otherwise

Variants compared on the same folds:
  raw            every setup (still one position per symbol)
  regime         regime-gated cells only
  regime_meta    regime gate + meta model EV threshold   <- the system
  shuffled_meta  regime gate + meta model trained on shuffled labels (sanity check: should not beat regime)

Run: uv run python -m alpha.backtest.walkforward --fold-months 3 --tfs 15m,1h
"""

import json
import sys
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.decision import PASS, Policy, decide, static_reasons
from alpha.models.meta import MetaModel, recency_weights
from alpha.regime.hmm import RegimeModel, attach_regime, load_regime_frames
from alpha.research.dataset import build_dataset
from alpha.strategies.labels import EXIT_MENU, apply_exit

THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
VARIANTS = ["raw", "regime", "regime_meta", "shuffled_meta"]


@dataclass(frozen=True)
class WFConfig:
    """Everything that defines how a fold is trained and traded. Baseline == the original Step 2 system."""

    name: str = "baseline"
    exits: str = "fixed"                 # an EXIT_MENU name, or "learn" (best per strategy on training data)
    ev_mode: str = "avg"                 # avg | setup (see MetaModel)
    calibrate: bool = False
    thresholds: tuple[float, ...] = THRESHOLDS
    ev_floor: float = 0.0                # thresholds below this are never chosen
    max_cost_grid: tuple = (None,)       # candidate max_cost_r values (None = no cost gate)
    inner_splits: int = 1                # 1 = single 75/25 split; k = average over k rolling validation windows
    half_life_days: float | None = None  # recency weighting for policy + meta
    rolling_days: int | None = None      # train only on the most recent N days
    max_same_side: int | None = None
    sizing: str = "fixed"
    daily_loss_r: float | None = None    # stop opening trades for the day after losing this many R
    min_trades: int = 20


@dataclass
class Variant:
    use_regime: bool
    use_meta: bool


VARIANT_CFG = {"raw": Variant(False, False), "regime": Variant(True, False),
               "regime_meta": Variant(True, True), "shuffled_meta": Variant(True, True)}


# ---------------------------------------------------------------------------------------------------------------
# simulation

def simulate(rows: pd.DataFrame, policy: Policy, use_regime: bool, use_meta: bool,
             daily_loss_r: float | None = None) -> pd.DataFrame:
    """Replay setups in time order. Returns every row with action/reason/risk_frac (trades have action != PASS)."""
    rows = rows.sort_index(kind="stable")
    static = static_reasons(rows, policy, use_regime, use_meta)
    out = rows.assign(action=PASS, reason=static, risk_frac=0.0)
    survivors = rows[static == ""]
    open_trades: list[tuple[str, int, pd.Timestamp, float]] = []  # symbol, side, exit_time, r
    decided = []
    for t, group in survivors.groupby(level=0, sort=True):
        if daily_loss_r is not None:
            today = sum(r for _, _, ex, r in open_trades if ex <= t and ex.normalize() == t.normalize())
            if today <= -daily_loss_r:
                decided.append(group.assign(action=PASS, reason="daily_loss_limit"))
                continue
        live = [(s, sd) for s, sd, ex, _ in open_trades if ex >= t]
        sides = {1: sum(sd == 1 for _, sd in live), -1: sum(sd == -1 for _, sd in live)}
        d = decide(group, policy, {s for s, _ in live}, use_regime, use_meta, open_sides=sides)
        decided.append(d)
        for tr in d[d["action"] != PASS].itertuples():
            open_trades.append((tr.symbol, int(tr.side), tr.exit_time, float(tr.r)))
    if decided:
        d = pd.concat(decided)
        key = [out.index.name or "index", "symbol", "strategy", "tf", "side"]
        out = out.reset_index().merge(d.reset_index()[[*key, "action", "reason"]], on=key, how="left",
                                      suffixes=("", "_d"))
        has = out["action_d"].notna()
        out.loc[has, "action"], out.loc[has, "reason"] = out.loc[has, "action_d"], out.loc[has, "reason_d"]
        out = out.drop(columns=["action_d", "reason_d"]).set_index(rows.index.name or "index")
    traded = (out["action"] != PASS).to_numpy()
    if traded.any():
        ev = out["ev_r"].to_numpy(float) if "ev_r" in out else np.full(len(out), np.nan)
        out.loc[traded, "risk_frac"] = [policy.risk_for(v) for v in ev[traded]]
    return out


def metrics(trades: pd.DataFrame, risk: float = 0.01, days: float | None = None) -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate": np.nan, "avg_r": np.nan, "total_r": 0.0, "final_equity": 1.0,
                "max_dd": 0.0, "sharpe": np.nan, "trades_per_day": 0.0}
    t = trades.sort_values("exit_time")
    rf = t["risk_frac"].to_numpy(float) if "risk_frac" in t and (t["risk_frac"] > 0).all() else np.full(len(t), risk)
    ret = rf * t["r"].to_numpy(float)
    eq = np.cumprod(1 + ret)
    dd = 1 - eq / np.maximum.accumulate(eq)
    daily = pd.Series(ret, index=t.index).groupby(t["exit_time"].dt.floor("1D").to_numpy()).sum()
    if days:
        daily = daily.reindex(pd.date_range(daily.index.min(), periods=int(days), freq="1D"), fill_value=0.0)
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan
    return {"trades": int(len(t)), "win_rate": float(t["win"].mean()), "avg_r": float(t["r"].mean()),
            "total_r": float(t["r"].sum()), "final_equity": float(eq[-1]), "max_dd": float(dd.max()),
            "sharpe": float(sharpe), "trades_per_day": float(len(t) / days) if days else np.nan}


# ---------------------------------------------------------------------------------------------------------------
# training one fold (shared with alpha.trainer so production trains exactly like the backtest)

@dataclass
class FoldModel:
    policy: Policy
    meta: MetaModel
    exits: dict[str, str] = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def latest_exit_time(ds: pd.DataFrame) -> pd.Series:
    """Latest exit over all labeled exit policies: purge on this so no training label leaks into the fold."""
    cols = [c for c in ds.columns if c.startswith("exit_time__")] or ["exit_time"]
    return ds[cols].max(axis=1)


def learn_exits(train: pd.DataFrame, cfg: WFConfig, w: np.ndarray | None) -> dict[str, str]:
    if cfg.exits == "fixed":
        return {}
    if cfg.exits != "learn":
        return {s: cfg.exits for s in train["strategy"].unique()}
    names = [n for n in EXIT_MENU if f"r__{n}" in train]
    choice = {}
    for strat, g in train.groupby("strategy"):
        ww = None if w is None else w[(train["strategy"] == strat).to_numpy()]
        means = {n: np.average(g[f"r__{n}"].to_numpy(float), weights=ww) for n in names}
        choice[strat] = max(means, key=means.get)
    return choice


def _inner_windows(train: pd.DataFrame, k: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    order = train.index.sort_values()
    if k <= 1:
        cuts = [(order[int(len(order) * 0.75)], None)]
    else:
        q = [1 - 0.1 * (k - i) for i in range(k)]  # e.g. k=3 -> windows starting at 70%, 80%, 90%
        cuts = [(order[int(len(order) * a)], order[min(int(len(order) * (a + 0.1)), len(order) - 1)]) for a in q]
    out = []
    lx = latest_exit_time(train)
    for start, end in cuts:
        tr = train[(lx < start).to_numpy()]
        va = train[(train.index >= start) & ((train.index < end) if end is not None else True)]
        out.append((tr, va))
    return out


def select_gates(train: pd.DataFrame, cfg: WFConfig, as_of: pd.Timestamp, seed: int = 0) -> tuple[float, float | None]:
    """Choose (EV threshold, max cost) by total R on inner validation windows (summed over windows)."""
    thresholds = [t for t in cfg.thresholds if t >= cfg.ev_floor] or [cfg.ev_floor]
    default = (max(0.1, cfg.ev_floor), cfg.max_cost_grid[0])
    scores: dict[tuple, list[float]] = {}
    for tr, va in _inner_windows(train, cfg.inner_splits):
        if len(tr) < 500 or len(va) < 100:
            continue
        w = recency_weights(tr.index, va.index.min(), cfg.half_life_days)
        pol = Policy.learn(tr, weights=w, max_same_side=cfg.max_same_side)
        va = MetaModel.fit(tr, seed, ev_mode=cfg.ev_mode, calibrate=cfg.calibrate, weights=w).score(va)
        for thr in thresholds:
            for cost in cfg.max_cost_grid:
                pol.min_ev_r, pol.max_cost_r = thr, cost
                sim = simulate(va, pol, True, True, cfg.daily_loss_r)
                taken = sim[sim["action"] != PASS]
                scores.setdefault((thr, cost), []).append((taken["r"].sum(), len(taken)))
    best, best_r = default, -np.inf
    for key, vals in scores.items():
        total_r, n = sum(v[0] for v in vals), sum(v[1] for v in vals)
        if n >= cfg.min_trades and total_r > best_r:
            best, best_r = key, total_r
    return best


def fit_fold(train: pd.DataFrame, cfg: WFConfig, as_of: pd.Timestamp, seed: int = 0) -> FoldModel:
    if cfg.rolling_days:
        train = train[train.index >= as_of - pd.Timedelta(days=cfg.rolling_days)]
    w = recency_weights(train.index, as_of, cfg.half_life_days)
    exits = learn_exits(train, cfg, w)
    train = apply_exit(train, exits) if exits else train.assign(exit_policy="fixed")
    thr, cost = select_gates(train, cfg, as_of, seed)
    policy = Policy.learn(train, weights=w, min_ev_r=thr, max_cost_r=cost, exits=exits,
                          max_same_side=cfg.max_same_side, sizing=cfg.sizing)
    meta = MetaModel.fit(train, seed, ev_mode=cfg.ev_mode, calibrate=cfg.calibrate, weights=w)
    return FoldModel(policy, meta, exits, {"threshold": thr, "max_cost_r": cost, "train_setups": int(len(train)),
                                           "enabled_cells": len(policy.enabled)})


# ---------------------------------------------------------------------------------------------------------------

def run(ds_raw: pd.DataFrame, frames: dict, cfg: WFConfig = WFConfig(), variants=tuple(VARIANTS),
        min_train_days: int = 90, seed: int = 0, fold_months: int = 1,
        test_from: pd.Timestamp | None = None, test_until: pd.Timestamp | None = None):
    """Walk forward. test_until excludes later data entirely (training and testing): keep a holdout untouched.
    test_from skips evaluating earlier folds (they still train on everything before them)."""
    if test_until is not None:
        ds_raw = ds_raw[ds_raw.index < test_until]
    first = ds_raw.index.min().normalize() + pd.Timedelta(days=min_train_days)
    first_month = pd.Timestamp(year=first.year, month=first.month, day=1, tz="UTC") + pd.offsets.MonthBegin(1)
    starts = pd.date_range(first_month, ds_raw.index.max(), freq=f"{fold_months}MS")
    if test_from is not None:
        starts = starts[starts >= test_from]
    lx_all = latest_exit_time(ds_raw)
    results, folds, last = [], [], None
    for i, fs in enumerate(starts):
        fe = starts[i + 1] if i + 1 < len(starts) else ds_raw.index.max() + pd.Timedelta(seconds=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rm = RegimeModel.fit({k: v[v["close_time"] < fs] for k, v in frames.items()}, n_init=3, seed=seed)
        regimes = {k: rm.filter(v) for k, v in frames.items()}
        in_scope = (ds_raw.index < fe) & ((lx_all < fs) | (ds_raw.index >= fs)).to_numpy()
        ds = attach_regime(ds_raw[in_scope], regimes)
        lx = latest_exit_time(ds)
        train = ds[(lx < fs).to_numpy()]  # purge: only outcomes known before the fold (under every exit policy)
        test = ds[ds.index >= fs]
        if len(train) < 1000 or test.empty:
            continue
        fm = fit_fold(train, cfg, fs, seed)
        test = apply_exit(test, fm.exits) if fm.exits else test.assign(exit_policy="fixed")
        days = (fe - fs).days
        fold = {"fold": str(fs.date()), "test_setups": len(test), "states": rm.names, **fm.info,
                "exits": fm.exits}
        meta_shuf = None
        if "shuffled_meta" in variants:
            tr_fit = apply_exit(train, fm.exits) if fm.exits else train
            shuffled = tr_fit.assign(win=np.random.default_rng(seed + i).permutation(tr_fit["win"].to_numpy()))
            meta_shuf = MetaModel.fit(shuffled, seed, ev_mode=cfg.ev_mode)
        for v in variants:
            vc = VARIANT_CFG[v]
            scored = (meta_shuf if v == "shuffled_meta" else fm.meta).score(test)
            sim = simulate(scored, fm.policy, vc.use_regime, vc.use_meta, cfg.daily_loss_r).assign(
                variant=v, fold=fold["fold"])
            results.append(sim)
            fold[v] = metrics(sim[sim["action"] != PASS], fm.policy.risk_per_trade, days)
        folds.append(fold)
        last = fm
        sys_m = fold.get("regime_meta", {})
        logger.info("[{}] fold {}: thr={} cost={} cells={} exits={} | system {:+.1f}R ({} trades)", cfg.name,
                    fold["fold"], fm.info["threshold"], fm.info["max_cost_r"], fm.info["enabled_cells"],
                    fm.exits or "fixed", sys_m.get("total_r", 0), sys_m.get("trades", 0))
    return pd.concat(results) if results else pd.DataFrame(), folds, last


def summary(rows: pd.DataFrame, folds: list[dict], variant: str = "regime_meta") -> dict:
    """Headline numbers + per-year R + calibration for one variant."""
    sysr = rows[rows["variant"] == variant]
    trades = sysr[sysr["action"] != PASS]
    days = max(1, sum(f["test_setups"] > 0 for f in folds)) * 91
    m = metrics(trades, 0.01, None)
    yr = trades.groupby(trades["exit_time"].dt.year)["r"].sum() if len(trades) else pd.Series(dtype=float)
    gated = sysr[~sysr["reason"].isin(["regime_off", "cost_too_high"])]
    calib = None
    if len(gated) >= 50 and "ev_r" in gated:
        b = pd.qcut(gated["ev_r"], 5, duplicates="drop")
        calib = gated.groupby(b, observed=True).agg(pred=("ev_r", "mean"), real=("r", "mean"), n=("r", "size"))
        calib = calib.round(3).reset_index(drop=True).to_dict("records")
    return {**m, "years": {int(k): round(float(v), 2) for k, v in yr.items()},
            "years_positive": float((yr > 0).mean()) if len(yr) else 0.0, "calibration": calib, "approx_days": days}


def report(all_rows: pd.DataFrame, folds: list[dict]) -> str:
    lines = [f"Walk-forward: {len(folds)} folds\n"]
    tab = {}
    for v in all_rows["variant"].unique():
        rows = all_rows[all_rows["variant"] == v]
        tab[v] = metrics(rows[rows["action"] != PASS])
    lines.append(pd.DataFrame(tab).T.round(3).to_string())
    sys_rows = all_rows[all_rows["variant"] == "regime_meta"]
    trades = sys_rows[sys_rows["action"] != PASS]
    if not trades.empty:
        for title, keys in [("strategy x regime", ["strategy", "regime"]), ("symbol x side", ["symbol", "action"]),
                            ("exit policy x outcome", ["exit_policy", "outcome"])]:
            g = trades.groupby(keys)
            lines.append(f"\nSystem trades by {title}:")
            lines.append(pd.DataFrame({"trades": g.size(), "win_rate": g["win"].mean(), "avg_r": g["r"].mean(),
                                       "total_r": g["r"].sum()}).round(3).sort_values("total_r", ascending=False).to_string())
        g = trades.groupby(trades["exit_time"].dt.year)
        lines.append("\nSystem by year:")
        lines.append(pd.DataFrame({"trades": g.size(), "win_rate": g["win"].mean(), "avg_r": g["r"].mean(),
                                   "total_r": g["r"].sum()}).round(3).to_string())
    lines.append("\nPASS reasons (system): " + json.dumps(sys_rows["reason"].replace("", "TRADED").value_counts().to_dict()))
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-months", type=int, default=1)
    ap.add_argument("--tfs", default="5m,15m,1h", help="timeframes to trade")
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}",
               filter=lambda r: "USDT.P" not in r["message"])
    pd.set_option("display.width", 200)
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        ds_raw = build_dataset(conn, s, tfs=tuple(a.tfs.split(",")))
        frames = load_regime_frames(conn, s.symbols)
    logger.info("dataset {} setups {} -> {}", len(ds_raw), ds_raw.index.min(), ds_raw.index.max())
    rows, folds, _ = run(ds_raw, frames, fold_months=a.fold_months)
    print(report(rows, folds))
    rows.to_parquet("data/wf_rows.parquet")
    with open("data/wf_folds.json", "w") as f:
        json.dump(folds, f, indent=2, default=str)


if __name__ == "__main__":
    main()
