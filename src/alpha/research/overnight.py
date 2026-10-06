"""Overnight improvement research (plan of 2026-10-06; rules in data/experiments/overnight_preregistration.json).

    uv run python -m alpha.research.overnight

Every candidate runs through the same pipeline as the live P6 book (ridge forecast -> inverse-vol weights -> 20% vol
target, 3x cap, 0.5x per coin -> 1% no-trade band -> costs: 60% maker / 40% taker, funding) with quarterly
walk-forward retraining, from 2021-01 to today. It is judged on:
    DEV      2021-01 -> 2024-07 (gates)
    INFO     2024-07 -> today   (already seen by P6: shown for information only)
    UNSEEN-9 the same candidate on ATOM, DOGE, DOT, NEAR, OP, ARB, WLD, CAKE, POL (never used by any model), with
             models trained on the 14 coins only; one look per candidate, recorded in the lockbox.
Nothing here touches the running paper engine.
"""

import json
import sys
from pathlib import Path

import lightgbm  # noqa: F401  (imported early: xgboost/catboost and lightgbm share OpenMP on macOS)
import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.research import signals as sig
from alpha.research.models import OOS_START, feature_frame, target
from alpha.research.panel import cached_panel, wide
from alpha.research.phase4 import apply_band_and_stop, momentum_scores, vol_target
from alpha.research.portfolio_sim import simulate, to_weights
from alpha.research.scorecard import deflated_sharpe, pbo_cscv
from alpha.research.screen import symbol_costs
from alpha.research.splits import DEV_END, open_lockbox
from alpha.strategy import p6

OUT = Path("data/experiments")
EXTRA9 = ["ATOMUSDT", "DOGEUSDT", "DOTUSDT", "NEARUSDT", "OPUSDT", "ARBUSDT", "WLDUSDT", "CAKEUSDT", "POLUSDT"]
LAST12 = pd.Timestamp("2025-10-01", tz="UTC")
LOCKBOX = OUT / "lockbox_unseen9.jsonl"


# ---------------------------------------------------------------------------------------------------------------
# data and costs

def subset(panel: pd.DataFrame, symbols: list[str], untradable: tuple[str, ...] = ()) -> pd.DataFrame:
    p = panel[panel.index.get_level_values("symbol").isin(symbols)].copy()
    if untradable:
        p.loc[p.index.get_level_values("symbol").isin(untradable), "eligible"] = False
    return p


def costs_wide(conn, panel: pd.DataFrame) -> pd.DataFrame:
    """symbol_costs, with coins that have no live spread history given the widest live spread (conservative)."""
    c = symbol_costs(conn, panel)
    live = {r[0] for r in conn.execute("SELECT DISTINCT split_part(symbol, '.', 1) FROM book_tick").fetchall()}
    missing = [s for s in c.index if s not in live]
    if missing:
        worst = c.loc[[s for s in c.index if s in live], "spread_bps"].max()
        add = 0.5 * (worst - c.loc[missing, "spread_bps"])
        c.loc[missing, "spread_bps"] = worst
        c.loc[missing, "taker_bps"] = c.loc[missing, "taker_bps"] + add
    return c


# ---------------------------------------------------------------------------------------------------------------
# features

def residual_momentum(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """X2: momentum on returns after removing each coin's 90-day BTC beta (beta known before each hour)."""
    ret = wide(panel, "ret")
    el = wide(panel, "eligible").fillna(False).astype(bool)
    btc = ret["BTCUSDT"]
    n = 2160
    cov = ret.rolling(n, min_periods=n // 2).cov(btc)
    beta = cov.div(btc.rolling(n, min_periods=n // 2).var(), axis=0).shift(1)
    resid = ret - beta.mul(btc, axis=0)
    cum = resid.fillna(0).cumsum()
    rvol = resid.rolling(sig.VOL_WINDOW, min_periods=48).std()
    return {f"rmom_{L}": sig._xs_rank((cum - cum.shift(L)) / (rvol * np.sqrt(L)), el) for L in (168, 336, 720)}


def features(panel: pd.DataFrame, extra: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    scores = sig.compute(panel)
    X, vol = feature_frame(panel, scores)
    if extra:
        add = pd.concat({k: v.stack(future_stack=True) for k, v in extra.items()}, axis=1)
        add.index.names = ["time", "symbol"]
        X = X.join(add)
    return X, target(panel, vol, p6.H)


# ---------------------------------------------------------------------------------------------------------------
# models (same rows / folds as P6's ridge)

def _train_rows(X, y, fs):
    times = X.index.get_level_values("time")
    mask = (times + pd.Timedelta(hours=2 * p6.H) <= fs) & (times.hour % 4 == 0)
    tr = X[mask].assign(y=y[mask]).dropna(subset=["y"])
    return tr[list(X.columns)], tr["y"]


def fit_predict(model: str, X_tr, y_tr, X_ap, fs):
    if model == "ridge":
        b = p6.fit(X_tr, y_tr, fs)
        return b.model.predict((X_ap[b.cols].fillna(0.0) - b.mu) / b.sd)
    Xt, yt = _train_rows(X_tr, y_tr, fs)
    if model == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.7,
                             colsample_bytree=0.7, n_jobs=4, random_state=0).fit(Xt, yt)
        return m.predict(X_ap[Xt.columns])
    if model == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.03, random_seed=0, verbose=False,
                              thread_count=4).fit(Xt, yt)
        return m.predict(X_ap[Xt.columns])
    raise ValueError(model)


def walk_scores(X_tr, y_tr, X_ap, model: str = "ridge", start=OOS_START, end=None) -> pd.DataFrame:
    end = end or pd.Timestamp.now(tz="UTC").floor("h")
    folds = pd.date_range(start, end, freq="QS", inclusive="left")
    t_ap = X_ap.index.get_level_values("time")
    parts = []
    for i, fs in enumerate(folds):
        fe = folds[i + 1] if i + 1 < len(folds) else end
        rows = X_ap[(t_ap >= fs) & (t_ap < fe)]
        if rows.empty:
            continue
        parts.append(pd.Series(fit_predict(model, X_tr, y_tr, rows, fs), index=rows.index))
    return pd.concat(parts).unstack("symbol")


# ---------------------------------------------------------------------------------------------------------------
# the book (replicates phase4.run_config for the ridge book, with the candidate hooks)

def guard_reversal(panel) -> pd.Series:
    """I3a: 0.5 for 7 days after BTC's 7-day move disagrees with its 30-day trend by more than 2 sigma."""
    r = wide(panel, "ret")["BTCUSDT"].fillna(0)
    cum = r.cumsum()
    r7, r30 = cum - cum.shift(168), cum - cum.shift(720)
    sd7 = r.rolling(sig.VOL_WINDOW, min_periods=48).std() * np.sqrt(168)
    hit = (np.sign(r7) != np.sign(r30)) & (r7.abs() > 2 * sd7)
    active = hit.astype(float).rolling(168, min_periods=1).max().fillna(0)
    return 1 - 0.5 * active


def guard_dispersion(panel) -> pd.Series:
    """I3b: gross x min(1, dispersion / its trailing-365-day median)."""
    ret = wide(panel, "ret").fillna(0)
    el = wide(panel, "eligible").fillna(False).astype(bool)
    cum = ret.cumsum()
    disp = (cum - cum.shift(720)).where(el).std(axis=1)
    med = disp.rolling(365 * 24, min_periods=90 * 24).median()
    return (disp / med).clip(upper=1.0).fillna(1.0)


def book(panel, score, costs, *, h=p6.H, scale=None, blend=None, fill=1.0, maker=p6.MAKER_SHARE_ASSUMED,
         cost_mult=1.0, start=OOS_START, end=None):
    ret = wide(panel, "ret")
    end = end or ret.index.max() + pd.Timedelta(hours=1)
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    el = wide(panel, "eligible").fillna(False).astype(bool)
    t = ret.index[(ret.index >= start) & (ret.index < end)]
    w = to_weights(score.reindex(t).reindex(columns=ret.columns), vol.loc[t], el.loc[t], "ts", every=h)
    w = vol_target(w, ret.loc[t])
    if blend is not None:
        w = 0.5 * (w + blend.reindex_like(w).fillna(0.0))
    if scale is not None:
        w = w.mul(scale.reindex(t).fillna(1.0).to_numpy(), axis=0)
    w = apply_band_and_stop(w, ret.loc[t], vol.loc[t], p6.BAND, None, h)
    if fill < 1.0:  # passive-only: each rebalance gets only `fill` of the way to target
        hours = ((w.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).to_numpy()
        W, out, prev = w.to_numpy(), np.zeros_like(w.to_numpy()), np.zeros(w.shape[1])
        for i in range(len(W)):
            if hours[i] % h == 0:
                prev = prev + fill * (W[i] - prev)
            out[i] = prev
        w = pd.DataFrame(out, index=w.index, columns=w.columns)
    per_side = cost_mult * (maker * costs["maker_bps"] + (1 - maker) * costs["taker_bps"])
    res = simulate(w, ret.loc[t], wide(panel, "funding_rate").loc[t], per_side.reindex(w.columns))
    return res, w


def rules_weights(panel, start=OOS_START) -> pd.DataFrame:
    """P3's pre-band weights (XS + TS momentum rules, 50/50, vol-targeted)."""
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    el = wide(panel, "eligible").fillna(False).astype(bool)
    t = ret.index[ret.index >= start]
    xs, ts = momentum_scores(panel, 1.0)
    w = (to_weights(xs.loc[t], vol.loc[t], el.loc[t], "xs", every=p6.H) +
         to_weights(ts.loc[t], vol.loc[t], el.loc[t], "ts", every=p6.H)) / 2
    return vol_target(w, ret.loc[t])


# ---------------------------------------------------------------------------------------------------------------
# evaluation

def money(daily: pd.Series, start_equity: float = 30_000.0) -> float:
    return float(start_equity * (1 + daily.fillna(0)).prod())


def summarize(name: str, res, w, costs_x15_net: pd.Series, n_trials: int, coins: list[str] | None = None) -> dict:
    d = res.daily()
    dev, info, last12 = d[d.index < DEV_END], d[d.index >= DEV_END], d[d.index >= LAST12]
    n = dev["net"]
    years = n.groupby(n.index.year).sum()
    sym = res.by_symbol["net"] if coins is None else res.by_symbol["net"].reindex(coins)
    pos = sym.clip(lower=0)
    eq = n.cumsum()
    dchg = (w.diff().abs() > 1e-9)
    months_dev = max(1.0, (dev.index.max() - dev.index.min()).days / 30.44)
    x15 = costs_x15_net
    return {
        "name": name,
        "dev_30k": money(n), "dev_net_pct": float(n.sum()), "dev_sharpe": float(n.mean() / n.std() * np.sqrt(365)),
        "dev_max_dd": float((eq.cummax() - eq).max()), "dev_win_days": float((n > 0).mean()),
        "dev_pf": float(n[n > 0].sum() / -n[n < 0].sum()) if (n < 0).any() else np.inf,
        "dev_net_x15": float(x15[x15.index < DEV_END].sum()),
        "dev_years_pos": float((years > 0).mean()), "years": {int(k): round(float(v), 3) for k, v in years.items()},
        "dev_coins_pos": float((sym > 0).mean()), "dev_max_coin_share": float(pos.max() / pos.sum()) if pos.sum() > 0 else 1.0,
        "dev_dsr": deflated_sharpe(n, n_trials),
        "trades_per_month": float(dchg[dchg.index < DEV_END].sum().sum() / months_dev),
        "costs_per_month_30k": float(dev["cost"].sum() / months_dev * 30_000),
        "info_30k": money(info["net"]), "info_sharpe": float(info["net"].mean() / info["net"].std() * np.sqrt(365)),
        "last12_30k": money(last12["net"]), "last12_win_days": float((last12["net"] > 0).mean()),
    }


def dev_gate(r: dict) -> tuple[bool, list[str]]:
    checks = {"net > 0": r["dev_net_pct"] > 0, "net > 0 at costs x1.5": r["dev_net_x15"] > 0,
              ">= 60% years positive": r["dev_years_pos"] >= 0.6, ">= 60% coins positive": r["dev_coins_pos"] >= 0.6,
              "no coin > 40% of profit": r["dev_max_coin_share"] <= 0.4}
    return all(checks.values()), [k for k, v in checks.items() if not v]


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    s = get_settings()
    core = s.symbols
    pool = core + EXTRA9
    prior = json.loads((OUT / "phase4_trials.json").read_text())["trials"]
    with psycopg.connect(s.database_url) as conn:
        wide_p = cached_panel(conn, pool, tag="panel_1h_wide_v1")
        costs = costs_wide(conn, wide_p)
    p14 = subset(wide_p, core)
    p23 = wide_p
    p9 = subset(wide_p, EXTRA9 + ["BTCUSDT"], untradable=("BTCUSDT",))  # BTC only feeds btc_trend / guards
    logger.info("panels: 14 {} rows, 23 {} rows, 9 {} rows", len(p14), len(p23), len(p9))

    rm14, rm9, rm23 = residual_momentum(p14), residual_momentum(p9), residual_momentum(p23)
    X14, y14 = features(p14)
    X9, _ = features(p9)
    X23, y23 = features(p23)
    X14r, y14r = features(p14, rm14)
    X9r, _ = features(p9, rm9)
    X9 = X9[X9.index.get_level_values("symbol").isin(EXTRA9)]
    X9r = X9r[X9r.index.get_level_values("symbol").isin(EXTRA9)]
    logger.info("features ready")

    sc = {"ridge14": walk_scores(X14, y14, X14), "ridge9": walk_scores(X14, y14, X9),
          "ridge23": walk_scores(X23, y23, X23),
          "xgb14": walk_scores(X14, y14, X14, "xgboost"), "xgb9": walk_scores(X14, y14, X9, "xgboost"),
          "cat14": walk_scores(X14, y14, X14, "catboost"), "cat9": walk_scores(X14, y14, X9, "catboost"),
          "res14": walk_scores(X14r, y14r, X14r), "res9": walk_scores(X14r, y14r, X9r)}
    logger.info("walk-forward scores ready")

    g14 = {"rev": guard_reversal(p14), "disp": guard_dispersion(p14)}
    g9 = {"rev": guard_reversal(p9), "disp": guard_dispersion(p9)}
    r14, r9 = rules_weights(p14), rules_weights(p9)
    # name -> (kwargs on the 14-coin panel, kwargs on the unseen 9)
    C = {
        "CURRENT_P6": (dict(panel=p14, score=sc["ridge14"]), dict(panel=p9, score=sc["ridge9"])),
        "I2_23coins": (dict(panel=p23, score=sc["ridge23"]), None),
        "I3a_reversal_guard": (dict(panel=p14, score=sc["ridge14"], scale=g14["rev"]),
                               dict(panel=p9, score=sc["ridge9"], scale=g9["rev"])),
        "I3b_dispersion_guard": (dict(panel=p14, score=sc["ridge14"], scale=g14["disp"]),
                                 dict(panel=p9, score=sc["ridge9"], scale=g9["disp"])),
        "I4_xgboost": (dict(panel=p14, score=sc["xgb14"]), dict(panel=p9, score=sc["xgb9"])),
        "I4_catboost": (dict(panel=p14, score=sc["cat14"]), dict(panel=p9, score=sc["cat9"])),
        "X1_weekly": (dict(panel=p14, score=sc["ridge14"], h=168), dict(panel=p9, score=sc["ridge9"], h=168)),
        "X2_residual_momentum": (dict(panel=p14, score=sc["res14"]), dict(panel=p9, score=sc["res9"])),
        "X3_ensemble_rules": (dict(panel=p14, score=sc["ridge14"], blend=r14),
                              dict(panel=p9, score=sc["ridge9"], blend=r9)),
        "X4_maker_only": (dict(panel=p14, score=sc["ridge14"], maker=1.0, fill=0.6),
                          dict(panel=p9, score=sc["ridge9"], maker=1.0, fill=0.6)),
    }
    n_trials = prior + len(C) - 1
    rows, dev_daily, daily_all = [], {}, {}
    for name, (kw, kw9) in C.items():
        res, w = book(costs=costs, **kw)
        res15, _ = book(costs=costs, cost_mult=1.5, **kw)
        x15 = res15.daily()["net"]
        r = summarize(name, res, w, x15, n_trials, coins=core if name != "I2_23coins" else None)
        ok, fails = dev_gate(r)
        r.update(dev_gate=ok, dev_fails="; ".join(fails))
        # UNSEEN-9: one look, recorded
        open_lockbox(name, "UNSEEN-9", LOCKBOX)
        if kw9 is None:  # the 23-coin book: the 9 new coins' own contribution inside it, at costs x1.5
            u = float(res15.by_symbol.loc[[c for c in EXTRA9 if c in res15.by_symbol.index], "net"].sum())
            u_30k = float("nan")
        else:
            u9, _ = book(costs=costs, cost_mult=1.5, **kw9)
            u = float(u9.daily()["net"].sum())
            u_30k = money(book(costs=costs, **kw9)[0].daily()["net"])
        r.update(unseen9_net_x15=u, unseen9_30k=u_30k, unseen9_pass=u > 0)
        rows.append(r)
        d = res.daily()["net"]
        dev_daily[name] = d[d.index < DEV_END]
        daily_all[name] = d
        logger.info("{}: DEV $30k->{:,.0f} sharpe {:.2f} | last12 $30k->{:,.0f} | unseen9 {:+.3f}", name, r["dev_30k"],
                    r["dev_sharpe"], r["last12_30k"], u)
    pbo = pbo_cscv(pd.DataFrame(dev_daily).fillna(0.0), n_blocks=10)
    base = next(r for r in rows if r["name"] == "CURRENT_P6")
    for r in rows:
        r["pbo_all_candidates"] = pbo
        r["better_than_current"] = bool(r["name"] != "CURRENT_P6" and r["dev_gate"] and r["unseen9_pass"]
                                        and r["dev_net_pct"] > base["dev_net_pct"] and r["dev_sharpe"] > base["dev_sharpe"])
    df = pd.DataFrame(rows)
    df.to_json(OUT / "overnight_results.json", orient="records", indent=2)
    pd.DataFrame(daily_all).to_parquet(OUT / "overnight_daily.parquet")
    (OUT / "overnight_trials.json").write_text(json.dumps({"trials": n_trials, "pbo": pbo}))
    pd.set_option("display.width", 250)
    cols = ["name", "dev_30k", "dev_sharpe", "dev_max_dd", "dev_gate", "info_30k", "last12_30k", "unseen9_30k",
            "unseen9_net_x15", "unseen9_pass", "better_than_current", "trades_per_month", "costs_per_month_30k"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"PBO across candidates: {pbo:.2f}; trials counted: {n_trials}")


if __name__ == "__main__":
    main()
