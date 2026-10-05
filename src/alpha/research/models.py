"""Phase 3: model ladder on the hourly perp panel, purged walk-forward inside DEV.

    uv run python -m alpha.research.models

Target: forward H-hour log return / (vol * sqrt(H)), per coin. Decision every H hours (H = 24 or 72).
Folds: quarterly from 2021-01-01 to DEV_END; each fold trains on rows whose target window ended before the fold
start minus an embargo of H hours (purge), then predicts the fold. Nothing after DEV_END is loaded into training or
testing.

Models (all map their score to weights with the same inverse-vol, gross-1 rule and the same costs):
    M0  rules: 0.5 * cross-sectional momentum blend + 0.5 * time-series trend blend (fixed, not fitted)
    M1  ridge regression on standardized features
    M2  LightGBM (shallow, 3 seeds averaged) on the same features + BTC HMM regime probabilities
    M2_shuffled  M2 trained on target shuffled within each fold's training set (must show no edge)
"""

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg
from loguru import logger
from sklearn.linear_model import Ridge

from alpha.config import get_settings
from alpha.features.build import load_candles
from alpha.regime.hmm import REGIMES, RegimeModel, regime_features
from alpha.research import signals as sig
from alpha.research.panel import cached_panel, wide
from alpha.research.portfolio_sim import simulate, summarize, to_weights
from alpha.research.screen import symbol_costs
from alpha.research.splits import DEV_END

OOS_START = pd.Timestamp("2021-01-01", tz="UTC")
OUT = Path("data/experiments")
FEATURES = ["xsmom_24", "xsmom_72", "xsmom_168", "xsmom_336", "xsmom_720", "tsmom_72", "tsmom_168", "tsmom_336",
            "tsmom_720", "donchian_480", "donchian_1320", "carry", "xscarry", "basis", "oi_trend", "ls_contra",
            "flow_24", "vol_ratio", "btc_trend"]
REGIME_FEATS = [f"p_{r}" for r in REGIMES]
LGB = dict(objective="regression", learning_rate=0.03, n_estimators=250, num_leaves=15, min_child_samples=500,
           subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)


def feature_frame(panel: pd.DataFrame, scores: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(long features indexed by (time, symbol), vol time x symbol)."""
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    cols = {k: scores[k] for k in FEATURES if k in scores}
    cols["vol_ratio"] = ret.rolling(24, min_periods=12).std() / ret.rolling(720, min_periods=200).std()
    cols["btc_trend"] = pd.DataFrame({c: scores["tsmom_720"]["BTCUSDT"] for c in ret.columns})
    long = pd.concat({k: v.stack(future_stack=True) for k, v in cols.items()}, axis=1)
    long.index.names = ["time", "symbol"]
    return long, vol


def target(panel: pd.DataFrame, vol: pd.DataFrame, h: int) -> pd.Series:
    cum = wide(panel, "ret").fillna(0).cumsum()
    y = ((cum.shift(-h) - cum) / (vol * np.sqrt(h))).clip(-5, 5)
    s = y.stack(future_stack=True)
    s.index.names = ["time", "symbol"]
    return s


def btc_regimes(conn, start: pd.Timestamp) -> RegimeModel:
    c = load_candles(conn, "BTCUSDT.P", "4h")
    f = regime_features(c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RegimeModel.fit({"BTC": f[f["close_time"] < start]}, n_init=3, seed=0), f


def regime_probs(rm: RegimeModel, feats: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    p = rm.filter(feats)
    keep = ["close_time"] + [c for c in REGIME_FEATS if c in p]
    p = p[keep].sort_values("close_time")
    j = pd.merge_asof(pd.DataFrame({"t": hours + pd.Timedelta(hours=1)}), p, left_on="t", right_on="close_time",
                      direction="backward")
    return j[[c for c in REGIME_FEATS if c in j]].set_index(hours)


def m0_score(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    xs = (scores["xsmom_168"] + scores["xsmom_336"] + scores["xsmom_720"]) / 3
    ts = (scores["tsmom_336"] / 3 + scores["tsmom_720"] / 3 + scores["donchian_480"]) / 3
    return 0.5 * xs + 0.5 * ts


def walk_forward(conn, panel, X: pd.DataFrame, y: pd.Series, h: int, model: str, seed: int = 0,
                 start: pd.Timestamp = OOS_START, end: pd.Timestamp = DEV_END) -> pd.DataFrame:
    """Out-of-sample predictions (time x symbol) for start..end, retrained quarterly on data known at each fold."""
    folds = pd.date_range(start, end, freq="QS", inclusive="left")
    preds = []
    times = X.index.get_level_values("time")
    sample = (times.hour % 4 == 0)  # thin training rows: neighbouring hours are near duplicates
    for i, fs in enumerate(folds):
        fe = folds[i + 1] if i + 1 < len(folds) else end
        Xf = X
        if model.startswith("M2"):
            rm, feats = btc_regimes(conn, fs)
            hours = pd.DatetimeIndex(times.unique())
            rp = regime_probs(rm, feats, hours)
            Xf = X.join(rp, on="time")
        train_mask = (times + pd.Timedelta(hours=2 * h) <= fs) & sample  # purge h + embargo h
        test_mask = (times >= fs) & (times < fe)
        tr = Xf[train_mask].assign(y=y[train_mask]).dropna(subset=["y"])
        te = Xf[test_mask]
        feats_cols = [c for c in Xf.columns]
        if model == "M1":
            Xtr = tr[feats_cols].fillna(0.0)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            m = Ridge(alpha=100.0).fit((Xtr - mu) / sd, tr["y"])
            p = m.predict((te[feats_cols].fillna(0.0) - mu) / sd)
        else:
            yt = tr["y"].to_numpy()
            if model == "M2_shuffled":
                yt = np.random.default_rng(seed + i).permutation(yt)
            p = np.zeros(len(te))
            for s in range(3):
                m = lgb.LGBMRegressor(**LGB, random_state=seed + s).fit(tr[feats_cols], yt)
                p += m.predict(te[feats_cols]) / 3
        preds.append(pd.Series(p, index=te.index))
        logger.info("{} h={} fold {}: train {} rows, test {}", model, h, fs.date(), len(tr), len(te))
    return pd.concat(preds).unstack("symbol")


def evaluate(score: pd.DataFrame, panel, costs, h: int) -> tuple[dict, pd.Series]:
    t = score.index[(score.index >= OOS_START) & (score.index < DEV_END)]
    ret = wide(panel, "ret").loc[t]
    vol = wide(panel, "ret").rolling(sig.VOL_WINDOW, min_periods=48).std().loc[t]
    el = wide(panel, "eligible").fillna(False).astype(bool).loc[t]
    w = to_weights(score.reindex(t).reindex(columns=ret.columns), vol, el, "ts", every=h)
    res = simulate(w, ret, wide(panel, "funding_rate").loc[t], costs["taker_bps"])
    return summarize(res, h), res.daily()["net"]


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        panel = cached_panel(conn, s.symbols)
        costs = symbol_costs(conn, panel)
        panel = panel[panel.index.get_level_values("time") < DEV_END]  # DEV only from here on
        scores = sig.compute(panel)
        X, vol = feature_frame(panel, scores)
        rows, daily = [], {}
        for h in (24, 72):
            y = target(panel, vol, h)
            cands = {"M0": m0_score(scores)}
            for m in ("M1", "M2", "M2_shuffled"):
                cands[m] = walk_forward(conn, panel, X, y, h, m)
            for name, score in cands.items():
                summ, d = evaluate(score, panel, costs, h)
                key = f"{name}_h{h}"
                daily[key] = d
                rows.append({"model": key, **{k: v for k, v in summ.items() if k != "years"}, "years": summ["years"]})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "phase3_models.csv", index=False)
    pd.DataFrame(daily).to_parquet(OUT / "phase3_daily.parquet")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------
# Phase 3b: separate coin selection (market-neutral) from direction timing, against a long-only benchmark

def xs_target(y: pd.Series) -> pd.Series:
    """Target demeaned across coins at each time: what a dollar-neutral book can earn."""
    return y - y.groupby(level="time").transform("mean")


def evaluate_kind(score: pd.DataFrame, panel, costs, h: int, kind: str) -> tuple[dict, pd.Series]:
    t = score.index[(score.index >= OOS_START) & (score.index < DEV_END)]
    ret = wide(panel, "ret").loc[t]
    vol = wide(panel, "ret").rolling(sig.VOL_WINDOW, min_periods=48).std().loc[t]
    el = wide(panel, "eligible").fillna(False).astype(bool).loc[t]
    w = to_weights(score.reindex(t).reindex(columns=ret.columns), vol, el, kind, every=h)
    res = simulate(w, ret, wide(panel, "funding_rate").loc[t], costs["taker_bps"])
    return summarize(res, h), res.daily()["net"]


def decompose() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        panel = cached_panel(conn, s.symbols)
        costs = symbol_costs(conn, panel)
        panel = panel[panel.index.get_level_values("time") < DEV_END]
        scores = sig.compute(panel)
        X, vol = feature_frame(panel, scores)
        rows, daily = [], {}
        ones = scores["tsmom_720"].notna().astype(float)
        cands = {("long_only", "ts", 72): ones}
        xs0 = (scores["xsmom_168"] + scores["xsmom_336"] + scores["xsmom_720"]) / 3
        ts0 = (scores["tsmom_336"] / 3 + scores["tsmom_720"] / 3 + scores["donchian_480"]) / 3
        for h in (24, 72):
            y = target(panel, vol, h)
            cands[("M0_xs", "xs", h)] = xs0
            cands[("M0_ts", "ts", h)] = ts0
            cands[("M1_xs", "xs", h)] = walk_forward(conn, panel, X, xs_target(y), h, "M1")
        for (name, kind, h), score in cands.items():
            summ, d = evaluate_kind(score, panel, costs, h, kind)
            key = f"{name}_h{h}"
            daily[key] = d
            rows.append({"model": key, **{k: v for k, v in summ.items() if k != "years"}, "years": summ["years"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase3b_decompose.csv", index=False)
    pd.DataFrame(daily).to_parquet(OUT / "phase3b_daily.parquet")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
