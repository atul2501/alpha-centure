"""Setup dataset: every strategy setup on every (perp symbol, timeframe), with its features at the signal
bar and its labeled outcome. This is the training table for the meta model and the input to backtests."""

from datetime import datetime

import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import PERP_SUFFIX, Settings
from alpha.features.build import CORE_COLS, RICH_COLS, build_frame
from alpha.strategies.labels import EXIT_MENU, Costs, Entry, ExitPolicy, label_multi
from alpha.strategies.playbook import PLAYBOOK, all_setups

TRADE_TFS = ("5m", "15m", "1h")


def setup_features(f: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Geometry of each setup, normalized so it is comparable across symbols and timeframes."""
    # All positional (.to_numpy): signal times repeat across setups, and index alignment would multiply rows.
    at = lambda col: f[col].reindex(s.index).to_numpy()
    close, atr, side = at("close"), at("atr"), s["side"].to_numpy()
    risk = np.abs(close - s["stop"].to_numpy())
    return pd.DataFrame({
        "risk_atr": risk / atr,
        "rr": np.abs(s["target"].to_numpy() - close) / risk,
        "dist_vwap_atr": (close - at("vwap")) / atr * side,
        "dist_ema50_atr": (close - at("ema50")) / atr * side,
        "trend_align": at("h4_trend") * side,
    }, index=s.index)


def with_features(rows: pd.DataFrame, f: pd.DataFrame, symbol: str, tf: str, cols: list[str]) -> pd.DataFrame:
    """Attach bar features + setup geometry to setup rows. Used identically by research and the live predictor."""
    out = rows.copy()
    feats = f[cols + ["close_time", "close"]].reindex(rows.index)
    geo = setup_features(f, rows)
    for c in feats.columns:
        out[c if c != "close" else "close_px"] = feats[c].to_numpy()
    for c in geo.columns:
        out[c] = geo[c].to_numpy()
    out["symbol"], out["tf"] = symbol, tf
    return out


def build_dataset(conn: psycopg.Connection, settings: Settings, tfs=TRADE_TFS, start: datetime | None = None,
                  end: datetime | None = None, rich: bool = False, costs: Costs = Costs(),
                  exits: dict[str, ExitPolicy] | None = None, entry: Entry = Entry()) -> pd.DataFrame:
    """Labeled setups for every perp symbol x timeframe. With `exits`, every setup is labeled under each exit
    policy (suffixed columns, see labels.apply_exit); the 'fixed' policy fills the default label columns."""
    exits = exits or {"fixed": EXIT_MENU["fixed"]}
    cols = CORE_COLS + (RICH_COLS if rich else [])
    parts = []
    for tf in tfs:
        if not any(tf in st.timeframes for st in PLAYBOOK):
            continue
        for sym in settings.symbols:
            symbol = sym + PERP_SUFFIX
            f = build_frame(conn, symbol, tf, start, end, rich=rich)
            if f.empty:
                continue
            s = all_setups(f, tf)
            if s.empty:
                continue
            lab = label_multi(s, f, costs, exits, entry, primary="fixed" if "fixed" in exits else next(iter(exits)))
            if lab.empty:
                continue
            parts.append(with_features(lab, f, symbol, tf, cols))
            logger.info("{} {}: {} bars, {} setups, {} labeled", symbol, tf, len(f), len(s), len(lab))
    if not parts:
        return pd.DataFrame()
    ds = pd.concat(parts)
    ds.index.name = "signal_time"
    return ds.sort_index()


def cached_dataset(conn, settings: Settings, tag: str, refresh: bool = False, **kw) -> pd.DataFrame:
    """build_dataset with a parquet cache under data/cache/<tag>.parquet (research only)."""
    from pathlib import Path

    path = Path("data/cache") / f"{tag}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    ds = build_dataset(conn, settings, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(path)
    return ds


def summarize(ds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = ds.groupby(by)
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate": g["win"].mean(),
        "avg_r": g["r"].mean(),
        "exp_pct": g["pnl"].mean() * 100,
        "total_pct": g["pnl"].sum() * 100,
        "pf": g["pnl"].apply(lambda p: p[p > 0].sum() / -p[p < 0].sum() if (p < 0).any() else np.inf),
        "avg_bars": g["bars_held"].mean(),
    })
    return out.round(3).sort_values("avg_r", ascending=False)
