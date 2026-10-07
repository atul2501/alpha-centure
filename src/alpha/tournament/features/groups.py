"""Feature groups. Every builder maps the raw dataset (alpha.tournament.data.dataset) to features known at the bar
close, using only rows <= t (rolling windows, never centered, never expanding statistics over the future).
Normalization that needs fitting (z-scoring for linear / deep models) happens per fold inside the model pipeline.

The REGIME group is not built here: regime features must be fit per fold on training data, so hybrids produce them
inside the walk-forward (alpha.tournament.models.hybrid).
"""

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from alpha.features.indicators import rsi

DAY = 96  # 15m bars per day
EPS = 1e-12


def _lr(s: pd.Series, n: int = 1) -> pd.Series:
    return np.log(s / s.shift(n))


def _z(s: pd.Series, n: int, minp: int | None = None) -> pd.Series:
    m = s.rolling(n, min_periods=minp or n // 2)
    return (s - m.mean()) / (m.std() + EPS)


def price(r: pd.DataFrame) -> pd.DataFrame:
    c = r["sol_close"]
    out = {f"ret_{n}": _lr(c, n) for n in (1, 2, 4, 8, 16, 32, 96)}
    rng = (r["sol_high"] - r["sol_low"]).replace(0, np.nan)
    out["clv"] = (2 * c - r["sol_high"] - r["sol_low"]) / rng      # close location in the bar's range
    out["body"] = (c - r["sol_open"]) / rng
    out["upper_wick"] = (r["sol_high"] - np.maximum(c, r["sol_open"])) / rng
    return pd.DataFrame(out)


def volume(r: pd.DataFrame) -> pd.DataFrame:
    v = np.log(r["sol_quote_volume"].replace(0, np.nan))
    return pd.DataFrame({
        "vol_z_96": _z(v, DAY), "vol_z_672": _z(v, 7 * DAY),
        "vol_chg_4": v.rolling(4).mean() - v.rolling(16).mean(),
        "trades_z_96": _z(np.log(r["sol_trades"].replace(0, np.nan)), DAY),
        "avg_trade_z": _z(np.log((r["sol_quote_volume"] / r["sol_trades"]).replace(0, np.nan)), 7 * DAY),
    })


def volatility(r: pd.DataFrame) -> pd.DataFrame:
    lr = _lr(r["sol_close"])
    hl = np.log(r["sol_high"] / r["sol_low"])
    rv16, rv96, rv672 = (lr.rolling(n, min_periods=n // 2).std() for n in (16, DAY, 7 * DAY))
    return pd.DataFrame({
        "rvol_16": rv16, "rvol_96": rv96, "rvol_672": rv672,
        "rvol_ratio_16_96": rv16 / (rv96 + EPS), "rvol_ratio_96_672": rv96 / (rv672 + EPS),
        "park_16": np.sqrt((hl**2).rolling(16, min_periods=8).mean() / (4 * np.log(2))),
        "range_z": _z(hl, DAY),
        "bb_width_80": 4 * r["sol_close"].rolling(80).std() / r["sol_close"].rolling(80).mean(),
    })


def momentum(r: pd.DataFrame) -> pd.DataFrame:
    c = r["sol_close"]
    lr = _lr(c)
    rv = lr.rolling(DAY, min_periods=48).std() + EPS
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c
    lo, hi = r["sol_low"].rolling(56).min(), r["sol_high"].rolling(56).max()
    return pd.DataFrame({
        "rsi_14": rsi(c, 14) / 100 - 0.5, "rsi_56": rsi(c, 56) / 100 - 0.5,
        "mom_z_16": _lr(c, 16) / (rv * 4), "mom_z_96": _lr(c, DAY) / (rv * np.sqrt(DAY)),
        "mom_z_384": _lr(c, 4 * DAY) / (rv * np.sqrt(4 * DAY)),
        "macd_hist": macd - macd.ewm(span=9, adjust=False).mean(),
        "stoch_56": (c - lo) / (hi - lo + EPS) - 0.5,
    })


def trend(r: pd.DataFrame) -> pd.DataFrame:
    c = r["sol_close"]
    out = {f"ema_gap_{n}": c / c.ewm(span=n, adjust=False).mean() - 1 for n in (20, 96, 384, 1344)}
    e96 = c.ewm(span=DAY, adjust=False).mean()
    out["ema96_slope"] = np.log(e96 / e96.shift(16))
    for n in (96, 384):
        lo, hi = r["sol_low"].rolling(n).min(), r["sol_high"].rolling(n).max()
        out[f"donchian_pos_{n}"] = (c - lo) / (hi - lo + EPS) - 0.5
    lr = _lr(c)
    # trend efficiency: |net move| / path length over a day (1 = straight line, 0 = pure chop)
    out["efficiency_96"] = lr.rolling(DAY).sum().abs() / (lr.abs().rolling(DAY).sum() + EPS)
    return pd.DataFrame(out)


def market_structure(r: pd.DataFrame) -> pd.DataFrame:
    c, h, l = r["sol_close"], r["sol_high"], r["sol_low"]
    t = r.index
    hod = (t.hour * 4 + t.minute // 15) / DAY
    out = {"hod_sin": np.sin(2 * np.pi * hod), "hod_cos": np.cos(2 * np.pi * hod),
           "dow_sin": np.sin(2 * np.pi * t.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * t.dayofweek / 7)}
    for n in (96, 672):
        out[f"dist_high_{n}"] = np.log(c / h.rolling(n).max())
        out[f"dist_low_{n}"] = np.log(c / l.rolling(n).min())
    hh = (h > h.shift(1)).astype(float)
    ll = (l < l.shift(1)).astype(float)
    out["hh_minus_ll_16"] = hh.rolling(16).sum() - ll.rolling(16).sum()
    lr = _lr(c)
    out["autocorr_96"] = lr.rolling(DAY).corr(lr.shift(1))
    return pd.DataFrame(out, index=r.index)


def order_flow(r: pd.DataFrame) -> pd.DataFrame:
    tb = r["sol_taker_buy_base"] / r["sol_volume"].replace(0, np.nan)
    out = {c: r[c] for c in ("flow_5", "flow_15", "flow_60", "flow_240", "cvd_slope_60", "buy_surprise_60",
                             "vol_burst_15", "trades_burst_15")}
    out["taker_buy_ratio"] = tb - 0.5
    out["taker_buy_z_96"] = _z(tb, DAY)
    delta = (2 * r["sol_taker_buy_base"] - r["sol_volume"]) * r["sol_close"]
    out["cvd_96_norm"] = delta.rolling(DAY).sum() / (r["sol_quote_volume"].rolling(DAY).sum() + EPS)
    return pd.DataFrame(out)


def order_book(r: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for k in range(1, 6):
        b, a = r[f"bid_{k}"], r[f"ask_{k}"]
        out[f"book_imb_{k}"] = (b - a) / (b + a)
    tot = np.log(r["bid_1"] + r["ask_1"])
    out["book_imb_1_chg_4"] = out["book_imb_1"] - out["book_imb_1"].shift(4)
    out["book_imb_1_chg_16"] = out["book_imb_1"] - out["book_imb_1"].shift(16)
    out["depth_1_z"] = _z(tot, 7 * DAY)
    out["depth_1_chg_4"] = tot - tot.shift(4)
    out["depth_slope"] = np.log((r["bid_5"] + r["ask_5"]) / (r["bid_1"] + r["ask_1"]))  # how far liquidity sits
    out["book_pressure"] = ((r["bid_2"] - r["bid_1"]) - (r["ask_2"] - r["ask_1"])) / (r["bid_2"] + r["ask_2"])
    return pd.DataFrame(out)


def funding(r: pd.DataFrame) -> pd.DataFrame:
    f = r["funding_last"]
    p = r["premium"]
    return pd.DataFrame({"funding_last": f * 1e4, "funding_z_90d": _z(f, 90 * DAY, 7 * DAY),
                         "premium_bps": p * 1e4, "premium_ma_32": p.rolling(32, min_periods=8).mean() * 1e4,
                         "premium_z_7d": _z(p, 7 * DAY)})


def open_interest(r: pd.DataFrame) -> pd.DataFrame:
    oi = np.log(r["oi_value"].replace(0, np.nan))
    return pd.DataFrame({"oi_chg_4": oi - oi.shift(4), "oi_chg_16": oi - oi.shift(16), "oi_chg_96": oi - oi.shift(DAY),
                         "oi_z_7d": _z(oi, 7 * DAY),
                         "oi_price_div": (oi - oi.shift(16)) * np.sign(_lr(r["sol_close"], 16))})


def derivatives(r: pd.DataFrame) -> pd.DataFrame:
    out = {c: np.log(r[c].replace(0, np.nan)) for c in ("toptrader_ls_count", "toptrader_ls_position", "ls_ratio",
                                                        "taker_ls_vol_ratio")}
    out["ls_ratio_chg_16"] = out["ls_ratio"] - out["ls_ratio"].shift(16)
    out["top_vs_all"] = out["toptrader_ls_position"] - out["ls_ratio"]
    return pd.DataFrame(out)


def _context(r: pd.DataFrame, tag: str) -> pd.DataFrame:
    c, s = r[f"{tag}_close"], r["sol_close"]
    lr, ls = _lr(c), _lr(s)
    out = {f"{tag}_ret_{n}": _lr(c, n) for n in (1, 4, 16, 96)}
    out[f"{tag}_rvol_96"] = lr.rolling(DAY, min_periods=48).std()
    cov = ls.rolling(7 * DAY, min_periods=DAY).cov(lr)
    beta = cov / (lr.rolling(7 * DAY, min_periods=DAY).var() + EPS)
    out[f"{tag}_beta_7d"] = beta
    out[f"{tag}_corr_96"] = ls.rolling(DAY, min_periods=48).corr(lr)
    out[f"sol_vs_{tag}_16"] = _lr(s, 16) - beta * _lr(c, 16)       # SOL idiosyncratic move
    out[f"sol_vs_{tag}_96"] = _lr(s, DAY) - beta * _lr(c, DAY)
    out[f"{tag}_ema_gap_384"] = c / c.ewm(span=384, adjust=False).mean() - 1
    return pd.DataFrame(out)


def btc_context(r: pd.DataFrame) -> pd.DataFrame:
    return _context(r, "btc")


def eth_context(r: pd.DataFrame) -> pd.DataFrame:
    return _context(r, "eth")


GROUPS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "PRICE": price, "VOLUME": volume, "VOLATILITY": volatility, "MOMENTUM": momentum, "TREND": trend,
    "MARKET_STRUCTURE": market_structure, "ORDER_FLOW": order_flow, "ORDER_BOOK": order_book, "FUNDING": funding,
    "OPEN_INTEREST": open_interest, "DERIVATIVES": derivatives, "BTC_CONTEXT": btc_context, "ETH_CONTEXT": eth_context,
}
# raw columns each group needs (dataset coverage report)
RAW_NEEDS = {
    "PRICE": ["sol_close"], "VOLUME": ["sol_quote_volume"], "VOLATILITY": ["sol_close"], "MOMENTUM": ["sol_close"],
    "TREND": ["sol_close"], "MARKET_STRUCTURE": ["sol_close"], "ORDER_FLOW": ["flow_60", "sol_taker_buy_base"],
    "ORDER_BOOK": ["bid_1", "ask_1", "bid_5", "ask_5"], "FUNDING": ["funding_last", "premium"],
    "OPEN_INTEREST": ["oi_value"], "DERIVATIVES": ["ls_ratio", "toptrader_ls_position"],
    "BTC_CONTEXT": ["btc_close"], "ETH_CONTEXT": ["eth_close"],
}
REGIME = "REGIME"  # built per fold by hybrids
NOT_TESTABLE = {"ORDER_BOOK_L2": "top-of-book spread / microprice / L2 OFI exist only since 2026-10-04 "
                                 "(orderbook_snap, book_tick): forward shadow only"}

# Named feature sets. 'core' has full DEV history; groups with late history get an eval_start in the runner.
FEATURE_SETS: dict[str, list[str]] = {
    "price": ["PRICE"],
    "price_volume": ["PRICE", "VOLUME"],
    "price_volume_flow": ["PRICE", "VOLUME", "ORDER_FLOW"],
    "technical": ["PRICE", "VOLUME", "VOLATILITY", "MOMENTUM", "TREND", "MARKET_STRUCTURE"],
    "core": ["PRICE", "VOLUME", "VOLATILITY", "MOMENTUM", "TREND", "MARKET_STRUCTURE", "ORDER_FLOW", "FUNDING",
             "BTC_CONTEXT", "ETH_CONTEXT"],
    "core_deriv": ["PRICE", "VOLUME", "VOLATILITY", "MOMENTUM", "TREND", "MARKET_STRUCTURE", "ORDER_FLOW", "FUNDING",
                   "BTC_CONTEXT", "ETH_CONTEXT", "OPEN_INTEREST", "DERIVATIVES"],
    "everything": list(GROUPS),
}
# compact set for sequence models (fewer channels; all full-history)
SEQ_COLUMNS = ["ret_1", "ret_4", "ret_16", "clv", "vol_z_96", "rvol_16", "rvol_ratio_16_96", "rsi_14", "mom_z_96",
               "ema_gap_96", "donchian_pos_96", "hod_sin", "hod_cos", "flow_15", "flow_60", "taker_buy_ratio",
               "funding_last", "premium_bps", "btc_ret_1", "btc_ret_16", "eth_ret_1", "sol_vs_btc_16"]


def build_features(raw: pd.DataFrame, groups: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """(features, group -> columns). Infinite values become NaN."""
    groups = groups or list(GROUPS)
    frames, cols = [], {}
    for g in groups:
        f = GROUPS[g](raw).replace([np.inf, -np.inf], np.nan).astype("float32")
        frames.append(f)
        cols[g] = list(f.columns)
    return pd.concat(frames, axis=1), cols


def columns_for(set_name_or_groups, group_cols: dict[str, list[str]]) -> list[str]:
    groups = FEATURE_SETS[set_name_or_groups] if isinstance(set_name_or_groups, str) else set_name_or_groups
    return [c for g in groups for c in group_cols[g]]


def group_coverage_start(feats: pd.DataFrame, group_cols: dict[str, list[str]], groups: list[str],
                         min_share: float = 0.9) -> pd.Timestamp:
    """First month from which every group in the set is >= min_share non-null (rolling 30 days)."""
    start = feats.index.min()
    for g in groups:
        ok = feats[group_cols[g]].notna().all(axis=1).astype(float).rolling(30 * DAY, min_periods=DAY).mean()
        good = ok[ok >= min_share]
        if good.empty:
            return pd.Timestamp.max.tz_localize("UTC")
        start = max(start, good.index.min() - pd.Timedelta(days=30))
    return start


def feature_set_id(name: str, columns: list[str]) -> str:
    return f"{name}_{hashlib.sha256(','.join(columns).encode()).hexdigest()[:8]}"
