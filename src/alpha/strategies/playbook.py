"""The strategy playbook. Thresholds here are sensible starting points; the backtest decides what survives."""

import numpy as np
import pandas as pd

from alpha.strategies.base import Strategy, build_setups, clamp_stop, r_target


def _side(long_cond: pd.Series, short_cond: pd.Series) -> pd.Series:
    return pd.Series(np.where(long_cond.fillna(False), 1, np.where(short_cond.fillna(False), -1, 0)),
                     index=long_cond.index)


class Breakout(Strategy):
    """Volatility squeeze, then a close outside the prior 20-bar range on above-average volume."""

    def signals(self, f):
        was_squeezed = f["squeeze_len"].shift(1) >= self.params.get("min_squeeze", 6)
        vol_ok = f["vol_z"] > self.params.get("min_vol_z", 1.0)
        side = _side(was_squeezed & vol_ok & (f["close"] > f["dc_hi"]),
                     was_squeezed & vol_ok & (f["close"] < f["dc_lo"]))
        mid = (f["dc_hi"] + f["dc_lo"]) / 2
        stop = clamp_stop(f["close"], mid, f["atr"], side)
        return build_setups(side, stop, r_target(f["close"], stop, side, 2.0))


class FailedBreakout(Strategy):
    """Previous bar closed outside the range, this bar closes back inside: fade the trapped side."""

    def signals(self, f):
        prev_hi, prev_lo = f["dc_hi"].shift(1), f["dc_lo"].shift(1)
        broke_up = f["close"].shift(1) > prev_hi
        broke_dn = f["close"].shift(1) < prev_lo
        side = _side(broke_dn & (f["close"] > prev_lo), broke_up & (f["close"] < prev_hi))
        wick_hi = f["high"].rolling(2).max()
        wick_lo = f["low"].rolling(2).min()
        raw_stop = np.where(side > 0, wick_lo - 0.1 * f["atr"], wick_hi + 0.1 * f["atr"])
        stop = clamp_stop(f["close"], pd.Series(raw_stop, index=f.index), f["atr"], side)
        target = (prev_hi + prev_lo) / 2  # back to the middle of the range
        ok = (target - f["close"]) * side > 0.5 * (f["close"] - stop).abs()
        return build_setups(side.where(ok, 0), stop, target)


class TrendPullback(Strategy):
    """Higher-timeframe trend, price dips to EMA20 / swing VWAP, then closes back in trend direction."""

    def signals(self, f):
        up = (f["h4_trend"] > 0) & (f["ema20"] > f["ema50"])
        dn = (f["h4_trend"] < 0) & (f["ema20"] < f["ema50"])
        touch_lo = (f["low"] <= f["ema20"]) | (f["low"] <= f["avwap_lo"])
        touch_hi = (f["high"] >= f["ema20"]) | (f["high"] >= f["avwap_hi"])
        bull = (f["close"] > f["open"]) & (f["close"] > f["ema20"])
        bear = (f["close"] < f["open"]) & (f["close"] < f["ema20"])
        side = _side(up & touch_lo & bull & (f["rsi"] < 65), dn & touch_hi & bear & (f["rsi"] > 35))
        swing_lo = f["low"].rolling(5).min() - 0.2 * f["atr"]
        swing_hi = f["high"].rolling(5).max() + 0.2 * f["atr"]
        stop = clamp_stop(f["close"], pd.Series(np.where(side > 0, swing_lo, swing_hi), index=f.index), f["atr"], side)
        return build_setups(side, stop, r_target(f["close"], stop, side, 2.0))


class MeanReversion(Strategy):
    """Stretched far outside the Bollinger band with an RSI extreme: fade back to the mean."""

    def signals(self, f):
        z = self.params.get("z", 2.2)
        side = _side((f["bb_z"] < -z) & (f["rsi"] < 30), (f["bb_z"] > z) & (f["rsi"] > 70))
        raw_stop = np.where(side > 0, f["low"] - f["atr"], f["high"] + f["atr"])
        stop = clamp_stop(f["close"], pd.Series(raw_stop, index=f.index), f["atr"], side)
        return build_setups(side, stop, f["bb_mid"])


class VwapReclaim(Strategy):
    """Session VWAP lost then reclaimed with buyers in control (or rejected from below with sellers)."""

    def signals(self, f):
        crossed_up = (f["close"].shift(1) < f["vwap"].shift(1)) & (f["close"] > f["vwap"])
        crossed_dn = (f["close"].shift(1) > f["vwap"].shift(1)) & (f["close"] < f["vwap"])
        was_below = (f["close"] < f["vwap"]).rolling(6).sum().shift(1) >= 4
        was_above = (f["close"] > f["vwap"]).rolling(6).sum().shift(1) >= 4
        settled = f["hour"] >= 2  # VWAP is noisy in the first hours of the UTC session
        side = _side(crossed_up & was_below & (f["taker_buy_ratio"] > 0.52) & settled,
                     crossed_dn & was_above & (f["taker_buy_ratio"] < 0.48) & settled)
        raw_stop = np.where(side > 0, np.minimum(f["low"], f["vwap"] - 0.5 * f["atr"]),
                            np.maximum(f["high"], f["vwap"] + 0.5 * f["atr"]))
        stop = clamp_stop(f["close"], pd.Series(raw_stop, index=f.index), f["atr"], side)
        band = f["vwap"] + np.sign(side) * f["vwap_sd"]
        target = pd.Series(np.where((band - f["close"]) * side > 1.5 * (f["close"] - stop).abs(), band,
                                    r_target(f["close"], stop, side, 1.5)), index=f.index)
        return build_setups(side, stop, target)


class PositioningSqueeze(Strategy):
    """Crowded positioning (extreme funding) plus a reversal candle: fade the crowd."""

    def signals(self, f):
        rng = (f["high"] - f["low"]).replace(0, np.nan)
        upper_wick = (f["high"] - f[["open", "close"]].max(axis=1)) / rng
        lower_wick = (f[["open", "close"]].min(axis=1) - f["low"]) / rng
        z = self.params.get("funding_z", 2.0)
        side = _side((f["funding_z"] < -z) & (lower_wick > 0.5) & (f["close"] > f["open"]),
                     (f["funding_z"] > z) & (upper_wick > 0.5) & (f["close"] < f["open"]))
        raw_stop = np.where(side > 0, f["low"] - 0.2 * f["atr"], f["high"] + 0.2 * f["atr"])
        stop = clamp_stop(f["close"], pd.Series(raw_stop, index=f.index), f["atr"], side)
        return build_setups(side, stop, r_target(f["close"], stop, side, 2.0))


PLAYBOOK: list[Strategy] = [
    Breakout("breakout", ("15m", "1h"), max_bars=48),
    FailedBreakout("failed_breakout", ("15m", "1h"), max_bars=24),
    TrendPullback("trend_pullback", ("1h", "15m"), max_bars=48),
    MeanReversion("mean_reversion", ("15m", "1h"), max_bars=24),
    VwapReclaim("vwap_reclaim", ("5m", "15m"), max_bars=36),
    PositioningSqueeze("positioning_squeeze", ("15m", "1h"), max_bars=48),
]


def all_setups(f: pd.DataFrame, tf: str, strategies: list[Strategy] | None = None) -> pd.DataFrame:
    parts = [s.setups(f) for s in (strategies or PLAYBOOK) if tf in s.timeframes]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()
