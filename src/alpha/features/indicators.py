"""Technical indicators on an OHLCV frame (columns: open, high, low, close, volume; UTC DatetimeIndex
of bar open times). Every value at row t uses only rows <= t, so nothing here looks ahead."""

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
                     axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ATR."""
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI, 0..100."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(up.notna())


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_up": mid + k * sd, "bb_lo": mid - k * sd,
                         "bb_width": 2 * k * sd / mid, "bb_z": (close - mid) / sd})


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5) -> pd.DataFrame:
    mid = ema(df["close"], n)
    a = atr(df, n)
    return pd.DataFrame({"kc_mid": mid, "kc_up": mid + mult * a, "kc_lo": mid - mult * a})


def squeeze(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """True while Bollinger bands sit inside Keltner channels (volatility compression)."""
    bb, kc = bollinger(df["close"], n), keltner(df, n)
    return (bb["bb_up"] < kc["kc_up"]) & (bb["bb_lo"] > kc["kc_lo"])


def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Range of the *previous* n bars (excludes the current bar, so a close above dc_hi is a breakout)."""
    return pd.DataFrame({"dc_hi": df["high"].rolling(n).max().shift(1),
                         "dc_lo": df["low"].rolling(n).min().shift(1)})


def session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP reset every UTC day, with volume-weighted standard deviation bands."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    v = df["volume"]
    day = df.index.floor("1D")
    cv = v.groupby(day).cumsum()
    cpv = (tp * v).groupby(day).cumsum()
    cp2v = (tp * tp * v).groupby(day).cumsum()
    vwap = cpv / cv.replace(0, np.nan)
    sd = np.sqrt((cp2v / cv.replace(0, np.nan) - vwap**2).clip(lower=0))
    return pd.DataFrame({"vwap": vwap, "vwap_sd": sd, "vwap_z": (df["close"] - vwap) / sd.replace(0, np.nan)})


def swing_points(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Pivot highs/lows: bar i is a swing high if its high is the max of bars i-k..i+k.
    It is only *known* at bar i+k, so the values are reported on the confirmation bar (no look-ahead).
    Columns: swing_hi / swing_lo = pivot price on its confirmation bar (NaN elsewhere),
             swing_hi_pos / swing_lo_pos = integer position of the pivot bar."""
    w = 2 * k + 1
    is_hi = df["high"] == df["high"].rolling(w, center=True).max()
    is_lo = df["low"] == df["low"].rolling(w, center=True).min()
    pos = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    return pd.DataFrame({
        "swing_hi": df["high"].where(is_hi).shift(k),
        "swing_lo": df["low"].where(is_lo).shift(k),
        "swing_hi_pos": pos.where(is_hi).shift(k),
        "swing_lo_pos": pos.where(is_lo).shift(k),
    })


def anchored_vwap(df: pd.DataFrame, anchor_pos: pd.Series) -> pd.Series:
    """VWAP from the bar at `anchor_pos` (integer position, forward-filled per row) up to each row."""
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    v = df["volume"].to_numpy()
    cpv = np.concatenate([[0.0], np.cumsum(tp * v)])
    cv = np.concatenate([[0.0], np.cumsum(v)])
    a = anchor_pos.to_numpy()
    out = np.full(len(df), np.nan)
    ok = ~np.isnan(a)
    ai = a[ok].astype(int)
    t = np.arange(len(df))[ok]
    denom = cv[t + 1] - cv[ai]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = np.where(denom > 0, (cpv[t + 1] - cpv[ai]) / denom, np.nan)
    return pd.Series(out, index=df.index)


def volume_z(volume: pd.Series, n: int = 50) -> pd.Series:
    lv = np.log1p(volume)
    return (lv - lv.rolling(n).mean()) / lv.rolling(n).std(ddof=0)


def realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(n).std(ddof=0)


def parkinson_vol(df: pd.DataFrame, n: int = 20) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(n).mean() / (4 * np.log(2)))


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Standard indicator set used by the strategies and the meta model."""
    out = df.copy()
    c = out["close"]
    out["ret_1"] = np.log(c).diff()
    for n in (3, 12, 48):
        out[f"ret_{n}"] = np.log(c / c.shift(n))
    out["atr"] = atr(out, 14)
    out["atr_pct"] = out["atr"] / c
    out["rsi"] = rsi(c, 14)
    out["ema20"], out["ema50"], out["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    out["ema20_dist"] = (c - out["ema20"]) / out["atr"]
    out["ema50_slope"] = out["ema50"].pct_change(5)
    out = out.join(bollinger(c)).join(keltner(out)).join(donchian(out)).join(session_vwap(out))
    out["squeeze"] = (out["bb_up"] < out["kc_up"]) & (out["bb_lo"] > out["kc_lo"])
    out["squeeze_len"] = out["squeeze"].groupby((~out["squeeze"]).cumsum()).cumsum()
    out["vol_z"] = volume_z(out["volume"])
    out["rvol"] = realized_vol(c)
    out["pvol"] = parkinson_vol(out)
    out["taker_buy_ratio"] = out["taker_buy_base"] / out["volume"].replace(0, np.nan)
    sw = swing_points(out)
    out = out.join(sw)
    out["last_swing_lo"] = out["swing_lo"].ffill()
    out["last_swing_hi"] = out["swing_hi"].ffill()
    out["avwap_lo"] = anchored_vwap(out, out["swing_lo_pos"].ffill())
    out["avwap_hi"] = anchored_vwap(out, out["swing_hi_pos"].ffill())
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    return out
