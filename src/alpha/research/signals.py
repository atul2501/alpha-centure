"""Signal families (Phase 2 hypotheses). Each returns a time x symbol score known at the bar close.

Direction is fixed BEFORE testing (score > 0 = want long):
    tsmom_L      time-series momentum: L-hour return / L-hour vol
    xsmom_L      cross-sectional momentum: rank of L-hour vol-scaled return across eligible coins
    xsrev_L      cross-sectional short-term reversal: minus rank of L-hour return
    carry        short coins with high funding: minus last settled funding (TS)
    xscarry      cross-sectional: minus rank of 24h-average funding
    basis        minus rank of 24h-average premium (rich perps underperform)
    oi_trend     OI growth confirms the move: sign(24h return) * z(24h OI change)
    ls_contra    fade the crowd: minus z(7d) of all-account long/short ratio
    toptrader    follow top-trader positioning: z(7d) of top-trader long/short ratio
    flow_L       taker buy pressure: z of L-hour taker-buy ratio (TS)
    donchian_N   breakout: +1 above the prior N-hour high, -1 below the prior low, held until the opposite break
    meanrev      minus Bollinger z (24h) (TS)
"""

import numpy as np
import pandas as pd

from alpha.research.panel import wide

VOL_WINDOW = 168


def _z(x: pd.DataFrame, n: int) -> pd.DataFrame:
    m, s = x.rolling(n, min_periods=n // 2).mean(), x.rolling(n, min_periods=n // 2).std()
    return (x - m) / s.replace(0, np.nan)


def _xs_rank(x: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Demeaned cross-sectional rank in [-1, 1] among eligible coins."""
    x = x.where(eligible)
    r = x.rank(axis=1, pct=True)
    return r.sub(r.mean(axis=1), axis=0) * 2


def _donchian(close: pd.DataFrame, n: int) -> pd.DataFrame:
    hi = close.rolling(n, min_periods=n).max().shift(1)
    lo = close.rolling(n, min_periods=n).min().shift(1)
    s = pd.DataFrame(np.where(close > hi, 1.0, np.where(close < lo, -1.0, np.nan)), index=close.index,
                     columns=close.columns)
    return s.ffill().fillna(0.0)


def compute(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All family scores, time x symbol."""
    ret = wide(panel, "ret")
    close = wide(panel, "close")
    el = wide(panel, "eligible").fillna(False).astype(bool)
    vol = ret.rolling(VOL_WINDOW, min_periods=48).std()
    cum = ret.fillna(0).cumsum()
    out: dict[str, pd.DataFrame] = {}
    for L in (24, 72, 168, 336, 720):
        r = cum - cum.shift(L)
        scaled = r / (vol * np.sqrt(L))
        out[f"tsmom_{L}"] = scaled.clip(-3, 3)
        out[f"xsmom_{L}"] = _xs_rank(scaled, el)
    for L in (1, 4, 24):
        out[f"xsrev_{L}"] = -_xs_rank(cum - cum.shift(L), el)
    fl = wide(panel, "funding_last")
    out["carry"] = -_z(fl, 24 * 30).clip(-3, 3)
    fr = wide(panel, "funding_rate").replace(0, np.nan).ffill()
    out["xscarry"] = -_xs_rank(fr.rolling(24, min_periods=8).mean(), el)
    prem = wide(panel, "premium")
    out["basis"] = -_xs_rank(prem.rolling(24, min_periods=12).mean(), el)
    oi = wide(panel, "oi_value").where(lambda x: x > 0)
    oi_chg = np.log(oi / oi.shift(24))
    out["oi_trend"] = (np.sign(cum - cum.shift(24)) * _z(oi_chg, 24 * 30)).clip(-3, 3)
    out["ls_contra"] = -_z(wide(panel, "ls_ratio"), 168).clip(-3, 3)
    out["toptrader"] = _z(wide(panel, "toptrader_ls"), 168).clip(-3, 3)
    tbr = wide(panel, "taker_buy_ratio")
    for L in (4, 24):
        out[f"flow_{L}"] = _z(tbr.rolling(L, min_periods=L).mean(), 24 * 30).clip(-3, 3)
    for n in (480, 1320):
        out[f"donchian_{n}"] = _donchian(close, n)
    mid = close.rolling(24, min_periods=24).mean()
    sd = close.rolling(24, min_periods=24).std()
    out["meanrev"] = (-(close - mid) / sd).clip(-3, 3)
    return out


KIND = {}  # 'xs' = cross-sectional (dollar-neutral), 'ts' = time-series (directional)
for _n in ["xsmom_24", "xsmom_72", "xsmom_168", "xsmom_336", "xsmom_720", "xsrev_1", "xsrev_4", "xsrev_24",
           "xscarry", "basis"]:
    KIND[_n] = "xs"


def kind(name: str) -> str:
    return KIND.get(name, "ts")
