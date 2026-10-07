"""Event backtest on the 15m grid, identical for every model.

Input: target position pos[t] in {-1, 0, +1} decided at the close of bar t (time index = bar open time). The
position is entered / changed at open[t+1] (~ close[t]) and earns r[t] = log(open[t+2] / open[t+1]) over the next
bar. Every change of position pays the per-side cost at t once per side (a flip pays two sides). A position open
at a funding settlement pays side * rate (long pays a positive rate).

Sizing: one SOL position of fixed notional = 1 x equity (weight 1). P&L is in account fractions (bps / 1e4).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.tournament.execution.fills import SideCosts

BAR = pd.Timedelta(minutes=15)


@dataclass
class Market:
    """Everything the backtest needs, aligned to the decision grid."""
    index: pd.DatetimeIndex
    ret_next: np.ndarray       # bps earned by a position decided at t
    costs: SideCosts
    funding_bar: np.ndarray    # funding rate (bps) settled while holding the position decided at t

    def slice(self, mask: np.ndarray) -> "Market":
        return Market(self.index[mask], self.ret_next[mask], SideCosts(self.costs.fee[mask], self.costs.slip[mask]),
                      self.funding_bar[mask])


def make_market(open_: pd.Series, costs: SideCosts, funding: pd.Series | None) -> Market:
    o = open_.astype(float).to_numpy()
    ret = np.full(len(o), np.nan)
    ret[:-2] = 1e4 * np.log(o[2:] / o[1:-1])
    idx = pd.DatetimeIndex(open_.index)
    fb = np.zeros(len(o))
    if funding is not None and len(funding):
        f = funding[(funding.index > idx[0]) & (funding.index <= idx[-1] + 2 * BAR)]
        # settlement tau is crossed by the position decided at t when close[t] < tau <= close[t+1]
        k = idx.searchsorted(f.index - 2 * BAR, side="left")
        ok = (k < len(idx))
        k, rates = k[ok], f.to_numpy()[ok] * 1e4
        ok2 = idx[k] < f.index[ok] - BAR
        np.add.at(fb, k[ok2], rates[ok2])
    return Market(idx, ret, costs, fb)


@dataclass
class Result:
    trades: pd.DataFrame       # one row per closed trade, index = exit time
    bar_pnl: pd.Series         # account fraction per bar (net, mark-to-market)
    position: np.ndarray


def run(pos: np.ndarray, m: Market) -> Result:
    pos = np.nan_to_num(np.asarray(pos, float)).clip(-1, 1).round()
    ret = np.nan_to_num(m.ret_next)
    pos = np.where(np.isfinite(m.ret_next), pos, 0.0)  # no data for the next bar -> flat (cannot mark)
    prev = np.concatenate([[0.0], pos[:-1]])
    sides = np.abs(pos - prev)                          # 0, 1 or 2 sides traded at t
    fee_t, slip_t = sides * m.costs.fee, sides * m.costs.slip
    fund_t = pos * m.funding_bar
    gross_t = pos * ret
    bar_net = gross_t - fee_t - slip_t - fund_t
    # close any position at the end of the window
    last_close_fee = abs(pos[-1]) * m.costs.fee[-1] if len(pos) else 0.0
    last_close_slip = abs(pos[-1]) * m.costs.slip[-1] if len(pos) else 0.0
    if len(pos):
        bar_net[-1] -= last_close_fee + last_close_slip
    bar_pnl = pd.Series(bar_net / 1e4, index=m.index)
    trades = _trades(pos, gross_t, m, last_close_fee, last_close_slip)
    return Result(trades, bar_pnl, pos)


def _trades(pos, gross_t, m: Market, end_fee: float, end_slip: float) -> pd.DataFrame:
    n = len(pos)
    if n == 0 or not np.any(pos):
        return _empty()
    change = np.flatnonzero(np.diff(np.concatenate([[0.0], pos])) != 0)  # bars where the position changes
    starts = [i for i in change if pos[i] != 0]
    rows = []
    cs = np.concatenate([[0.0], np.cumsum(gross_t)])
    fs = np.concatenate([[0.0], np.cumsum(pos * m.funding_bar)])
    for s in starts:
        nxt = np.searchsorted(change, s, side="right")
        e = change[nxt] if nxt < len(change) else n  # first bar with a different position (exit decided there)
        side = pos[s]
        if e < n:
            exit_fee, exit_slip = m.costs.fee[e], m.costs.slip[e]
        else:
            exit_fee, exit_slip = end_fee, end_slip
        fee = m.costs.fee[s] + exit_fee
        slip = m.costs.slip[s] + exit_slip
        gross = cs[e] - cs[s]
        fund = fs[e] - fs[s]
        exit_ts = m.index[e] + BAR if e < n else m.index[-1] + 2 * BAR
        rows.append((m.index[s] + BAR, exit_ts, int(side), int(e - s), gross, fee, slip, fund))
    t = pd.DataFrame(rows, columns=["entry_ts", "exit_ts", "side", "bars", "gross_bps", "fee_bps", "slip_bps",
                                    "funding_bps"])
    t["cost_bps"] = t["fee_bps"] + t["slip_bps"]
    t["net_bps"] = t["gross_bps"] - t["cost_bps"] - t["funding_bps"]
    t["symbol"] = "SOLUSDT"
    t["weight"] = 1.0
    return t.set_index(pd.DatetimeIndex(t["exit_ts"]))


def _empty() -> pd.DataFrame:
    cols = ["entry_ts", "exit_ts", "side", "bars", "gross_bps", "fee_bps", "slip_bps", "funding_bps", "cost_bps",
            "net_bps", "symbol", "weight"]
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"))
