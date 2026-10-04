"""Simulate each setup forward on the real high/low path and label the outcome after costs.

Shared by research (dataset/backtest) and the live scorer, so a trade is always judged the same way.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    fee: float = 0.0005          # per side, Binance USD-M taker
    slippage: float = 0.0001     # per side on market fills, fraction of price
    funding_hours: tuple[int, ...] = (0, 8, 16)  # UTC settlement hours
    fee_entry: float | None = None   # set (e.g. 0.0002) when entries are maker limit orders
    fee_target: float | None = None  # set when target/partial exits rest as maker limit orders

    @property
    def round_trip(self) -> float:
        """Conservative all-taker round trip, used to express cost in R."""
        return 2 * (self.fee + self.slippage)

    def entry_cost(self, limit: bool) -> float:
        return self.fee_entry if (limit and self.fee_entry is not None) else self.fee + self.slippage

    def exit_cost(self, kind: str) -> float:
        if kind == "target" and self.fee_target is not None:
            return self.fee_target
        return self.fee + self.slippage


@dataclass(frozen=True)
class Entry:
    """market: next bar's open. limit: resting order at signal close -/+ offset_atr*ATR (better price), valid for
    fill_bars bars; it fills only if price trades through it, otherwise the setup is 'unfilled' (no trade, no cost)."""

    mode: str = "market"
    offset_atr: float = 0.0
    fill_bars: int = 2


@dataclass(frozen=True)
class ExitPolicy:
    """How an open trade is managed. Stops only move using *closed* bars and apply from the next bar."""

    name: str = "fixed"
    breakeven_at_r: float | None = None  # close >= +X R -> stop to entry (+ fees)
    trail_atr: float | None = None       # chandelier: best high/low since entry -/+ trail_atr * ATR
    trail_start_r: float = 1.0           # trailing starts once a close reached +trail_start_r R
    partial_at_r: float | None = None    # take partial_frac off at +X R
    partial_frac: float = 0.5
    keep_target: bool = True             # False: no fixed target, let the trail/time limit exit
    time_mult: float = 1.0               # time limit = max_bars * time_mult


EXIT_MENU: dict[str, ExitPolicy] = {
    "fixed": ExitPolicy("fixed"),
    "breakeven": ExitPolicy("breakeven", breakeven_at_r=1.0),
    "trail": ExitPolicy("trail", trail_atr=2.5, trail_start_r=1.0, keep_target=False, time_mult=2.0),
    "partial_trail": ExitPolicy("partial_trail", partial_at_r=1.0, partial_frac=0.5, breakeven_at_r=1.0,
                                trail_atr=2.5, trail_start_r=1.0, keep_target=False, time_mult=2.0),
}

LABEL_COLS = ["entry_time", "entry", "exit_time", "exit", "outcome", "bars_held", "risk", "gross", "funding_cost",
              "pnl", "r", "win"]
EXIT_LABEL_COLS = ["exit_time", "outcome", "bars_held", "gross", "pnl", "r", "win"]


def label_setups(setups: pd.DataFrame, f: pd.DataFrame, costs: Costs = Costs(),
                 exit: ExitPolicy = EXIT_MENU["fixed"], entry: Entry = Entry()) -> pd.DataFrame:
    """Label each setup (indexed by signal bar) with one exit policy.

    - market entry at the next bar's open; limit entry as described in `Entry`.
    - Stop and target are checked on each bar's low/high. If both are touched in the same bar, the stop wins
      (the order inside a bar is unknown, so assume the worse case). Partial take-profit follows the same rule.
    - An entry at or through the stop is invalid (dropped). A later gap through the stop fills at that bar's open.
    - Timeout exits at the close of the last bar. pnl and r are after fees, slippage and funding.
    - Setups still open when the data ends are dropped (unknown). Unfilled limit entries are dropped.
    """
    return label_multi(setups, f, costs, {exit.name: exit}, entry, primary=exit.name)


def label_multi(setups: pd.DataFrame, f: pd.DataFrame, costs: Costs, exits: dict[str, ExitPolicy],
                entry: Entry = Entry(), primary: str = "fixed") -> pd.DataFrame:
    """Label every setup under several exit policies in one pass. The primary policy fills LABEL_COLS; every
    policy also gets suffixed columns `<col>__<name>`. A setup is kept only if all policies resolved."""
    if setups.empty:
        return setups.assign(**{c: [] for c in LABEL_COLS})
    arr = {k: f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    arr["atr"] = f["atr"].to_numpy(dtype=float) if "atr" in f else _atr(f)
    funding = f["funding"].to_numpy(dtype=float) if "funding" in f else np.zeros(len(f))
    idx = f.index
    pos = idx.get_indexer(setups.index)
    keep, results = [], []
    for p, side, stop, target, max_bars in zip(pos, setups["side"].to_numpy(), setups["stop"].to_numpy(dtype=float),
                                               setups["target"].to_numpy(dtype=float),
                                               setups["max_bars"].to_numpy()):
        if p < 0 or p + 1 >= len(f):
            keep.append(False)
            continue
        fill = _fill(arr, p, int(side), stop, entry)
        if fill is None:
            keep.append(False)
            continue
        e, entry_px, limit_fill = fill
        res = {}
        for name, ex in exits.items():
            r = _simulate(arr, funding, idx, p, e, entry_px, limit_fill, int(side), stop, target, int(max_bars), ex,
                          costs)
            if r is None:
                break
            res[name] = r
        else:
            keep.append(True)
            results.append(res)
            continue
        keep.append(False)
    out = setups[np.array(keep, dtype=bool)].copy()
    if not results:
        return out.assign(**{c: [] for c in LABEL_COLS})
    prim = pd.DataFrame([r[primary] for r in results])
    for col in LABEL_COLS:  # positional: signal times repeat across setups, so never join on the index
        out[col] = prim[col].to_numpy()
    if len(exits) > 1:
        for name in exits:
            d = pd.DataFrame([r[name] for r in results])
            for col in EXIT_LABEL_COLS:
                out[f"{col}__{name}"] = d[col].to_numpy()
    return out


def apply_exit(ds: pd.DataFrame, choice: str | dict[str, str]) -> pd.DataFrame:
    """Make the chosen exit policy (one name, or per-strategy dict) the active label columns."""
    out = ds.copy()
    names = (pd.Series(choice, index=ds.index) if isinstance(choice, str)
             else ds["strategy"].map(choice).fillna("fixed"))
    out["exit_policy"] = names.to_numpy()
    for name in names.unique():
        m = (names == name).to_numpy()
        if f"r__{name}" not in ds:
            if name == "fixed":
                continue
            raise KeyError(f"dataset has no labels for exit policy {name!r}")
        for col in EXIT_LABEL_COLS:
            out.loc[m, col] = ds.loc[m, f"{col}__{name}"].to_numpy()
    out["win"] = out["win"].astype(bool)
    return out


# ---------------------------------------------------------------------------------------------------------------

def _fill(arr, p: int, side: int, stop: float, entry: Entry):
    """Returns (entry bar, entry price, is_limit_fill) or None if not filled / invalid."""
    o, h, l = arr["open"], arr["high"], arr["low"]
    if entry.mode == "market":
        e = p + 1
        px = o[e]
        return (e, px, False) if (px - stop) * side > 0 else None
    limit = arr["close"][p] - side * entry.offset_atr * arr["atr"][p]
    for e in range(p + 1, min(p + 1 + entry.fill_bars, len(o))):
        touched = (l[e] <= limit) if side > 0 else (h[e] >= limit)
        if touched:
            # a gap through the limit fills at the (better) open
            px = min(limit, o[e]) if side > 0 else max(limit, o[e])
            return (e, px, True) if (px - stop) * side > 0 else None
    return None


def _simulate(arr, funding, idx, p, e, entry_px, limit_fill, side, stop0, target, max_bars, ex: ExitPolicy,
              costs: Costs):
    o, h, l, c, atr = arr["open"], arr["high"], arr["low"], arr["close"], arr["atr"]
    n = len(o)
    risk = (entry_px - stop0) * side
    if risk <= 0:
        return None
    tgt = target if (ex.keep_target and not np.isnan(target)) else np.nan
    if not np.isnan(tgt) and (tgt - entry_px) * side <= 0:
        return None
    planned_last = e + int(round(max_bars * ex.time_mult)) - 1
    last = min(planned_last, n - 1)
    stop = stop0
    partial_px = entry_px + side * ex.partial_at_r * risk if ex.partial_at_r else np.nan
    partial_done = False
    best = entry_px  # best high (long) / low (short) since entry, from closed bars
    trailing = False
    legs = []  # (fraction, price, kind)
    remaining = 1.0
    exit_px, outcome, j = c[last], "timeout", last
    for j in range(e, last + 1):
        hit_stop = (l[j] <= stop) if side > 0 else (h[j] >= stop)
        if hit_stop:
            exit_px = min(stop, o[j]) if side > 0 else max(stop, o[j])
            outcome = "stop" if stop == stop0 else ("breakeven" if not trailing else "trail")
            break
        if not partial_done and not np.isnan(partial_px):
            if (h[j] >= partial_px) if side > 0 else (l[j] <= partial_px):
                legs.append((ex.partial_frac, partial_px, "target"))
                remaining -= ex.partial_frac
                partial_done = True
        if not np.isnan(tgt) and ((h[j] >= tgt) if side > 0 else (l[j] <= tgt)):
            exit_px, outcome = tgt, "target"
            break
        # --- bar closed without exit: update the stop for the next bar ---
        best = max(best, h[j]) if side > 0 else min(best, l[j])
        close_r = (c[j] - entry_px) * side / risk
        if ex.breakeven_at_r is not None and close_r >= ex.breakeven_at_r:
            be = entry_px * (1 + side * costs.round_trip)
            stop = max(stop, be) if side > 0 else min(stop, be)
        if ex.trail_atr is not None and (trailing or close_r >= ex.trail_start_r) and not np.isnan(atr[j]):
            trailing = True
            ts = best - side * ex.trail_atr * atr[j]
            stop = max(stop, ts) if side > 0 else min(stop, ts)
    else:
        if planned_last > last:  # data ends before the trade resolved: outcome unknown yet
            return None
        j = last
    exit_kind = "target" if outcome == "target" else "stop"
    legs.append((remaining, exit_px, exit_kind))
    gross = sum(fr * side * (px - entry_px) / entry_px for fr, px, _ in legs)
    fees = costs.entry_cost(limit_fill) + sum(fr * costs.exit_cost(kind) for fr, _, kind in legs)
    settlements = _count_settlements(idx[e], idx[j], costs.funding_hours)
    fund_cost = side * (funding[p] if not np.isnan(funding[p]) else 0.0) * settlements
    pnl = gross - fees - fund_cost
    r_risk = risk / entry_px
    return {"entry_time": idx[e], "entry": entry_px, "exit_time": idx[j], "exit": exit_px, "outcome": outcome,
            "bars_held": j - e + 1, "risk": r_risk, "gross": gross, "funding_cost": fund_cost, "pnl": pnl,
            "r": pnl / r_risk, "win": pnl > 0}


def _atr(f: pd.DataFrame, n: int = 14) -> np.ndarray:
    from alpha.features.indicators import atr

    return atr(f, n).to_numpy(dtype=float)


def _count_settlements(start: pd.Timestamp, end: pd.Timestamp, hours: tuple[int, ...]) -> int:
    """Funding settlements strictly after entry bar open and at or before the exit bar open."""
    if end <= start:
        return 0
    days = pd.date_range(start.floor("1D"), end.floor("1D"), freq="1D")
    times = [d + pd.Timedelta(hours=hh) for d in days for hh in hours]
    return sum(start < t <= end for t in times)
