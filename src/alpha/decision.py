"""The single LONG / SHORT / PASS decision function, shared by the backtest and the live predictor.

Input: candidate setups for one moment (already scored by the meta model, with regime attached).
Output: the same rows with `action` (LONG / SHORT / PASS) and `reason`, at most one trade per symbol.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LONG, SHORT, PASS = "LONG", "SHORT", "PASS"


@dataclass
class Policy:
    enabled: set[tuple[str, str, str]] = field(default_factory=set)  # (strategy, tf, regime)
    min_ev_r: float = 0.1
    risk_per_trade: float = 0.01
    min_trades_per_cell: int = 30
    max_cost_r: float | None = None        # skip setups whose round-trip cost exceeds this many R
    round_trip: float = 0.0012             # taker fees + slippage both sides, to express cost in R
    exits: dict[str, str] = field(default_factory=dict)  # strategy -> exit policy name (default 'fixed')
    max_same_side: int | None = None       # cap on simultaneous LONGs (or SHORTs) across symbols
    sizing: str = "fixed"                  # fixed | ev (risk scaled by expected R, capped)

    @classmethod
    def learn(cls, train: pd.DataFrame, min_trades: int = 30, weights: np.ndarray | None = None, **kw) -> "Policy":
        """Enable a (strategy, tf, regime) cell only if it made money after costs in the training data,
        with enough trades and a positive lower bound (mean R minus half a standard error).
        With recency `weights`, means/standard errors are weighted and n is the effective sample size."""
        w = np.ones(len(train)) if weights is None else np.asarray(weights, dtype=float)
        t = train[["strategy", "tf", "regime", "r"]].assign(w=w, wr=w * train["r"].to_numpy())
        g = t.groupby(["strategy", "tf", "regime"])
        sw, swr = g["w"].sum(), g["wr"].sum()
        mean = swr / sw
        dev2 = (t["w"] * (t["r"] - t.set_index(["strategy", "tf", "regime"]).index.map(mean).to_numpy()) ** 2)
        var = dev2.groupby([t["strategy"], t["tf"], t["regime"]]).sum() / sw
        n_eff = sw**2 / g["w"].apply(lambda x: (x**2).sum())
        stats = pd.DataFrame({"n": g.size(), "n_eff": n_eff, "mean": mean, "se": np.sqrt(var / n_eff)})
        ok = stats[(stats["n"] >= min_trades) & (stats["n_eff"] >= min_trades / 2) & (stats["mean"] - 0.5 * stats["se"] > 0)]
        return cls(enabled=set(ok.index), min_trades_per_cell=min_trades, **kw)

    def risk_for(self, ev_r: float) -> float:
        if self.sizing == "ev" and np.isfinite(ev_r):
            return float(np.clip(self.risk_per_trade * ev_r / 0.25, 0.25 * self.risk_per_trade, self.risk_per_trade))
        return self.risk_per_trade

    def to_dict(self) -> dict:
        return {"enabled": sorted(map(list, self.enabled)), "min_ev_r": self.min_ev_r,
                "risk_per_trade": self.risk_per_trade, "min_trades_per_cell": self.min_trades_per_cell,
                "max_cost_r": self.max_cost_r, "round_trip": self.round_trip, "exits": dict(self.exits),
                "max_same_side": self.max_same_side, "sizing": self.sizing}

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        return cls(enabled={tuple(x) for x in d["enabled"]}, min_ev_r=d["min_ev_r"],
                   risk_per_trade=d["risk_per_trade"], min_trades_per_cell=d["min_trades_per_cell"],
                   max_cost_r=d.get("max_cost_r"), round_trip=d.get("round_trip", 0.0012),
                   exits=d.get("exits", {}), max_same_side=d.get("max_same_side"), sizing=d.get("sizing", "fixed"))


def static_reasons(cands: pd.DataFrame, policy: Policy, use_regime: bool = True, use_meta: bool = True) -> np.ndarray:
    """Gates that depend only on each setup itself (not on time or other setups). Vectorized, so a backtest
    can pre-filter a whole dataset with exactly the rules decide() applies. '' = passes."""
    reasons = np.full(len(cands), "", dtype=object)
    if policy.max_cost_r is not None and {"close_px", "stop"} <= set(cands.columns):
        close = cands["close_px"].to_numpy(float)
        risk = np.abs(close - cands["stop"].to_numpy(float)) / close
        with np.errstate(divide="ignore", invalid="ignore"):
            reasons[~(policy.round_trip / risk <= policy.max_cost_r)] = "cost_too_high"
    if use_meta:
        reasons[(cands["ev_r"].fillna(-np.inf) < policy.min_ev_r).to_numpy()] = "edge_below_min"
    if use_regime:
        keys = zip(cands["strategy"], cands["tf"], cands["regime"])
        reasons[np.array([k not in policy.enabled for k in keys], dtype=bool)] = "regime_off"
    return reasons


def decide(cands: pd.DataFrame, policy: Policy, busy_symbols: set[str] = frozenset(),
           use_regime: bool = True, use_meta: bool = True, open_sides: dict[int, int] | None = None) -> pd.DataFrame:
    """cands columns: symbol, strategy, tf, side, regime, ev_r (+ anything else, passed through).

    Gates in order (first failing gate gives the PASS reason):
      position_open  -> symbol already has an open trade
      regime_off     -> (strategy, tf, regime) not enabled by the policy
      edge_below_min -> meta model expected R below the threshold
      cost_too_high  -> round-trip cost is more than policy.max_cost_r R (stop too tight for the fees)
      conflict       -> long and short both survive for the same symbol
      not_best       -> another setup for the same symbol has a higher expected R
      exposure_cap   -> too many positions in the same direction (alts move together)
    """
    idx_name = cands.index.name or "index"
    out = cands.reset_index()  # positional RangeIndex: signal times repeat across setups
    out["action"], out["reason"] = PASS, ""
    alive = np.ones(len(out), dtype=bool)

    def gate(mask: np.ndarray, reason: str) -> None:
        nonlocal alive
        hit = alive & mask
        out.loc[hit, "reason"] = reason
        alive &= ~mask

    gate(out["symbol"].isin(busy_symbols).to_numpy(), "position_open")
    static = static_reasons(out, policy, use_regime, use_meta)
    for reason in ("regime_off", "cost_too_high", "edge_below_min"):
        gate(static == reason, reason)

    if alive.any():
        live = out[alive]
        sides = live.groupby("symbol")["side"].nunique()
        gate(out["symbol"].isin(sides[sides > 1].index).to_numpy(), "conflict")
    if alive.any():
        live = out[alive]
        score = live["ev_r"].fillna(0.0) if use_meta else pd.Series(0.0, index=live.index)
        best = live.assign(_s=score).sort_values("_s", ascending=False, kind="stable").groupby("symbol").head(1).index
        not_best = alive.copy()
        not_best[best] = False
        gate(not_best, "not_best")
    if policy.max_same_side is not None and alive.any():
        # keep the highest-EV new trades per direction until the cap (counting positions already open)
        open_sides = open_sides or {}
        over = np.zeros(len(out), dtype=bool)
        for side in (1, -1):
            idx = np.flatnonzero(alive & (out["side"].to_numpy() == side))
            room = max(0, policy.max_same_side - open_sides.get(side, 0))
            ranked = idx[np.argsort(-out["ev_r"].fillna(0.0).to_numpy()[idx], kind="stable")]
            over[ranked[room:]] = True
        gate(over, "exposure_cap")
    out.loc[alive, "action"] = np.where(out.loc[alive, "side"] > 0, LONG, SHORT)
    return out.set_index(idx_name)
