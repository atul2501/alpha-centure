"""Naive baselines and fixed rules (nothing fitted; the shared policy still picks threshold / hold on validation)."""

import numpy as np
import pandas as pd

from alpha.tournament.models.base import BaseTradingModel


class _Rule(BaseTradingModel):
    family = "rules"
    complexity = 0
    target_kinds = ("reg", "cls")

    def fit(self, X, y, train, val):
        return self


class Flat(_Rule):
    name = "flat"
    description = "never trades (the zero benchmark)"

    def _raw(self, X, rows):
        return np.zeros(rows.sum())


class BuyHold(_Rule):
    name = "buy_hold"
    description = "always long"
    min_val_trades = 1

    def _raw(self, X, rows):
        return np.ones(rows.sum())


class RandomEntries(_Rule):
    name = "random"
    description = "random scores; with quantile thresholds it trades at the same rate as other entries"

    def _raw(self, X, rows):
        rng = np.random.default_rng(self.seed + int(X.index[rows][0].value // 10**9) % 10_000)
        return rng.standard_normal(rows.sum())


class Momentum(_Rule):
    name = "momentum"
    description = "sign and size of the vol-scaled return over `lookback` (mom_z_16 / 96 / 384)"

    def _raw(self, X, rows):
        return X[f"mom_z_{self.params.get('lookback', 96)}"].to_numpy()[rows]


class Reversal(_Rule):
    name = "reversal"
    description = "fade the vol-scaled return of the last `lookback` bars"

    def _raw(self, X, rows):
        lb = self.params.get("lookback", 4)
        r = X[f"ret_{lb}"].to_numpy() / (X["rvol_96"].to_numpy() * np.sqrt(lb) + 1e-12)
        return -r[rows]


class Breakout(_Rule):
    name = "breakout"
    description = "position inside the `lookback`-bar Donchian channel (near the high -> long)"

    def _raw(self, X, rows):
        return X[f"donchian_pos_{self.params.get('lookback', 96)}"].to_numpy()[rows]


class FlowFollow(_Rule):
    name = "flow_follow"
    description = "follow taker order-flow imbalance over `window` minutes"

    def _raw(self, X, rows):
        return X[f"flow_{self.params.get('window', 60)}"].to_numpy()[rows]
