"""Hybrids: a regime model (mode 2) fit on the fold's training rows produces forward-filtered regime features,
which are appended to the tabular features of a predictor (tree / linear / deep) fit on the same training rows.

    params: {regime: hsmm, regime_params: {...}, predictor: lightgbm, predictor_params: {...}}

The REGIME feature group therefore never exists outside a fold: it cannot carry information from later bars.
"""

import numpy as np
import pandas as pd

from alpha.tournament.models import registry
from alpha.tournament.models.base import BaseTradingModel


class RegimeHybrid(BaseTradingModel):
    name = "hybrid"
    family = "hybrid"
    complexity = 3
    input_kind = "series"  # needs the full feature frame (regime inputs) next to the predictor's columns
    target_kinds = ("reg", "cls")

    def fit(self, X, y, train, val):
        rcls = registry.get(self.params["regime"])
        pcls = registry.get(self.params["predictor"])
        self.base_cols = self.params.get("columns") or [c for c in getattr(self, "context").group_cols_for(
            self.params.get("feature_set", "core"))]
        from alpha.tournament.targets.targets import Target

        rt = Target("reg", self.target.h)  # the regime model's own state means use the regression target
        self.regime = rcls(rt, self.params.get("regime_params", {}), seed=self.seed)
        yr = self.context.target(rt).iloc[: len(X)]
        self.regime.fit(X, yr, train, val)
        R = self.regime.transform(X)
        self.regime_cols = list(R.columns)
        self.pred = pcls(self.target, self.params.get("predictor_params", {}), seed=self.seed,
                         columns=self.base_cols + self.regime_cols)
        self.pred.context = self.context
        self.pred.fit(pd.concat([X[self.base_cols], R], axis=1), y, train, val)
        self._R = (len(X), R)
        self.complexity = max(3, getattr(pcls, "complexity", 2) + 1)
        return self

    def _frame(self, X):
        if self._R[0] == len(X):
            R = self._R[1]
        else:
            R = self.regime.transform(X)
            self._R = (len(X), R)
        return pd.concat([X[self.base_cols], R], axis=1)

    def _raw(self, X, rows):
        return self.pred._raw(self._frame(X), rows)

    def feature_importance(self):
        return self.pred.feature_importance()

    def __getstate__(self):
        d = super().__getstate__()
        d.pop("_R", None)
        d["_R"] = (None, None)
        return d
