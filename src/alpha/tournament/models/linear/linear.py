"""Linear baselines (sklearn) on standardized features (preprocessing fit on training rows only)."""

import numpy as np
from sklearn import linear_model as lm
from sklearn import svm

from alpha.tournament.models.base import BaseTradingModel, Preprocessor, train_rows


class _Linear(BaseTradingModel):
    family = "linear"
    complexity = 1
    stride = 1
    max_rows: int | None = None

    def make(self):
        raise NotImplementedError

    def fit(self, X, y, train, val):
        Xc = self._cols(X)
        idx = train_rows(train, y, stride=self.params.get("stride", self.stride), max_rows=self.max_rows)
        self.pre = Preprocessor().fit(Xc.iloc[idx])
        self.m = self.make()
        yy = y.to_numpy()[idx]
        self.m.fit(self.pre.transform(Xc.iloc[idx]), yy.astype(int) if self.target.kind == "cls" else yy)
        self.feature_names = list(Xc.columns)
        return self

    def _raw(self, X, rows):
        Z = self.pre.transform(self._cols(X)[rows])
        if self.target.kind == "cls":
            return _proba3(self.m, Z)
        return self.m.predict(Z)

    def feature_importance(self):
        c = getattr(self.m, "coef_", None)
        if c is None:
            return None
        c = np.atleast_2d(c)
        imp = np.abs(c).mean(axis=0) if c.shape[0] > 1 else np.abs(c[0])
        return dict(zip(self.feature_names, map(float, imp)))


def _proba3(m, Z) -> np.ndarray:
    """(n, 3) probabilities over classes {0 short, 1 flat, 2 long}; missing classes get 0."""
    out = np.zeros((len(Z), 3))
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(Z)
    else:  # margin classifiers: softmax of the decision function
        d = m.decision_function(Z)
        d = np.column_stack([-d, d]) if d.ndim == 1 else d
        e = np.exp(d - d.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
    for j, c in enumerate(m.classes_):
        out[:, int(c)] = p[:, j]
    return out


class LinearReg(_Linear):
    name, target_kinds = "linear_regression", ("reg",)

    def make(self):
        return lm.LinearRegression()


class Ridge(_Linear):
    name, target_kinds = "ridge", ("reg",)

    def make(self):
        return lm.Ridge(alpha=self.params.get("alpha", 100.0))


class Lasso(_Linear):
    name, target_kinds = "lasso", ("reg",)

    def make(self):
        return lm.Lasso(alpha=self.params.get("alpha", 1e-3), max_iter=5000)


class ElasticNet(_Linear):
    name, target_kinds = "elasticnet", ("reg",)

    def make(self):
        return lm.ElasticNet(alpha=self.params.get("alpha", 1e-3), l1_ratio=self.params.get("l1_ratio", 0.5),
                             max_iter=5000)


class Logistic(_Linear):
    name, target_kinds = "logistic", ("cls",)

    def make(self):
        return lm.LogisticRegression(C=self.params.get("C", 0.01), max_iter=1000)


class LinearSVM(_Linear):
    name, target_kinds = "linear_svm", ("cls",)
    stride = 2

    def make(self):
        return svm.LinearSVC(C=self.params.get("C", 0.001), max_iter=5000)


class SVR(_Linear):
    name, target_kinds = "svr", ("reg",)
    max_rows = 15_000  # kernel SVR is O(n^2): fit on the most recent 15k training rows (stride 4)
    stride = 4

    def make(self):
        return svm.SVR(C=self.params.get("C", 0.1), epsilon=self.params.get("epsilon", 0.5), kernel="rbf",
                       gamma="scale", cache_size=1000)
