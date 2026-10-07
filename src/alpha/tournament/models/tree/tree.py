"""Tree / tabular models. NaNs are passed through where the library supports them, else median-imputed (train)."""

import numpy as np
from sklearn import ensemble as sk_ens
from sklearn import tree as sk_tree

from alpha.tournament.models.base import BaseTradingModel, Preprocessor, train_rows
from alpha.tournament.models.linear.linear import _proba3

THREADS = 4


class _Tree(BaseTradingModel):
    family = "tree"
    complexity = 2
    stride = 1
    needs_impute = False
    early_stopping = False

    def make(self, cls: bool):
        raise NotImplementedError

    def fit(self, X, y, train, val):
        Xc = self._cols(X)
        self.feature_names = list(Xc.columns)
        idx = train_rows(train, y, stride=self.params.get("stride", self.stride))
        cls = self.target.kind == "cls"
        self.pre = Preprocessor(clip=1e9).fit(Xc.iloc[idx]) if self.needs_impute else None
        A = self._arr(Xc.iloc[idx])
        yy = y.to_numpy()[idx]
        yy = yy.astype(int) if cls else yy
        self.m = self.make(cls)
        if self.early_stopping:
            vidx = train_rows(val, y)
            self._fit_es(A, yy, self._arr(Xc.iloc[vidx]), y.to_numpy()[vidx].astype(int) if cls else y.to_numpy()[vidx])
        else:
            self.m.fit(A, yy)
        return self

    def _fit_es(self, A, yy, Av, yv):
        self.m.fit(A, yy)

    def _arr(self, Xc):
        if self.pre is not None:
            return self.pre.transform(Xc)
        return Xc.to_numpy(np.float32)

    def _raw(self, X, rows):
        A = self._arr(self._cols(X)[rows])
        if self.target.kind == "cls":
            return _proba3(self.m, A)
        return self.m.predict(A)

    def feature_importance(self):
        imp = getattr(self.m, "feature_importances_", None)
        if imp is None:
            return None
        return dict(zip(self.feature_names, map(float, imp)))


class DecisionTree(_Tree):
    name = "decision_tree"
    needs_impute = True

    def make(self, cls):
        k = dict(max_depth=self.params.get("max_depth", 4), min_samples_leaf=self.params.get("min_leaf", 2000),
                 random_state=self.seed)
        return sk_tree.DecisionTreeClassifier(**k) if cls else sk_tree.DecisionTreeRegressor(**k)


class RandomForest(_Tree):
    name = "random_forest"
    stride = 4
    needs_impute = True

    def make(self, cls):
        k = dict(n_estimators=self.params.get("n_estimators", 200), max_depth=self.params.get("max_depth", 6),
                 min_samples_leaf=self.params.get("min_leaf", 500), max_features=0.3, n_jobs=THREADS,
                 random_state=self.seed)
        return sk_ens.RandomForestClassifier(**k) if cls else sk_ens.RandomForestRegressor(**k)


class ExtraTrees(_Tree):
    name = "extra_trees"
    stride = 4
    needs_impute = True

    def make(self, cls):
        k = dict(n_estimators=self.params.get("n_estimators", 200), max_depth=self.params.get("max_depth", 6),
                 min_samples_leaf=self.params.get("min_leaf", 500), max_features=0.3, n_jobs=THREADS,
                 random_state=self.seed)
        return sk_ens.ExtraTreesClassifier(**k) if cls else sk_ens.ExtraTreesRegressor(**k)


class GradientBoosting(_Tree):
    name = "gradient_boosting"
    stride = 8  # sklearn's exact GBM is slow; overlapping labels make neighbouring rows near-duplicates anyway
    needs_impute = True

    def make(self, cls):
        k = dict(n_estimators=self.params.get("n_estimators", 150), max_depth=self.params.get("max_depth", 3),
                 learning_rate=self.params.get("lr", 0.05), subsample=0.7, min_samples_leaf=300,
                 random_state=self.seed)
        return sk_ens.GradientBoostingClassifier(**k) if cls else sk_ens.GradientBoostingRegressor(**k)


class HistGB(_Tree):
    name = "hist_gb"
    early_stopping = True

    def make(self, cls):
        k = dict(max_iter=self.params.get("n_estimators", 300), learning_rate=self.params.get("lr", 0.03),
                 max_leaf_nodes=self.params.get("num_leaves", 15), min_samples_leaf=500, l2_regularization=5.0,
                 random_state=self.seed, early_stopping=False)
        return sk_ens.HistGradientBoostingClassifier(**k) if cls else sk_ens.HistGradientBoostingRegressor(**k)


class LightGBM(_Tree):
    name = "lightgbm"
    early_stopping = True

    def make(self, cls):
        import lightgbm as lgb

        k = dict(learning_rate=self.params.get("lr", 0.03), n_estimators=self.params.get("n_estimators", 400),
                 num_leaves=self.params.get("num_leaves", 15), min_child_samples=self.params.get("min_leaf", 500),
                 subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1,
                 random_state=self.seed, n_jobs=THREADS)
        return lgb.LGBMClassifier(**k) if cls else lgb.LGBMRegressor(**k)

    def _fit_es(self, A, yy, Av, yv):
        import lightgbm as lgb

        if len(Av) < 100:
            self.m.fit(A, yy)
            return
        self.m.fit(A, yy, eval_set=[(Av, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])


class XGBoost(_Tree):
    name = "xgboost"
    early_stopping = True

    def make(self, cls):
        import xgboost as xgb

        k = dict(learning_rate=self.params.get("lr", 0.03), n_estimators=self.params.get("n_estimators", 400),
                 max_depth=self.params.get("max_depth", 4), min_child_weight=self.params.get("min_leaf", 200),
                 subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=THREADS,
                 random_state=self.seed, early_stopping_rounds=50)
        return xgb.XGBClassifier(objective="multi:softprob", **k) if cls else xgb.XGBRegressor(**k)

    def _fit_es(self, A, yy, Av, yv):
        if self.target.kind == "cls":  # xgboost needs contiguous labels 0..K-1
            self.labels_ = np.unique(yy)
            remap = {c: i for i, c in enumerate(self.labels_)}
            yy = np.vectorize(remap.get)(yy)
            keep = np.isin(yv, self.labels_)
            Av, yv = Av[keep], np.vectorize(remap.get)(yv[keep]) if keep.any() else yv[keep]
        if len(Av) < 100:
            self.m.set_params(early_stopping_rounds=None)
            self.m.fit(A, yy, verbose=False)
            return
        self.m.fit(A, yy, eval_set=[(Av, yv)], verbose=False)

    def _raw(self, X, rows):
        A = self._arr(self._cols(X)[rows])
        if self.target.kind == "cls":
            p = self.m.predict_proba(A)
            out = np.zeros((len(A), 3))
            for j, c in enumerate(self.labels_):
                out[:, int(c)] = p[:, j]
            return out
        return self.m.predict(A)


class CatBoost(_Tree):
    name = "catboost"
    early_stopping = True

    def make(self, cls):
        import catboost as cb

        k = dict(learning_rate=self.params.get("lr", 0.05), iterations=self.params.get("n_estimators", 400),
                 depth=self.params.get("max_depth", 4), l2_leaf_reg=10.0, random_seed=self.seed, verbose=False,
                 thread_count=THREADS, allow_writing_files=False, od_type="Iter", od_wait=50)
        return cb.CatBoostClassifier(loss_function="MultiClass", **k) if cls else cb.CatBoostRegressor(**k)

    def _fit_es(self, A, yy, Av, yv):
        if len(Av) < 100 or (self.target.kind == "cls" and not np.isin(yv, np.unique(yy)).all()):
            self.m.fit(A, yy)
            return
        self.m.fit(A, yy, eval_set=(Av, yv), use_best_model=True)

    def _raw(self, X, rows):
        A = self._arr(self._cols(X)[rows])
        if self.target.kind == "cls":
            p = self.m.predict_proba(A)
            out = np.zeros((len(A), 3))
            for j, c in enumerate(self.m.classes_):
                out[:, int(c)] = p[:, j]
            return out
        return self.m.predict(A)
