"""Meta-labeling (two stages).

    stage 1  candidate trades = a base experiment's DEV-OOS 'forced' positions (its validation-best signals,
             including folds where its own policy abstained)
    stage 2  a classifier decides TAKE / SKIP for each candidate, trained on candidates' NET outcome
             (net_bps > 0 after fees, slippage and funding) from folds <= k-2 whose trades closed before fold
             k-1 starts minus the gap; the TAKE probability threshold is chosen on fold k-1; applied to fold k.
    Meta features: the bar's core features at the decision, the base score, the side.
Meta models: logistic, lightgbm, xgboost, catboost, random_forest, mlp.
"""

import numpy as np
import pandas as pd

from alpha.tournament.backtesting import engine
from alpha.tournament.features import groups as fg
from alpha.tournament.models.base import BaseTradingModel, Preprocessor
from alpha.tournament.validation.splits import GAP

BAR = pd.Timedelta(minutes=15)


class MetaLabel(BaseTradingModel):
    name = "meta"
    family = "meta"
    complexity = 3
    input_kind = "members"


def _clf(name: str, seed: int):
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=0.05, max_iter=2000)
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50,
                                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=seed)
    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, min_child_weight=20,
                                 subsample=0.7, colsample_bytree=0.7, random_state=seed, n_jobs=4)
    if name == "catboost":
        import catboost as cb

        return cb.CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=False, random_seed=seed,
                                     allow_writing_files=False, thread_count=4)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=30, random_state=seed, n_jobs=4)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=500, early_stopping=True,
                             random_state=seed)
    raise ValueError(name)


def run_meta(ctx, spec, repo):
    from alpha.tournament.models.ensemble.ensembles import member_frames, resolve
    from alpha.tournament.training import runner

    params = spec.params[0]
    base = resolve(repo, [params["base"]])[0]
    eid, chash = runner.experiment_id(spec, ctx)
    out = runner.Outcome(eid, spec, "running", n_configs=1)
    commit, dirty = runner.git_info()
    repo.upsert_model("meta", "meta", 3, "members", False, (__doc__ or "").strip().split("\n")[0])
    repo.start_experiment(dict(experiment_id=eid, stage=spec.stage, entry=spec.entry, model_id="meta",
                               dataset_id=ctx.dataset_id, grid=[{"base": base, "meta_model": params["meta_model"]}],
                               n_configs=1, cost_config=ctx.cost.to_json(), seed=spec.seed, git_commit=commit,
                               git_dirty_hash=dirty, config_hash=chash, status="running", shuffled_control=False))
    repo.members(eid, [(base, "base", None)])
    idx = ctx.feats.index
    oos_mask = np.zeros(len(idx), bool)
    fold_of = np.full(len(idx), -1)
    for f in ctx.folds:
        _, _, te = f.masks(idx)
        oos_mask |= te
        fold_of[te] = f.k
    oidx = idx[oos_mask]
    g_oos = np.flatnonzero(oos_mask)
    fold = pd.Series(fold_of[oos_mask], index=oidx)
    # stage 1: the base's forced positions, rebuilt from its stored scores and per-fold thresholds
    pred = repo.oos_predictions(base)
    wf = repo.df("SELECT fold, threshold FROM tournament.walk_forward_runs WHERE experiment_id = %s", (base,))
    from alpha.tournament.analytics.robustness import _positions

    pred = pred.join(wf.set_index("fold")["threshold"], on="fold")
    cand = _positions(pred.dropna(subset=["threshold"]), oidx)
    m_oos = ctx.market.slice(oos_mask)
    tr = engine.run(cand, m_oos).trades
    score = pred["score"].reindex(oidx).to_numpy()
    cols = fg.columns_for("core", ctx.group_cols)
    dec = pd.DatetimeIndex(tr["entry_ts"]) - BAR
    loc = oidx.get_indexer(dec)
    F = ctx.feats[cols].reindex(dec).reset_index(drop=True)
    F["base_score"] = score[loc]
    F["side"] = tr["side"].to_numpy()
    F.index = dec
    lab = (tr["net_bps"].to_numpy() > 0).astype(int)
    t_fold = fold.reindex(dec).to_numpy()
    exit_ts = pd.DatetimeIndex(tr["exit_ts"])
    pos_pol = np.zeros(len(idx))
    pos_forced = np.zeros(len(idx))
    folds_info, score_rows = [], []
    p_all = np.full(len(tr), np.nan)
    for k in sorted(fold.unique()):
        if k < 2:
            continue
        cur_bars = (fold == k).to_numpy()
        prev_start = oidx[(fold == k - 1).to_numpy()][0]
        trn = (t_fold <= k - 2) & (exit_ts < prev_start - GAP)
        val = t_fold == k - 1
        tst = t_fold == k
        if trn.sum() < 50 or len(np.unique(lab[trn])) < 2:
            folds_info.append({"fold": k, "abstained": True})
            continue
        pre = Preprocessor().fit(F[trn])
        clf = _clf(params["meta_model"], spec.seed).fit(pre.transform(F[trn]), lab[trn])
        p_val = clf.predict_proba(pre.transform(F[val]))[:, 1] if val.any() else np.array([])
        p_tst = clf.predict_proba(pre.transform(F[tst]))[:, 1] if tst.any() else np.array([])
        p_all[tst] = p_tst
        best = None
        for th in (0.0, *np.quantile(p_val, [0.25, 0.5, 0.75]).tolist()) if len(p_val) >= 10 else (0.0,):
            take = p_val >= th
            net = float(tr["net_bps"].to_numpy()[val][take].sum())
            if take.sum() >= 5 and (best is None or net > best[1]):
                best = (th, net)
        # fold-k positions: the base candidates, with SKIPPED trades removed
        p_fold = cand[cur_bars].copy()
        if best is None:
            folds_info.append({"fold": k, "abstained": True})
            pos_forced[g_oos[cur_bars]] = p_fold
            continue
        th, vnet = best
        cur_start = np.flatnonzero(cur_bars)[0]
        for j in np.flatnonzero(tst):
            if p_all[j] < th:
                a = loc[j] - cur_start
                b = a + int(tr["bars"].iloc[j])
                p_fold[max(0, a):max(0, min(len(p_fold), b))] = 0.0
        pos_forced[g_oos[cur_bars]] = p_fold
        pos_pol[g_oos[cur_bars]] = p_fold if vnet > 0 else 0.0
        folds_info.append({"fold": k, "threshold": th, "val_net_bps": vnet, "abstained": vnet <= 0,
                           "candidates": int(tst.sum()), "taken": int((p_tst >= th).sum())})
        score_rows.append(pd.DataFrame({"fold": k, "score": score[cur_bars], "target_h": 1, "threshold": th,
                                        "p_short": np.nan, "p_flat": np.nan, "p_long": np.nan}, index=oidx[cur_bars]))
    runner.evaluate(ctx, out, pos_pol, pos_forced, oos_mask, fold_of, 3,
                    pd.concat(score_rows) if score_rows else None)
    out.status = "done"
    out.folds = folds_info
    wfr = []
    for fi in folds_info:
        f = ctx.folds[int(fi["fold"])]
        wfr.append((eid, int(fi["fold"]), f.test_start, f.test_end, 0, fi.get("threshold"), bool(fi["abstained"]), 0,
                    None, fi.get("val_net_bps"), 0, None, None))
    runner._persist(repo, out, [], [], wfr, out.oos_index)
    return out
