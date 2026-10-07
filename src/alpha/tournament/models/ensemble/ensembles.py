"""Ensembles and stacking built ONLY from members' stored out-of-fold (DEV-OOS) output.

A member's test-fold scores / positions were produced by a model fit before that fold, so they are legitimate
inputs at that time. Anything an ensemble learns (weights, stacker, thresholds) for test fold k uses member
output from folds < k only (nested walk-forward):

    voting        sign of the majority of members' policy positions (>= `quorum` share agree); nothing learned
    prob_average  members' scores standardized by their own std over earlier folds, weighted by each member's
                  net P&L over earlier folds (softmax of positive Sharpe; equal if none positive); threshold and
                  hold chosen on fold k-1, applied to fold k (starts at fold 1)
    stacking      meta LightGBM on members' standardized scores, trained on folds <= k-2 (purged), threshold
                  chosen on fold k-1, applied to fold k (starts at fold 2)
Members are given as "<stage>.<entry>" and resolved to their latest finished experiment.
"""

import numpy as np
import pandas as pd

from alpha.tournament.backtesting import engine, policy
from alpha.tournament.models.base import BaseTradingModel
from alpha.tournament.targets.targets import Target
from alpha.tournament.validation.splits import GAP


class _Members(BaseTradingModel):
    family = "ensemble"
    complexity = 4
    input_kind = "members"
    target_kinds = ("reg", "cls")


class Voting(_Members):
    name = "voting"


class ProbAverage(_Members):
    name = "prob_average"


class Stacking(_Members):
    name = "stacking"


def resolve(repo, members: list[str]) -> list[str]:
    out = []
    for m in members:
        if m.count(".") >= 2:
            out.append(m)
            continue
        stage, entry = m.split(".", 1)
        r = repo.conn.execute("SELECT experiment_id FROM tournament.experiments WHERE stage = %s AND entry = %s "
                              "AND status = 'done' AND NOT shuffled_control ORDER BY finished_at DESC LIMIT 1",
                              (stage, entry)).fetchone()
        if r is None:
            raise LookupError(f"member {m} has no finished experiment")
        out.append(r[0])
    return out


def member_frames(repo, eids: list[str], index: pd.DatetimeIndex) -> dict:
    """score (bars x members), position (bars x members), fold (bars), h per member."""
    scores, pos, hs, fold = {}, {}, {}, None
    for e in eids:
        p = repo.oos_predictions(e)
        scores[e] = p["score"].reindex(index)
        hs[e] = int(p["target_h"].mode().iloc[0]) if len(p) else 4
        f = p["fold"].reindex(index)
        fold = f if fold is None else fold.fillna(f)
        sig = repo.df("SELECT bar_ts, position FROM tournament.signals WHERE experiment_id = %s AND variant = 'policy' "
                      "ORDER BY bar_ts", (e,))
        s = pd.Series(sig["position"].to_numpy(float), index=pd.DatetimeIndex(sig["bar_ts"])) if len(sig) \
            else pd.Series(dtype=float)
        pos[e] = s.reindex(index, method="ffill").fillna(0.0) if len(s) else pd.Series(0.0, index=index)
    return {"score": pd.DataFrame(scores), "pos": pd.DataFrame(pos), "h": hs, "fold": fold}


def _z_by_past_folds(S: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """Each member's score divided by its std over all earlier folds (fold 0 uses its own first half)."""
    out = S.copy() * np.nan
    ks = sorted(fold.dropna().unique())
    for k in ks:
        cur = fold == k
        past = fold < k
        if past.sum() < 100:
            past = cur & (np.cumsum(cur) <= cur.sum() / 2)
        sd = S[past].std().replace(0, np.nan)
        out[cur] = S[cur] / sd
    return out


def _choose(score: np.ndarray, m: engine.Market, holds: list[int]) -> tuple[float, int, float] | None:
    best = None
    for h in holds:
        for th in policy.candidate_thresholds(score):
            p = policy.positions(score, th, h)
            r = engine.run(p, m)
            net = float(r.bar_pnl.sum() * 1e4)
            if len(r.trades) >= 10 and (best is None or net > best[2]):
                best = (th, h, net)
    return best


def run_members_entry(ctx, spec, repo):
    from alpha.tournament.training import runner

    if repo is None:
        raise RuntimeError("ensembles need the tournament database (members' stored OOS output)")
    if spec.model == "meta":
        from alpha.tournament.models.meta.meta import run_meta

        return run_meta(ctx, spec, repo)
    eid, chash = runner.experiment_id(spec, ctx)
    params = spec.params[0]
    members = resolve(repo, params["members"])
    out = runner.Outcome(eid, spec, "running", n_configs=1)
    commit, dirty = runner.git_info()
    repo.upsert_model(spec.model, "ensemble", 4, "members", False, (Voting.__doc__ or ""))
    repo.start_experiment(dict(experiment_id=eid, stage=spec.stage, entry=spec.entry, model_id=spec.model,
                               dataset_id=ctx.dataset_id, grid=[{"members": members, **{k: v for k, v in params.items()
                                                                                       if k != "members"}}],
                               n_configs=1, cost_config=ctx.cost.to_json(), seed=spec.seed, git_commit=commit,
                               git_dirty_hash=dirty, config_hash=chash, status="running", shuffled_control=False))
    repo.members(eid, [(m, "member", None) for m in members])
    idx = ctx.feats.index
    oos_mask = np.zeros(len(idx), bool)
    fold_of = np.full(len(idx), -1)
    for f in ctx.folds:
        _, _, te = f.masks(idx)
        oos_mask |= te
        fold_of[te] = f.k
    oidx = idx[oos_mask]
    fr = member_frames(repo, members, oidx)
    fold = pd.Series(fold_of[oos_mask], index=oidx)
    pos_pol = np.zeros(len(idx))
    pos_forced = np.zeros(len(idx))
    holds = sorted(set(fr["h"].values()))
    folds_info, score_rows = [], []
    if spec.model == "voting":
        P = fr["pos"].to_numpy()
        q = float(params.get("quorum", 0.5))
        longs, shorts = (P > 0).mean(axis=1), (P < 0).mean(axis=1)
        v = np.where(longs > q, 1.0, np.where(shorts > q, -1.0, 0.0))
        pos_pol[oos_mask] = v
        pos_forced[oos_mask] = v
        score_rows.append(pd.DataFrame({"fold": fold.to_numpy(), "score": longs - shorts, "target_h": 1,
                                        "threshold": q, "p_short": np.nan, "p_flat": np.nan, "p_long": np.nan},
                                       index=oidx))
    else:
        Z = _z_by_past_folds(fr["score"], fold)
        y = ctx.target(Target("reg", max(holds))).reindex(oidx)
        start = 1 if spec.model == "prob_average" else 2
        for k in sorted(fold.unique()):
            if k < start:
                continue
            cur, prev = (fold == k).to_numpy(), (fold == k - 1).to_numpy()
            if spec.model == "prob_average":
                past = (fold < k).to_numpy()
                w = []
                for e in members:
                    pm = fr["pos"][e].to_numpy()[past]
                    mk = ctx.market.slice(np.isin(np.arange(len(idx)), np.flatnonzero(oos_mask)[past]))
                    bp = engine.run(pm, mk).bar_pnl
                    d = bp.groupby(bp.index.floor("D")).sum()
                    w.append(max(0.0, d.mean() / d.std()) if d.std() > 0 else 0.0)
                w = np.array(w)
                w = np.exp(w * 10) * (w > 0) if (w > 0).any() else np.ones(len(members))
                w = w / w.sum()
                s_all = (Z.fillna(0.0).to_numpy() * w).sum(axis=1)
                weights = dict(zip(members, w))
            else:
                import lightgbm as lgb

                # folds <= k-2, purged: labels must end before fold k-1 (where the threshold is chosen) starts
                tr = (fold <= k - 2).to_numpy() & (oidx < oidx[prev][0] - GAP) & y.notna().to_numpy()
                if tr.sum() < 1000:
                    continue
                st = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=500,
                                       subsample=0.7, subsample_freq=1, verbose=-1, random_state=spec.seed)
                st.fit(Z.to_numpy()[tr], y.to_numpy()[tr])
                s_all = st.predict(Z.to_numpy())
                weights = dict(zip(members, st.feature_importances_ / max(1, st.feature_importances_.sum())))
            prev_g = np.flatnonzero(oos_mask)[prev]
            mprev = ctx.market.slice(np.isin(np.arange(len(idx)), prev_g))
            ch = _choose(s_all[prev], mprev, holds)
            if ch is None:
                folds_info.append({"fold": k, "abstained": True})
                continue
            th, h, vnet = ch
            p = policy.positions(s_all[cur], th, h)
            g = np.flatnonzero(oos_mask)[cur]
            pos_forced[g] = p
            pos_pol[g] = p if vnet > 0 else 0.0
            folds_info.append({"fold": k, "threshold": th, "hold": h, "val_net_bps": vnet, "abstained": vnet <= 0,
                               **{f"w_{i}": wv for i, wv in enumerate(weights.values())}})
            score_rows.append(pd.DataFrame({"fold": k, "score": s_all[cur], "target_h": h, "threshold": th,
                                            "p_short": np.nan, "p_flat": np.nan, "p_long": np.nan}, index=oidx[cur]))
    runner.evaluate(ctx, out, pos_pol, pos_forced, oos_mask, fold_of, 4,
                    pd.concat(score_rows) if score_rows else None)
    out.status = "done"
    out.folds = folds_info
    wf = []
    for fi in folds_info:
        f = ctx.folds[int(fi["fold"])]
        wf.append((eid, int(fi["fold"]), f.test_start, f.test_end, 0, fi.get("threshold"), bool(fi["abstained"]),
                   0, None, fi.get("val_net_bps"), 0, None, None))
    runner._persist(repo, out, [], [], wf, out.oos_index)
    return out
