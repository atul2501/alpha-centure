"""Self-training loop.

Weekly (alpha-trainer.timer):   uv run python -m alpha.trainer
  1. Train a challenger on everything whose outcome was known before (now - holdout).
  2. Evaluate it out-of-sample on the holdout weeks.
  3. Compare with the champion's LIVE scored record over the same weeks (truly out-of-sample for it too).
  4. Promote only if the challenger is better; the promoted model is then refit on all data and saved.
Daily (alpha-drift.timer):      uv run python -m alpha.trainer --drift
  Marks the champion 'degraded' (predictor then PASSes everything) if live results or feature drift are bad,
  and triggers an early retrain.
Flags: --dry-run (train + report, change nothing), --holdout-days N, --force (promote regardless)
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg
from loguru import logger

from alpha.backtest.walkforward import WFConfig, fit_fold, latest_exit_time, metrics, simulate
from alpha.config import Settings, get_settings
from alpha.decision import PASS
from alpha.models.bundle import (DRIFT_FEATURES, Bundle, champion_row, drift_reference, load_champion, psi, register,
                                 set_status)
from alpha.models.meta import MetaModel
from alpha.regime.hmm import RegimeModel, attach_regime, load_regime_frames
from alpha.research.dataset import build_dataset
from alpha.strategies.labels import EXIT_MENU, apply_exit

MIN_HOLDOUT_TRADES = 5
DEGRADED_MIN_TRADES = 10
DEGRADED_AVG_R = -0.3
PSI_LIMIT = 0.25


def production_config(settings: Settings) -> tuple[str, WFConfig]:
    """(dataset kind, config) used in production, picked from the experiment harness by name."""
    from alpha.research.experiment import CONFIGS

    return CONFIGS[settings.production_config]


def fit_bundle(ds_raw: pd.DataFrame, frames: dict, end: pd.Timestamp, cfg: WFConfig = WFConfig(),
               seed: int = 0) -> Bundle:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rm = RegimeModel.fit({k: v[v["close_time"] < end] for k, v in frames.items()}, seed=seed)
    ds = attach_regime(ds_raw, {k: rm.filter(v) for k, v in frames.items()})
    train = ds[(latest_exit_time(ds) < end).to_numpy()]
    fm = fit_fold(train, cfg, end, seed)
    b = Bundle(Bundle.new_version(), rm, fm.meta, fm.policy, end.to_pydatetime())
    trained = apply_exit(train, fm.exits) if fm.exits else train
    b.metrics = {**fm.info, "config": cfg.name, "regime_states": rm.names, "exits": fm.exits,
                 "top_features": fm.meta.importance(10).round(1).to_dict(), "drift_ref": drift_reference(trained),
                 "daily_loss_r": cfg.daily_loss_r}
    return b


def evaluate(b: Bundle, ds_raw: pd.DataFrame, frames: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    rows = ds_raw[(ds_raw.index >= start) & (ds_raw.index < end)]
    if rows.empty:
        return metrics(rows)
    rows = apply_exit(rows, b.policy.exits) if b.policy.exits else rows.assign(exit_policy="fixed")
    ds = attach_regime(rows, {k: b.regime.filter(v) for k, v in frames.items()})
    sim = simulate(b.meta.score(ds), b.policy, True, True, b.metrics.get("daily_loss_r"))
    return metrics(sim[sim["action"] != PASS], b.policy.risk_per_trade, max(1, (end - start).days))


def live_record(conn: psycopg.Connection, version: str, since: datetime) -> dict:
    rows = conn.execute("""SELECT r, exit_time FROM signals WHERE model_version = %s AND action <> 'PASS'
                           AND scored_at IS NOT NULL AND bar_time >= %s""", (version, since)).fetchall()
    t = pd.DataFrame(rows, columns=["r", "exit_time"])
    if t.empty:
        return metrics(t)
    t["win"] = t["r"] > 0
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    return metrics(t, 0.01, max(1, (datetime.now(timezone.utc) - since).days))


def should_promote(challenger: dict, champion: dict | None) -> tuple[bool, str]:
    if champion is None:
        return True, "no champion yet"
    if challenger["trades"] < MIN_HOLDOUT_TRADES:
        return False, f"challenger made {challenger['trades']} holdout trades (< {MIN_HOLDOUT_TRADES}): not enough evidence"
    champ_r = champion["avg_r"] if champion["trades"] >= MIN_HOLDOUT_TRADES else 0.0
    if challenger["avg_r"] <= max(0.0, champ_r):
        return False, f"challenger avg R {challenger['avg_r']:.3f} <= max(0, champion {champ_r:.3f})"
    return True, f"challenger avg R {challenger['avg_r']:.3f} beats champion {champ_r:.3f}"


def run_weekly(settings: Settings, holdout_days: int = 28, dry_run: bool = False, force: bool = False) -> None:
    now = pd.Timestamp.now(tz="UTC")
    cut = now - pd.Timedelta(days=holdout_days)
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        kind, cfg = production_config(settings)
        from alpha.research.experiment import DATASETS, TFS

        d = DATASETS[kind]
        logger.info("building dataset ({} entries, {} timeframes, config {})", kind, ",".join(TFS), cfg.name)
        ds_raw = build_dataset(conn, settings, tfs=TFS, costs=d["costs"], entry=d["entry"],
                               exits=dict(EXIT_MENU) if cfg.exits != "fixed" else None)
        frames = load_regime_frames(conn, settings.symbols)
        logger.info("dataset: {} labeled setups", len(ds_raw))

        challenger = fit_bundle(ds_raw, frames, cut, cfg)
        challenger.metrics.update(dataset=kind, tfs=list(TFS))
        ch_m = evaluate(challenger, ds_raw, frames, cut, now)
        champ = champion_row(conn)
        champ_m = live_record(conn, champ["version"], cut.to_pydatetime()) if champ else None
        ok, why = should_promote(ch_m, champ_m)
        if force:
            ok, why = True, f"forced ({why})"
        logger.info("challenger holdout: {} | champion live: {} | promote={} ({})", _fmt(ch_m),
                    _fmt(champ_m) if champ_m else "-", ok, why)
        if dry_run:
            logger.info("dry run: nothing saved")
            return
        if not ok:
            challenger.metrics.update(holdout=ch_m, champion_live=champ_m, decision=why)
            register(conn, challenger, "challenger", "", note=why)
            return
        final = fit_bundle(ds_raw, frames, now, cfg)  # same recipe, all data
        final.metrics.update(dataset=kind, tfs=list(TFS))
        final.metrics.update(holdout=ch_m, champion_live=champ_m, decision=why)
        path = final.save(settings.models_dir)
        if champ:
            set_status(conn, champ["version"], "retired", note=f"replaced by {final.version}")
        register(conn, final, "champion", str(path.resolve()), note=why)
        logger.info("promoted {} -> {}", final.version, path)


def drift_check(settings: Settings, days: int = 7) -> bool:
    """Returns True if the champion was marked degraded."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        loaded = load_champion(conn)
        if not loaded:
            logger.info("no champion")
            return False
        b, status = loaded
        live = live_record(conn, b.version, since)
        problems = []
        if live["trades"] >= DEGRADED_MIN_TRADES and live["avg_r"] < DEGRADED_AVG_R:
            problems.append(f"live avg R {live['avg_r']:.2f} over {live['trades']} trades")
        feats = conn.execute("""SELECT features FROM signals WHERE model_version = %s AND strategy <> 'none'
                                AND bar_time >= %s""", (b.version, since)).fetchall()
        if len(feats) >= 200:
            fdf = pd.DataFrame([f[0] for f in feats])
            ref = b.metrics.get("drift_ref", {})
            scores = {c: psi(ref[c], fdf[c].dropna().to_numpy(dtype=float)) for c in DRIFT_FEATURES
                      if c in ref and c in fdf}
            drifted = {c: round(v, 3) for c, v in scores.items() if v > PSI_LIMIT}
            if len(drifted) >= 2:
                problems.append(f"feature drift PSI {drifted}")
        if problems:
            set_status(conn, b.version, "degraded", note="; ".join(problems))
            logger.warning("champion {} degraded: {}", b.version, problems)
            return True
        if status == "degraded":
            logger.info("champion {} still degraded (waiting for retrain)", b.version)
        else:
            logger.info("champion {} healthy: {}", b.version, _fmt(live))
        return False


def _fmt(m: dict) -> str:
    return f"{m['trades']} trades, avg R {m['avg_r']:.3f}, total {m['total_r']:.1f}R" if m["trades"] else "0 trades"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--holdout-days", type=int, default=28)
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
               filter=lambda r: "USDT.P" not in r["message"])
    s = get_settings()
    if a.drift:
        if drift_check(s):
            run_weekly(s, a.holdout_days)  # early retrain
        return
    run_weekly(s, a.holdout_days, a.dry_run, a.force)


if __name__ == "__main__":
    main()
