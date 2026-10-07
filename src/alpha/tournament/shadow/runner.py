"""Forward shadow trading of frozen tournament finalists: the only true out-of-sample test left.

    uv run python -m alpha.tournament shadow [--once]

Finalists are frozen in data/experiments/sol15_finalists.json (written by hand or by the selection step, at most
3, only candidates that PASSED every DEV gate). Each is opened once in the lockbox (split SOL15-FORWARD). For a
finalist the frozen model is its last walk-forward fold's artifact with that fold's threshold and hold.

Every closed 15m bar after the freeze: build features on the latest bars (allow_forward=True: this module is the
only caller allowed past DEV_END), score, apply the shared policy, and append to tournament.signals / trades with
variant 'shadow'. Fills are simulated with the same cost model; no order is ever sent anywhere.
Gate after >= 150 trades or 8 weeks: net > 0, PF > 1, realized fill cost <= 1.5x modelled.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha.research.splits import LockboxError, open_lockbox
from alpha.tournament.backtesting import engine, policy
from alpha.tournament.data import dataset as dsm
from alpha.tournament.database.repo import Repo
from alpha.tournament.execution.fills import CostConfig, side_costs
from alpha.tournament.features import groups as fg
from alpha.tournament.models.base import BaseTradingModel
from alpha.tournament.targets.targets import score_scale

FINALISTS = Path("data/experiments/sol15_finalists.json")
HISTORY_DAYS = 60  # features need <= 14 days of warm-up (EMA 1344); 60 is ample


def load_finalists() -> list[dict]:
    if not FINALISTS.exists():
        return []
    return json.loads(FINALISTS.read_text()).get("finalists", [])[:3]


def step(conn, fin: dict, since: pd.Timestamp) -> int:
    repo = Repo(conn)
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    raw = dsm.load_raw(conn, end=end, allow_forward=True, start=end - pd.Timedelta(days=HISTORY_DAYS))
    feats, _ = fg.build_features(raw)
    model: BaseTradingModel = BaseTradingModel.load(Path(fin["model_path"]))
    model.context = type("C", (), {"raw": raw})()  # series models read raw closes
    rows = np.asarray(feats.index >= since)
    raw_pred = model.predict(feats, rows)
    scale = (np.ones(len(feats)) if getattr(model, "output_units", "target") == "bps"
             else score_scale(model.target, raw["sol_close"]).to_numpy())
    pos = np.zeros(len(feats))
    pos[rows] = policy.positions((raw_pred * scale)[rows], float(fin["threshold"]), int(fin["hold"]))
    funding = dsm.funding_events(conn, end=end, allow_forward=True)
    m = engine.make_market(raw["sol_open"], side_costs(raw, CostConfig()), funding).slice(rows)
    r = engine.run(pos[rows], m)
    eid = fin["experiment_id"]
    repo.conn.execute("DELETE FROM tournament.signals WHERE experiment_id = %s AND variant = 'shadow'", (eid,))
    repo.conn.execute("DELETE FROM tournament.trades WHERE experiment_id = %s AND variant = 'shadow'", (eid,))
    repo.signals(eid, "shadow", feats.index[rows], pos[rows])
    repo.trades(eid, "shadow", r.trades)
    repo.commit()
    return len(r.trades)


def run(conn, once: bool = False) -> None:
    fins = load_finalists()
    if not fins:
        print("No frozen finalists (data/experiments/sol15_finalists.json): the tournament found no candidate that "
              "passed every DEV gate, so there is nothing to shadow-trade.")
        return
    for f in fins:
        try:
            open_lockbox(f["experiment_id"], "SOL15-FORWARD")
        except LockboxError:
            pass  # already opened at an earlier start: shadow continues, it is not a second look at fixed data
    since = pd.Timestamp(json.loads(FINALISTS.read_text())["frozen_at"])
    while True:
        for f in fins:
            n = step(conn, f, since)
            logger.info("shadow {}: {} trades since {}", f["experiment_id"], n, since)
        if once:
            return
        nxt = pd.Timestamp.now(tz="UTC").floor("15min") + pd.Timedelta(minutes=15, seconds=20)
        time.sleep(max(5.0, (nxt - pd.Timestamp.now(tz="UTC")).total_seconds()))
