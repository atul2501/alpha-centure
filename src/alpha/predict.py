"""Live predictor + scorer: uv run python -m alpha.predict

- LISTENs for `candle_closed` (sent by the collector), waits a few seconds so context candles (BTC, 4h) land,
  then evaluates the bar with exactly the research code path:
  build_frame -> all_setups -> with_features -> regime filter -> meta.score -> decide().
- Writes every candidate setup (taken or PASS, with reason) to `signals`, or one 'none' row if nothing fired.
- Every few minutes, scores finished setups with label_setups() (PASS rows too: counterfactual).
- Picks up a newly promoted champion automatically.

--replay-hours N evaluates the last N hours of closed bars once (no live data needed) and exits.
"""

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg
from loguru import logger
from psycopg.types.json import Jsonb

from alpha.binance.parse import interval_td, ms_to_dt
from alpha.collectors.klines import CANDLE_CLOSED_CHANNEL
from alpha.config import PERP_SUFFIX, Settings, get_settings
from alpha.db import TABLES
from alpha.decision import PASS, decide
from alpha.features.build import CORE_COLS, _df, base_symbol, build_frame, load_candles
from alpha.models.bundle import Bundle, champion_row, load_champion
from alpha.models.meta import GEOMETRY_COLS
from alpha.regime.hmm import REGIME_TF, REGIMES, attach_regime, regime_features
from alpha.research.dataset import TRADE_TFS, with_features
from alpha.strategies.labels import EXIT_MENU, Costs, Entry, label_setups
from alpha.strategies.playbook import all_setups

LIVE_HISTORY_BARS = 400
REGIME_HISTORY_BARS = 1000  # forward filter forgets its start state quickly; ~166 days of 4h is plenty
SETTLE_SECONDS = 3
STALE_AFTER = timedelta(minutes=3)
MAX_SPREAD_BPS = 10.0
SCORE_EVERY_S = 300
CHAMPION_CHECK_S = 300
SIGNAL_COLS = TABLES["signals"][0]


def _clean(v):
    if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def open_book(conn, version: str, at: datetime) -> tuple[set[str], dict[int, int], float]:
    """State of taken trades at `at`, mirroring backtest.simulate():
    (symbols with an open trade, open trade count per side, realized R today from trades closed by `at`)."""
    rows = conn.execute("""SELECT symbol, tf, bar_time, max_bars, exit_time, scored_at, side, r FROM signals
                           WHERE model_version = %s AND action <> 'PASS' AND bar_time < %s
                             AND bar_time > %s - interval '30 days'""", (version, at, at)).fetchall()
    busy, sides, today = set(), {1: 0, -1: 0}, 0.0
    for sym, tf, bt, mb, exit_time, scored, side, r in rows:
        still_open = (scored is None and bt + interval_td(tf) * (mb * 2 + 1) > at) or \
                     (scored is not None and exit_time is not None and exit_time >= at)
        if still_open:
            busy.add(sym)
            sides[int(side)] += 1
        elif scored is not None and exit_time is not None and exit_time.date() == at.date() and r is not None:
            today += r
    return busy, sides, today


def busy_symbols(conn, version: str, at: datetime) -> set[str]:
    return open_book(conn, version, at)[0]


def evaluate_bar(conn, bundle: Bundle, status: str, symbol: str, tf: str, bar_time: datetime,
                 live: bool = True) -> pd.DataFrame | None:
    step = interval_td(tf)
    f = build_frame(conn, symbol, tf, start=bar_time - step * LIVE_HISTORY_BARS, end=bar_time + step)
    if f.empty or f.index[-1] != pd.Timestamp(bar_time):
        logger.warning("{} {} {}: bar not in DB yet", symbol, tf, bar_time)
        return None
    bar = f.iloc[-1]
    base = {"symbol": symbol, "tf": tf, "bar_time": f.index[-1], "model_version": bundle.version}
    s = all_setups(f, tf)
    s = s[s.index == f.index[-1]]
    if s.empty:
        return pd.DataFrame([{**base, "strategy": "none", "side": 0, "action": PASS, "reason": "no_setup",
                              "close_px": bar["close"]}])

    rows = with_features(s, f, symbol, tf, CORE_COLS)
    reg_start = bar_time - interval_td(REGIME_TF) * REGIME_HISTORY_BARS
    reg = regime_features(load_candles(conn, symbol, REGIME_TF, reg_start, bar_time + step))
    rows = attach_regime(rows, {base_symbol(symbol): bundle.regime.filter(reg)})
    scored = bundle.meta.score(rows)

    gate = None
    if status == "degraded":
        gate = "model_degraded"
    elif live and datetime.now(timezone.utc) - bar["close_time"] > STALE_AFTER:
        gate = "stale_data"
    elif live:
        sp = conn.execute("""SELECT spread_bps FROM orderbook_snap WHERE symbol = %s AND ts > now() - interval '5 minutes'
                             ORDER BY ts DESC LIMIT 1""", (symbol,)).fetchone()
        if sp and sp[0] > MAX_SPREAD_BPS:
            gate = "wide_spread"
    busy, sides, today_r = open_book(conn, bundle.version, bar_time)
    daily_loss_r = bundle.metrics.get("daily_loss_r")
    if not gate and daily_loss_r is not None and today_r <= -daily_loss_r:
        gate = "daily_loss_limit"
    if gate:
        d = scored.assign(action=PASS, reason=gate)
    else:
        d = decide(scored, bundle.policy, busy, open_sides=sides)

    feat_cols = CORE_COLS + GEOMETRY_COLS
    out = d.reset_index(drop=True).assign(**{k: v for k, v in base.items() if k != "symbol"})
    out["regime_probs"] = [{r: _clean(row.get(f"p_{r}")) for r in REGIMES} for _, row in d.iterrows()]
    out["features"] = [{c: _clean(row.get(c)) for c in feat_cols} for _, row in d.iterrows()]
    out["max_bars"] = d["max_bars"].astype(int).to_numpy()
    out["exit_policy"] = [bundle.policy.exits.get(st, "fixed") for st in d["strategy"]]
    return out


def write_signals(conn, rows: pd.DataFrame) -> None:
    cols = list(SIGNAL_COLS)
    ph = ", ".join(["%s"] * len(cols))
    pk = TABLES["signals"][1]
    upd = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in pk)
    sql = f"INSERT INTO signals ({', '.join(cols)}) VALUES ({ph}) ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {upd}"
    data = []
    for _, r in rows.iterrows():
        vals = []
        for c in cols:
            v = r.get(c)
            if c == "exit_policy" and (v is None or v != v):
                v = "fixed"  # 'none' rows (no setup) carry no exit policy
            vals.append(Jsonb(v) if c in ("regime_probs", "features") and v is not None else _clean(v))
        data.append(vals)
    with conn.cursor() as cur:
        cur.executemany(sql, data)


def score_pending(conn, costs: Costs = Costs(), entry: Entry = Entry()) -> int:
    """Fill outcomes of setups whose result is now known (taken and PASS alike), each with its own exit policy."""
    pending = _df(conn, """SELECT symbol, tf, bar_time, strategy, side, model_version, stop, target, max_bars,
                                  exit_policy
                           FROM signals WHERE scored_at IS NULL AND strategy <> 'none'
                           AND bar_time > now() - interval '30 days'""")
    n = 0
    for (symbol, tf, exit_name), g in pending.groupby(["symbol", "tf", "exit_policy"]):
        step = interval_td(tf)
        f = build_frame(conn, symbol, tf, start=g["bar_time"].min().to_pydatetime() - step)
        if f.empty:
            continue
        s = g.set_index(pd.DatetimeIndex(g["bar_time"]))
        lab = label_setups(s[["strategy", "side", "stop", "target", "max_bars"]].assign(
            model_version=s["model_version"]), f, costs, EXIT_MENU.get(exit_name, EXIT_MENU["fixed"]), entry)
        for t, r in lab.iterrows():
            conn.execute("""UPDATE signals SET outcome=%s, entry=%s, exit=%s, exit_time=%s, r=%s, pnl=%s, scored_at=now()
                            WHERE symbol=%s AND tf=%s AND bar_time=%s AND strategy=%s AND side=%s AND model_version=%s""",
                         (r["outcome"], float(r["entry"]), float(r["exit"]), r["exit_time"].to_pydatetime(),
                          float(r["r"]), float(r["pnl"]), symbol, tf, t.to_pydatetime(), r["strategy"], int(r["side"]),
                          r["model_version"]))
            n += 1
    return n


def execution(bundle: Bundle) -> tuple[Costs, Entry]:
    """Costs + entry mode the champion was trained with (market or maker), so live scoring matches research."""
    from alpha.research.experiment import DATASETS

    d = DATASETS.get(bundle.metrics.get("dataset", "market"), DATASETS["market"])
    return d["costs"], d["entry"]


def wanted(settings: Settings, symbol: str, tf: str) -> bool:
    return symbol.endswith(PERP_SUFFIX) and base_symbol(symbol) in settings.symbols and tf in TRADE_TFS


def run_live(settings: Settings) -> None:
    listen = psycopg.connect(settings.database_url, autocommit=True)
    conn = psycopg.connect(settings.database_url, autocommit=True)
    listen.execute(f"LISTEN {CANDLE_CLOSED_CHANNEL}")
    bundle, status = _wait_for_champion(conn)
    logger.info("predictor up with model {} ({})", bundle.version, status)
    queue: list[tuple[float, str, str, datetime]] = []
    last_score = last_champ = 0.0
    while True:
        for n in listen.notifies(timeout=1.0):
            symbol, tf, open_ms = n.payload.split("|")
            if wanted(settings, symbol, tf) and tf in bundle.metrics.get("tfs", TRADE_TFS):
                queue.append((time.time() + SETTLE_SECONDS, symbol, tf, ms_to_dt(open_ms)))
        due = [q for q in queue if q[0] <= time.time()]
        queue = [q for q in queue if q[0] > time.time()]
        for _, symbol, tf, bar_time in due:
            try:
                rows = evaluate_bar(conn, bundle, status, symbol, tf, bar_time)
                if rows is not None:
                    write_signals(conn, rows)
                    taken = rows[rows["action"] != PASS]
                    logger.info("{} {} {}: {}", symbol, tf, bar_time.strftime("%H:%M"),
                                ", ".join(f"{r.action} {r.strategy} ev={r.ev_r:.2f}" for r in taken.itertuples())
                                or f"PASS ({', '.join(sorted(set(rows['reason'])))})")
            except Exception:
                logger.exception("evaluate {} {} {} failed", symbol, tf, bar_time)
        if time.time() - last_score > SCORE_EVERY_S:
            last_score = time.time()
            try:
                k = score_pending(conn, *execution(bundle))
                if k:
                    logger.info("scored {} setups", k)
            except Exception:
                logger.exception("scoring failed")
        if time.time() - last_champ > CHAMPION_CHECK_S:
            last_champ = time.time()
            row = champion_row(conn)
            if row and (row["version"] != bundle.version or row["status"] != status):
                bundle, status = _wait_for_champion(conn)
                logger.info("now using model {} ({})", bundle.version, status)


def _wait_for_champion(conn) -> tuple[Bundle, str]:
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = load_champion(conn)
        if loaded:
            return loaded
        logger.warning("no champion model yet (run: uv run python -m alpha.trainer). retrying in 60s")
        time.sleep(60)


def replay(settings: Settings, hours: int) -> None:
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        bundle, status = _wait_for_champion(conn)
        end = datetime.now(timezone.utc)
        n = 0
        for tf in bundle.metrics.get("tfs", TRADE_TFS):
            for sym in settings.symbols:
                symbol = sym + PERP_SUFFIX
                times = conn.execute("""SELECT open_time FROM candles WHERE symbol=%s AND interval=%s
                                        AND open_time >= %s ORDER BY open_time""",
                                     (symbol, tf, end - timedelta(hours=hours))).fetchall()
                for (bt,) in times:
                    rows = evaluate_bar(conn, bundle, status, symbol, tf, bt, live=False)
                    if rows is not None:
                        write_signals(conn, rows)
                        n += len(rows)
        logger.info("replayed {}h: {} signal rows; scored {}", hours, n, score_pending(conn, *execution(bundle)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-hours", type=int, default=0)
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    s = get_settings()
    if a.replay_hours:
        replay(s, a.replay_hours)
    else:
        run_live(s)


if __name__ == "__main__":
    main()
