"""Postgres writers / readers for the tournament schema (psycopg 3, COPY for bulk rows)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

SQL = Path(__file__).resolve().parents[4] / "sql" / "008_tournament.sql"


def _j(x) -> str:
    return json.dumps(x, default=_default)


def _default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _f(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def clean_json(d):
    """JSON-safe: NaN/inf -> null (Postgres jsonb rejects NaN)."""
    if isinstance(d, dict):
        return {str(k): clean_json(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [clean_json(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return _f(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    if isinstance(d, pd.Timestamp):
        return d.isoformat()
    return d


class Repo:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    @staticmethod
    def init_schema(conn: psycopg.Connection) -> None:
        conn.execute(SQL.read_text())
        conn.commit()

    # ---- catalogue ----
    def upsert_model(self, model_id, family, complexity, input_kind, experimental, description):
        self.conn.execute("""INSERT INTO tournament.models VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (model_id) DO UPDATE SET family = EXCLUDED.family, complexity = EXCLUDED.complexity,
            input_kind = EXCLUDED.input_kind, experimental = EXCLUDED.experimental, description = EXCLUDED.description""",
                          (model_id, family, complexity, input_kind, experimental, description))

    def upsert_model_version(self, model_version_id, model_id, code_hash):
        self.conn.execute("INSERT INTO tournament.model_versions (model_version_id, model_id, code_hash) "
                          "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING", (model_version_id, model_id, code_hash))

    def upsert_feature_set(self, fs_id, name, groups, columns, eval_start):
        self.conn.execute("INSERT INTO tournament.feature_sets (feature_set_id, name, groups, columns, eval_start) "
                          "VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", (fs_id, name, groups, columns, eval_start))

    def upsert_hp(self, hp_id, model_id, params):
        self.conn.execute("INSERT INTO tournament.hyperparameters VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                          (hp_id, model_id, _j(clean_json(params))))

    # ---- experiments ----
    def status(self, experiment_id) -> str | None:
        r = self.conn.execute("SELECT status FROM tournament.experiments WHERE experiment_id = %s",
                              (experiment_id,)).fetchone()
        return r[0] if r else None

    def start_experiment(self, row: dict) -> None:
        self.conn.execute("DELETE FROM tournament.experiments WHERE experiment_id = %s", (row["experiment_id"],))
        for t in ("predictions", "signals", "trades", "performance_metrics", "feature_importance",
                  "ensemble_members", "experiment_artifacts"):
            self.conn.execute(f"DELETE FROM tournament.{t} WHERE experiment_id = %s", (row["experiment_id"],))
        cols = list(row)
        vals = [(_j(clean_json(v)) if isinstance(v, (dict, list)) else v) for v in row.values()]
        self.conn.execute(f"INSERT INTO tournament.experiments ({', '.join(cols)}) "
                          f"VALUES ({', '.join(['%s'] * len(cols))})", vals)
        self.conn.commit()

    def finish_experiment(self, experiment_id, status, reason="", oos_start=None, oos_end=None):
        self.conn.execute("""UPDATE tournament.experiments SET status = %s, status_reason = %s, finished_at = now(),
                             oos_start = COALESCE(%s, oos_start), oos_end = COALESCE(%s, oos_end)
                             WHERE experiment_id = %s""", (status, reason[:2000], oos_start, oos_end, experiment_id))
        self.conn.commit()

    def training_runs(self, rows: list[tuple]):
        self._copy("training_runs", ["experiment_id", "fold", "config_idx", "train_start", "train_end", "n_train",
                                     "fit_seconds", "predict_seconds", "cpu_percent", "rss_mb", "gpu_mb",
                                     "model_bytes"], rows)

    def validation_runs(self, rows: list[tuple]):
        self._copy("validation_runs", ["experiment_id", "fold", "config_idx", "threshold_idx", "val_start", "val_end",
                                       "val_net_bps", "val_trades", "chosen"], rows)

    def walk_forward_runs(self, rows: list[tuple]):
        self._copy("walk_forward_runs", ["experiment_id", "fold", "test_start", "test_end", "config_idx", "threshold",
                                         "abstained", "trades", "gross_bps", "net_bps", "forced_trades",
                                         "forced_gross_bps", "forced_net_bps"], rows)

    def oos_run(self, experiment_id, variant, start, end, summary, passes=None, failed=None):
        self.conn.execute("""INSERT INTO tournament.oos_runs VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (experiment_id, variant) DO UPDATE SET summary = EXCLUDED.summary, passes = EXCLUDED.passes,
            failed = EXCLUDED.failed, oos_start = EXCLUDED.oos_start, oos_end = EXCLUDED.oos_end""",
                          (experiment_id, variant, start, end, _j(clean_json(summary)), passes, failed))

    def predictions(self, experiment_id, fold, index, score, proba, h):
        p = np.asarray(proba, float) if proba is not None else np.full((len(index), 3), np.nan)
        rows = [(experiment_id, fold, t, _f(s), _f(a), _f(b), _f(c), h)
                for t, s, (a, b, c) in zip(index, score, p)]
        self._copy("predictions", ["experiment_id", "fold", "bar_ts", "score", "p_short", "p_flat", "p_long",
                                   "target_h"], rows)

    def signals(self, experiment_id, variant, index, pos):
        pos = np.asarray(pos)
        prev = np.concatenate([[np.nan], pos[:-1]])
        ch = pos != prev
        self._copy("signals", ["experiment_id", "variant", "bar_ts", "position"],
                   [(experiment_id, variant, t, int(p)) for t, p in zip(index[ch], pos[ch])])

    def trades(self, experiment_id, variant, trades: pd.DataFrame):
        if trades.empty:
            return
        dims = [d for d in ("trend", "vol", "volume", "character") if d in trades]
        rows = []
        for i, r in enumerate(trades.itertuples(index=False)):
            reg = {d: getattr(r, d) for d in dims} if dims else None
            rows.append((experiment_id, variant, i, r.entry_ts, r.exit_ts, int(r.side), int(r.bars), float(r.gross_bps),
                         float(r.fee_bps), float(r.slip_bps), float(r.funding_bps), float(r.net_bps),
                         _j(reg) if reg else None, int(getattr(r, "fold", -1))))
        self._copy("trades", ["experiment_id", "variant", "trade_no", "entry_ts", "exit_ts", "side", "bars",
                              "gross_bps", "fee_bps", "slip_bps", "funding_bps", "net_bps", "regime", "fold"], rows)

    def metrics(self, experiment_id, variant, scope, table: pd.DataFrame | dict):
        rows = []
        if isinstance(table, dict):
            for k, v in table.items():
                if isinstance(v, (int, float, np.floating, np.integer, bool, np.bool_)) or v is None:
                    rows.append((experiment_id, variant, scope, "all", k, _f(v)))
        else:
            for bucket, r in table.iterrows():
                for k, v in r.items():
                    rows.append((experiment_id, variant, scope, str(bucket), k, _f(v)))
        self.conn.execute("DELETE FROM tournament.performance_metrics WHERE experiment_id = %s AND variant = %s "
                          "AND scope = %s", (experiment_id, variant, scope))
        self._copy("performance_metrics", ["experiment_id", "variant", "scope", "bucket", "metric", "value"], rows)

    def importance(self, experiment_id, fold, imp: dict, kind):
        self._copy("feature_importance", ["experiment_id", "fold", "feature", "importance", "kind"],
                   [(experiment_id, fold, k, float(v), kind) for k, v in imp.items() if np.isfinite(v)])

    def artifact(self, experiment_id, kind, path):
        p = Path(path)
        self.conn.execute("INSERT INTO tournament.experiment_artifacts (experiment_id, kind, path, bytes) "
                          "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                          (experiment_id, kind, str(p), p.stat().st_size if p.exists() else None))

    def members(self, experiment_id, rows: list[tuple]):
        self._copy("ensemble_members", ["experiment_id", "member_experiment_id", "role", "weight"],
                   [(experiment_id, m, role, _f(w)) for m, role, w in rows])

    def registry(self, experiment_id, status, reason=""):
        self.conn.execute("INSERT INTO tournament.registry (experiment_id, status, reason) VALUES (%s,%s,%s)",
                          (experiment_id, status, reason))

    def commit(self):
        self.conn.commit()

    def _copy(self, table, cols, rows):
        if not rows:
            return
        with self.conn.cursor().copy(f"COPY tournament.{table} ({', '.join(cols)}) FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)

    # ---- readers ----
    def df(self, sql, params=None) -> pd.DataFrame:
        cur = self.conn.execute(sql, params)
        cols = [d.name for d in cur.description]
        out = pd.DataFrame(cur.fetchall(), columns=cols)
        for c in out.columns:
            if isinstance(out[c].dtype, pd.DatetimeTZDtype):
                out[c] = out[c].dt.tz_convert("UTC")
        return out

    def oos_predictions(self, experiment_id) -> pd.DataFrame:
        d = self.df("SELECT fold, bar_ts, score, p_short, p_flat, p_long, target_h FROM tournament.predictions "
                    "WHERE experiment_id = %s ORDER BY bar_ts", (experiment_id,))
        return d.set_index("bar_ts")

    def experiment_trades(self, experiment_id, variant="policy") -> pd.DataFrame:
        d = self.df("SELECT * FROM tournament.trades WHERE experiment_id = %s AND variant = %s ORDER BY trade_no",
                    (experiment_id, variant))
        if d.empty:
            return d
        return d.set_index(pd.DatetimeIndex(d["exit_ts"]))
