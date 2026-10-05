"""A trained model version = regime HMM + meta model + policy, saved together and tracked in model_registry."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from alpha.decision import Policy
from alpha.models.meta import MetaModel
from alpha.regime.hmm import RegimeModel

DRIFT_FEATURES = ["atr_pct", "vol_z", "rvol", "bb_width", "funding_z", "rsi"]


@dataclass
class Bundle:
    version: str
    regime: RegimeModel
    meta: MetaModel
    policy: Policy
    train_end: datetime
    metrics: dict = field(default_factory=dict)

    @staticmethod
    def new_version() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def save(self, models_dir: str | Path) -> Path:
        d = Path(models_dir) / self.version
        d.mkdir(parents=True, exist_ok=True)
        self.regime.save(d / "regime.joblib")
        self.meta.save(d / "meta.joblib")
        (d / "policy.json").write_text(json.dumps(self.policy.to_dict(), indent=2))
        (d / "metrics.json").write_text(json.dumps(json_safe(self.metrics), indent=2))
        (d / "train_end.txt").write_text(self.train_end.isoformat())
        return d

    @classmethod
    def load(cls, path: str | Path, version: str) -> "Bundle":
        d = Path(path)
        return cls(version, RegimeModel.load(d / "regime.joblib"), MetaModel.load(d / "meta.joblib"),
                   Policy.from_dict(json.loads((d / "policy.json").read_text())),
                   datetime.fromisoformat((d / "train_end.txt").read_text()),
                   json.loads((d / "metrics.json").read_text()))


def drift_reference(train: pd.DataFrame) -> dict:
    """Decile bin edges + proportions of key features at training time, for PSI drift checks later."""
    ref = {}
    for c in DRIFT_FEATURES:
        if c not in train:
            continue
        x = train[c].dropna().to_numpy(dtype=float)
        if len(x) < 100:
            continue
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        counts = np.histogram(np.clip(x, edges[0], edges[-1]), edges)[0]
        ref[c] = {"edges": edges.tolist(), "p": (counts / counts.sum()).tolist()}
    return ref


def psi(ref: dict, values: np.ndarray) -> float:
    """Population stability index of values vs the training reference. >0.25 is a large shift."""
    edges, p = np.array(ref["edges"]), np.array(ref["p"])
    q = np.histogram(np.clip(values, edges[0], edges[-1]), edges)[0].astype(float)
    if q.sum() == 0:
        return float("nan")
    q = np.clip(q / q.sum(), 1e-4, None)
    p = np.clip(p, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


# ---------- registry ----------

def json_safe(x):
    """NaN/inf -> None (Postgres jsonb rejects NaN), numpy scalars -> Python, recursively."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.isoformat()
    return x


def register(conn: psycopg.Connection, b: Bundle, status: str, path: str, note: str = "") -> None:
    conn.execute(
        """INSERT INTO model_registry (version, status, train_end, path, policy, metrics, note)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (version) DO UPDATE SET status = EXCLUDED.status, metrics = EXCLUDED.metrics, note = EXCLUDED.note""",
        (b.version, status, b.train_end, path, Jsonb(json_safe(b.policy.to_dict())), Jsonb(json_safe(b.metrics)), note))


def set_status(conn: psycopg.Connection, version: str, status: str, note: str | None = None) -> None:
    conn.execute("UPDATE model_registry SET status = %s, note = coalesce(%s, note) WHERE version = %s",
                 (status, note, version))


def champion_row(conn: psycopg.Connection) -> dict | None:
    row = conn.execute("""SELECT version, path, status FROM model_registry
                          WHERE status IN ('champion', 'degraded') ORDER BY created_at DESC LIMIT 1""").fetchone()
    return {"version": row[0], "path": row[1], "status": row[2]} if row else None


def load_champion(conn: psycopg.Connection) -> tuple[Bundle, str] | None:
    row = champion_row(conn)
    if not row:
        return None
    return Bundle.load(row["path"], row["version"]), row["status"]
