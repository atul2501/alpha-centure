"""Strategy trace: a strategy-agnostic record of HOW a decision was built, for the dashboard's Strategy View.

    signal layers  per-coin scores the strategy reads (forecast, momentum, carry...)
    book layers    per-coin weights of each sub-strategy, with its share of the mix (blend_weight)
    blend layer    sum of blend_weight x book
    final layer    what is traded after vol targeting and caps (also in `final`)

Any strategy that fills a StrategyTrace is drawn by the same page, so swapping V4_carry for another strategy needs no
dashboard change. Stored as JSON in paper_decisions.trace.

    python -m alpha.strategy.trace --snapshot     record the current strategy breakdown now (action SNAPSHOT)
"""

from dataclasses import asdict, dataclass, field

SIGNAL, BOOK, BLEND, FINAL = "signal", "book", "blend", "final"


@dataclass
class Layer:
    key: str
    title: str
    kind: str                                   # signal | book | blend | final
    description: str = ""
    values: dict[str, float] = field(default_factory=dict)
    blend_weight: float | None = None           # books only: share of the mix
    features: dict[str, dict[str, float]] | None = None   # symbol -> feature -> contribution to the forecast
    inputs: list[str] | None = None             # keys of the layers that feed this one (None = the previous step)


@dataclass
class StrategyTrace:
    strategy: str
    bar_time: str
    schedule: str = ""
    params: dict = field(default_factory=dict)
    layers: list[Layer] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "StrategyTrace":
        d = dict(d)
        d["layers"] = [Layer(**x) for x in d.get("layers", [])]
        return StrategyTrace(**d)

    def of(self, kind: str) -> list[Layer]:
        return [x for x in self.layers if x.kind == kind]

    def feeds(self, x: Layer) -> list[str]:
        """Keys of the layers flowing into x: declared inputs, else every layer of the previous step present."""
        if x.inputs is not None:
            return [k for k in x.inputs if any(y.key == k for y in self.layers)]
        order = [SIGNAL, BOOK, BLEND, FINAL]
        for prev in reversed(order[:order.index(x.kind)]):
            if self.of(prev):
                return [y.key for y in self.of(prev)]
        return []

    def symbols(self) -> list[str]:
        """Every coin in the trace, largest final position first."""
        seen = dict.fromkeys(self.final)
        for x in self.layers:
            seen.update(dict.fromkeys(x.values))
        return sorted(seen, key=lambda s: -abs(self.final.get(s, 0.0)))


def gross(values: dict[str, float]) -> float:
    return sum(abs(v) for v in values.values())


def net(values: dict[str, float]) -> float:
    return sum(values.values())


def contributions(t: StrategyTrace) -> dict[str, dict[str, float]]:
    """book key -> symbol -> blend_weight x book weight (these add up to the blend layer)."""
    return {b.key: {s: (b.blend_weight if b.blend_weight is not None else 1.0) * v for s, v in b.values.items()}
            for b in t.of(BOOK)}


def scale_factor(t: StrategyTrace) -> float | None:
    """Final gross / blend gross: how much vol targeting (and caps) scaled the mix."""
    blend = t.of(BLEND)
    g = gross(blend[0].values) if blend else 0.0
    return gross(t.final) / g if g > 0 else None


def snapshot() -> int:
    """Compute the current V4_carry breakdown from the database and store it as a SNAPSHOT decision row."""
    import pandas as pd
    import psycopg
    from psycopg.types.json import Jsonb

    from alpha.config import get_settings
    from alpha.strategy import p6

    s = get_settings()
    b = p6.RidgeBundle.latest(s.models_dir)
    if b is None:
        raise SystemExit(f"no model in {s.models_dir}")
    now = pd.Timestamp.now(tz="UTC")
    with psycopg.connect(s.database_url) as conn:
        last, w, _, tr = p6.live_targets_explained(conn, s.symbols, b, now)
        tr.notes = "snapshot: what the strategy would hold if it rebalanced on this bar (nothing traded)"
        eq = conn.execute("SELECT equity FROM paper_equity ORDER BY ts DESC LIMIT 1").fetchone()
        did = conn.execute(
            "INSERT INTO paper_decisions (bar_time, model, action, reason, equity, targets, current, trace) "
            "VALUES (%s, %s, 'SNAPSHOT', %s, %s, %s, NULL, %s) RETURNING id",
            (last.to_pydatetime(), p6.NAME, "manual breakdown snapshot (not traded)", float(eq[0]) if eq else 0.0,
             Jsonb({k: round(float(v), 5) for k, v in w.items() if abs(v) > 1e-6}),
             Jsonb(tr.to_json()))).fetchone()[0]
    print(f"snapshot stored: decision id {did}, bar {last}")
    return did


if __name__ == "__main__":
    import sys
    if "--snapshot" in sys.argv:
        snapshot()
    else:
        print(__doc__)
