"""Experiment harness: compare walk-forward configs on the same folds, never touching the frozen holdout.

    uv run python -m alpha.research.experiment --configs baseline,exits_learn --ref baseline
    uv run python -m alpha.research.experiment --final <config>      # ONE run on the holdout, at the very end

Acceptance rule (vs --ref): total R higher AND positive in >= 60% of years AND max DD not worse by > 5pp.
Every run is appended to data/experiments/log.jsonl so we always know how many configs were tried.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import psycopg
from loguru import logger

from alpha.backtest.walkforward import WFConfig, report, run, summary
from alpha.config import get_settings
from alpha.regime.hmm import load_regime_frames
from alpha.research.dataset import cached_dataset
from alpha.strategies.labels import EXIT_MENU, Costs, Entry

HOLDOUT_START = pd.Timestamp("2025-10-01", tz="UTC")
OUT = Path("data/experiments")
TFS = ("15m", "1h")
MAKER_COSTS = Costs(fee_entry=0.0002, fee_target=0.0002)
MAKER_ENTRY = Entry("limit", offset_atr=0.1, fill_bars=2)

# v2: perp-only context/benchmark/regime (v1 caches used spot context and must not be mixed in)
DATASETS = {
    "market": dict(tag="mkt_15m1h_v2", costs=Costs(), entry=Entry()),
    "maker": dict(tag="maker_15m1h_v2", costs=MAKER_COSTS, entry=MAKER_ENTRY),
}

# ---- configs: each phase builds on the previous phase's winner (edit as results come in) ----
B = WFConfig()
CONFIGS: dict[str, tuple[str, WFConfig]] = {
    "baseline": ("market", B),
    # Phase A: exits
    "exits_breakeven": ("market", replace(B, name="exits_breakeven", exits="breakeven")),
    "exits_trail": ("market", replace(B, name="exits_trail", exits="trail")),
    "exits_partial": ("market", replace(B, name="exits_partial", exits="partial_trail")),
    "exits_learn": ("market", replace(B, name="exits_learn", exits="learn")),
}


def add_phase(prefix: str, base: str, variants: dict[str, dict], dataset: str | None = None) -> None:
    """Register configs that extend an existing one, e.g. add_phase('B', 'exits_learn', {...})."""
    ds, cfg = CONFIGS[base]
    for name, kw in variants.items():
        CONFIGS[name] = (dataset or ds, replace(cfg, name=name, **kw))


# Phase A result (pre-holdout): breakeven accepted (+14R, DD 26%); trail/learn far higher R (+102/+128R) but DD
# 47-53%. Phase B is tested on both; risk controls (Phase E) are tried early on learn because DD is the constraint.
GATES = dict(ev_mode="setup", calibrate=True, max_cost_grid=(0.1, 0.15, 0.2, None), ev_floor=0.05, inner_splits=3)
RISK = dict(sizing="ev", max_same_side=2, daily_loss_r=3.0)
add_phase("B", "exits_breakeven", {"be_gates": GATES})
add_phase("B", "exits_learn", {
    "learn_ev_setup": dict(ev_mode="setup"),
    "learn_gates": GATES,
})
add_phase("E", "exits_learn", {"learn_risk": RISK, "learn_gates_risk": {**GATES, **RISK}})

# Phase B/E result: be_gates wins (+37.5R, avg +0.112R, DD 16%). Trailing exits did not survive careful gating
# (learn_gates +3R / DD 68%) and risk controls barely cut their DD (learn_risk DD 48%). C and D build on be_gates.
add_phase("C", "be_gates", {
    "be_gates_hl365": dict(half_life_days=365),
    "be_gates_hl730": dict(half_life_days=730),
    "be_gates_roll730": dict(rolling_days=730),
})
add_phase("D", "be_gates", {"be_gates_maker": {}}, dataset="maker")
add_phase("E", "be_gates", {"be_gates_risk": RISK})

# Phase C/D/E result: be_gates_roll730 accepted (+47.1R, avg +0.144R, DD 10.3%). Risk controls halved DD on
# be_gates (16.1% -> 7.7%) at ~-1R, so the final candidate adds them to the 2-year rolling window.
add_phase("E", "be_gates_roll730", {"roll730_risk": RISK})


def load(dataset: str):
    s = get_settings()
    d = DATASETS[dataset]
    with psycopg.connect(s.database_url) as conn:
        ds = cached_dataset(conn, s, d["tag"], tfs=TFS, costs=d["costs"], entry=d["entry"], exits=dict(EXIT_MENU))
        frames = load_regime_frames(conn, s.symbols)
    return ds, frames


def evaluate(name: str, final: bool = False) -> dict:
    dataset, cfg = CONFIGS[name]
    ds, frames = load(dataset)
    t0 = time.time()
    kw = dict(test_from=HOLDOUT_START) if final else dict(test_until=HOLDOUT_START)
    rows, folds, _ = run(ds, frames, cfg, variants=("regime_meta",), fold_months=3, **kw)
    res = {"name": name, "dataset": dataset, "final_holdout": final, "config": asdict(cfg),
           "seconds": round(time.time() - t0), **summary(rows, folds)}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{name}{'__HOLDOUT' if final else ''}"
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    rows.to_parquet(OUT / f"{tag}_rows.parquet")
    with open(OUT / "log.jsonl", "a") as f:
        f.write(json.dumps({k: res[k] for k in ("name", "final_holdout", "trades", "total_r", "avg_r", "max_dd",
                                                 "years_positive")}, default=str) + "\n")
    if final:
        print(report(rows, folds))
    return res


def accept(res: dict, ref: dict) -> tuple[bool, str]:
    checks = {
        "more total R": res["total_r"] > ref["total_r"],
        ">=60% years positive": res["years_positive"] >= 0.6,
        "DD not >5pp worse": res["max_dd"] <= ref["max_dd"] + 0.05,
    }
    return all(checks.values()), ", ".join(f"{'✓' if v else '✗'} {k}" for k, v in checks.items())


def table(results: list[dict], ref: dict | None) -> str:
    rows = []
    for r in results:
        ok, why = accept(r, ref) if ref and r["name"] != ref["name"] else (None, "reference")
        rows.append({"config": r["name"], "trades": r["trades"], "avg_r": r["avg_r"], "total_r": r["total_r"],
                     "max_dd": r["max_dd"], "yrs+": r["years_positive"],
                     "years": " ".join(f"{int(y) % 100}:{v:+.0f}" for y, v in r["years"].items()),
                     "accept": {True: "YES", False: "no", None: "-"}[ok], "why": why})
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="baseline")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--final", default=None, help="config to run ONCE on the frozen holdout")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}",
               filter=lambda r: "USDT.P" not in r["message"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    if a.list:
        print("\n".join(f"{k:22s} [{v[0]}] {v[1]}" for k, v in CONFIGS.items()))
        return
    if a.final:
        res = evaluate(a.final, final=True)
        print(json.dumps({k: res[k] for k in ("trades", "avg_r", "total_r", "max_dd", "years", "calibration")},
                         indent=2, default=str))
        return
    names = a.configs.split(",")
    results = {}
    for n in dict.fromkeys([a.ref, *names]):
        cached = OUT / f"{n}.json"
        results[n] = json.loads(cached.read_text()) if cached.exists() and n not in names else evaluate(n)
        logger.info("{}: {} trades, total {:+.1f}R, avg {:+.3f}R, DD {:.1%}", n, results[n]["trades"],
                    results[n]["total_r"], results[n]["avg_r"] or 0, results[n]["max_dd"])
    print(table(list(results.values()), results.get(a.ref)))
    for n in names:
        if results[n].get("calibration"):
            print(f"\n{n} calibration (EV quintiles of regime-passed setups): predicted vs realized R")
            print(pd.DataFrame(results[n]["calibration"]).to_string(index=False))


if __name__ == "__main__":
    main()
