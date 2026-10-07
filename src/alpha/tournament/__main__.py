"""SOL 15m model tournament CLI.

    uv run python -m alpha.tournament data                       # build / register the DEV dataset + coverage
    uv run python -m alpha.tournament run configs/tournament/stage_2.yaml [--workers 3] [--force]
    uv run python -m alpha.tournament reproduce <experiment_id>  # re-run and compare with stored metrics
    uv run python -m alpha.tournament analyze                    # significance, robustness, gates, registry
    uv run python -m alpha.tournament report                     # reports/MODEL_TOURNAMENT_REPORT.md
"""

import argparse
import json
import sys
import warnings

import psycopg
from loguru import logger

from alpha.config import get_settings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    ap = argparse.ArgumentParser(prog="alpha.tournament")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--force", action="store_true")
    r.add_argument("--only", default=None, help="comma-separated entry names")
    rp = sub.add_parser("reproduce")
    rp.add_argument("experiment_id")
    an = sub.add_parser("analyze")
    an.add_argument("--top", type=int, default=10)
    sub.add_parser("report")
    cp = sub.add_parser("compare")
    cp.add_argument("--coins", default=None, help="comma-separated subset (default: the 23-coin universe)")
    cp.add_argument("--models-only", action="store_true")
    sh = sub.add_parser("shadow")
    sh.add_argument("--once", action="store_true")
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        from alpha.tournament.database.repo import Repo

        Repo.init_schema(conn)
        if a.cmd == "data":
            from alpha.tournament.data import dataset as dsm
            from alpha.tournament.features.groups import RAW_NEEDS

            ds = dsm.build(conn, rebuild=True)
            cov = dsm.register(conn, ds, RAW_NEEDS)
            print(ds.dataset_id, len(ds.frame), ds.frame.index.min(), "->", ds.frame.index.max())
            print(json.dumps(cov, indent=1))
        elif a.cmd == "run":
            from alpha.tournament.experiments.matrix import run_config

            run_config(conn, a.config, workers=a.workers, force=a.force,
                       only=a.only.split(",") if a.only else None)
        elif a.cmd == "reproduce":
            from alpha.tournament.experiments.matrix import reproduce

            ok = reproduce(conn, a.experiment_id)
            sys.exit(0 if ok else 1)
        elif a.cmd == "analyze":
            from alpha.tournament.analytics.selection import analyze

            analyze(conn, top=a.top)
        elif a.cmd == "report":
            from alpha.tournament.reports.report import write_report

            print(write_report(conn))
        elif a.cmd == "compare":
            from alpha.tournament.experiments import compare

            if a.coins or a.models_only:
                compare.run_models(conn, a.coins.split(",") if a.coins else get_settings().symbols)
            else:
                res = compare.main(conn)
                for k, v in res.items():
                    print(f"{k:34s} net ${v['net_usd']:>9,.0f}  sharpe {v['sharpe']:.2f}  maxDD {100 * v['max_dd']:.1f}%")
        elif a.cmd == "shadow":
            from alpha.tournament.shadow.runner import run as shadow_run

            shadow_run(conn, once=a.once)


if __name__ == "__main__":
    main()
