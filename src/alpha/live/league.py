"""Shadow league: P6 and its pre-registered challengers, scored every day on live data. Never trades.

    uv run python -m alpha.live.league                                # store daily rows since LEAGUE_START
    uv run python -m alpha.live.league --dry-run --since 2026-10-01   # print only (days before LEAGUE_START = info)
    uv run python -m alpha.live.league --report                       # score table + switch-rule status

Why: every held-out set (VALID-A, VALID-B, unseen-9, fresh-14) is used, so past data can no longer show that a
variant beats P6. Only data after LEAGUE_START can. Every candidate runs the live P6 pipeline in the research
simulator (report.shadow_from_weights): the engine's monthly ridge bundles, 72h rebalance, 20% vol target, 3x gross
and 0.5x per-coin caps, 1% band, 60% maker / 40% taker costs (one frozen cost table), funding.

PRE-REGISTERED (2026-10-08, before any league data existed; changing anything below = a new pre-registration):
    candidates  P6              the live book (reference)
                R1_ensemble     0.5 P6 + 0.5 momentum rules (P3: xs + ts momentum, vol-targeted)     round 2
                R5_cost_band    P6, but a coin changes only if |expected 72h return| > 1.5x round trip round 2
                R6_carry_sleeve 0.75 P6 + 0.25 cross-sectional funding carry (vol-targeted)          round 2
                N1_R1_revol     R1 re-scaled to the full 20% vol target (R1 runs below it)           new
                BENCH_BTC       1x long BTC: context only, never a switch candidate
    rule        reviews at 90 and 180 days after LEAGUE_START. A challenger replaces P6 only if ALL hold:
                  1. its net return > P6's net return
                  2. Newey-West t-stat of the daily (challenger - P6) difference >= 2.5 (4 challengers)
                  3. its max drawdown <= P6's max drawdown + 5 percentage points
                otherwise P6 stays. The league reports only; a switch is a manual change after a review.
"""

import argparse
import resource
import sys
import time

import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.db import SQL_DIR
from alpha.live.report import p6_scores, shadow_from_weights, stats
from alpha.research import signals as sig
from alpha.research.models import feature_frame
from alpha.research.panel import CACHE, build_panel, wide
from alpha.research.phase4 import momentum_scores, vol_target
from alpha.research.portfolio_sim import newey_west_t, to_weights
from alpha.research.screen import symbol_costs
from alpha.research.splits import DEV
from alpha.strategy import p6

LEAGUE_START = pd.Timestamp("2026-10-09", tz="UTC")
REVIEW_DAYS = (90, 180)
REFERENCE = "P6"
BENCHMARK = "BENCH_BTC"
T_MIN = 2.5
DD_SLACK = 0.05
NW_LAGS = 5  # 72h holdings: daily P&L overlaps ~3 days
COSTS_FILE = CACHE / "league_costs.csv"
SQL = SQL_DIR / "008_shadow.sql"


# ---------------------------------------------------------------------------------------------------------------
# costs: one frozen table for the whole league

def league_costs(conn, symbols: list[str]) -> pd.DataFrame:
    """symbol_costs on the DEV panel, computed once and frozen. Coins with no live spread get the widest live
    spread, and coins missing entirely get the most expensive coin's costs (never a silent zero cost)."""
    if COSTS_FILE.exists():
        c = pd.read_csv(COSTS_FILE, index_col=0)
    else:
        c = symbol_costs(conn, build_panel(conn, symbols, start=DEV.start, end=DEV.end))
        live = {r[0] for r in conn.execute("SELECT DISTINCT split_part(symbol, '.', 1) FROM book_tick").fetchall()}
        missing = [s for s in c.index if s not in live]
        if missing and len(missing) < len(c):
            worst = c.loc[[s for s in c.index if s in live], "spread_bps"].max()
            c.loc[missing, "taker_bps"] += 0.5 * (worst - c.loc[missing, "spread_bps"])
            c.loc[missing, "spread_bps"] = worst
        COSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        c.to_csv(COSTS_FILE)
    return c.reindex(symbols).fillna(c.max())


# ---------------------------------------------------------------------------------------------------------------
# candidate books (pre-band weights; the band / cost / funding step is shared)

def _frames(panel):
    ret = wide(panel, "ret")
    return ret, ret.rolling(sig.VOL_WINDOW, min_periods=48).std(), wide(panel, "eligible").fillna(False).astype(bool)


def rules_weights(panel) -> pd.DataFrame:
    """P3's book: xs + ts momentum rules, 50/50, vol-targeted."""
    ret, vol, el = _frames(panel)
    xs, ts = momentum_scores(panel, 1.0)
    w = (to_weights(xs.reindex(columns=ret.columns), vol, el, "xs", every=p6.H) +
         to_weights(ts.reindex(columns=ret.columns), vol, el, "ts", every=p6.H)) / 2
    return vol_target(w, ret)


def carry_weights(panel, scores: dict) -> pd.DataFrame:
    """Cross-sectional carry (short high 24h-average funding, long low), vol-targeted."""
    ret, vol, el = _frames(panel)
    return vol_target(to_weights(scores["xscarry"].reindex(columns=ret.columns), vol, el, "xs", every=p6.H), ret)


def cost_band(score: pd.DataFrame, costs: pd.DataFrame, maker: float = p6.MAKER_SHARE_ASSUMED):
    """At a rebalance a coin changes only if |expected 72h return| (forecast x vol x sqrt(72)) > 1.5x its round
    trip cost; weaker forecasts keep the current position. Closing to zero is always allowed."""
    rt = 2 * (maker * costs["maker_bps"] + (1 - maker) * costs["taker_bps"]) / 1e4

    def band_fn(w, ret, vol, h):
        exp = (score.reindex_like(w).abs() * vol.reindex_like(w) * np.sqrt(h)).to_numpy()
        thr = 1.5 * rt.reindex(w.columns).fillna(rt.max()).to_numpy()
        hours = ((w.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).to_numpy()
        W, out, prev = w.to_numpy(), np.zeros_like(w.to_numpy()), np.zeros(w.shape[1])
        for i in range(len(W)):
            if hours[i] % h == 0:
                go = (np.nan_to_num(exp[i]) > thr) | (W[i] == 0)
                go &= np.abs(W[i] - prev) > 1e-3
                prev = np.where(go, W[i], prev)
            out[i] = prev
        return pd.DataFrame(out, index=w.index, columns=w.columns)
    return band_fn


def candidates(panel, scores: dict, sc: pd.DataFrame, costs: pd.DataFrame) -> dict:
    """name -> (pre-band weights, band_fn or None)."""
    ret = wide(panel, "ret")
    base = p6.target_weights(panel, sc)
    r1 = 0.5 * base + 0.5 * rules_weights(panel).reindex_like(base).fillna(0.0)
    btc = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    btc["BTCUSDT"] = 1.0
    return {
        "P6": (base, None),
        "R1_ensemble": (r1, None),
        "R5_cost_band": (base, cost_band(sc, costs)),
        "R6_carry_sleeve": (0.75 * base + 0.25 * carry_weights(panel, scores).reindex_like(base).fillna(0.0), None),
        "N1_R1_revol": (vol_target(r1, ret), None),
        BENCHMARK: (btc, None),
    }


def run(conn, symbols: list[str], models_dir: str, since: pd.Timestamp, end: pd.Timestamp,
        costs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily rows (day, strategy, gross, funding, cost, turnover, net) for every candidate over [since, end)."""
    panel = build_panel(conn, symbols, start=since - pd.Timedelta(days=p6.HISTORY_DAYS), end=end)
    scores = sig.compute(panel)
    X, _ = feature_frame(panel, scores)
    sc = p6_scores(X, end, models_dir)
    if sc is None:
        raise SystemExit(f"no p6_ridge_*.joblib in {models_dir}")
    costs = league_costs(conn, symbols) if costs is None else costs
    out = []
    for name, (w, band_fn) in candidates(panel, scores, sc, costs).items():
        d = shadow_from_weights(panel, w, costs, since, end, band_fn)
        out.append(d.assign(strategy=name).rename_axis("day").reset_index())
    df = pd.concat(out, ignore_index=True)
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df[["day", "strategy", "gross", "funding", "cost", "turnover", "net"]]


def store(conn, df: pd.DataFrame) -> None:
    conn.execute(SQL.read_text())
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO shadow_daily (day, strategy, gross, funding, cost, turnover, net)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (day, strategy) DO UPDATE SET gross = EXCLUDED.gross,
                             funding = EXCLUDED.funding, cost = EXCLUDED.cost, turnover = EXCLUDED.turnover,
                             net = EXCLUDED.net, computed_at = now()""",
                        [tuple(r) for r in df.itertuples(index=False)])
    conn.commit()


# ---------------------------------------------------------------------------------------------------------------
# scoring

def max_dd(daily: pd.Series) -> float:
    eq = daily.cumsum()
    return float((eq.cummax() - eq).max()) if len(eq) else 0.0


def rule(daily: pd.DataFrame) -> pd.DataFrame:
    """daily: day x strategy net returns (fraction of starting equity). One row per challenger with the three
    pre-registered conditions and whether all hold."""
    ref = daily[REFERENCE]
    rows = {}
    for name in daily.columns:
        if name in (REFERENCE, BENCHMARK):
            continue
        diff = (daily[name] - ref).dropna()
        t = newey_west_t(diff, NW_LAGS)
        c1 = daily[name].sum() > ref.sum()
        c2 = bool(np.isfinite(t) and t >= T_MIN)
        c3 = max_dd(daily[name]) <= max_dd(ref) + DD_SLACK
        rows[name] = {"net_minus_p6": daily[name].sum() - ref.sum(), "t_vs_p6": t,
                      "beats_net": c1, "t_ok": c2, "dd_ok": c3, "switch": c1 and c2 and c3}
    return pd.DataFrame(rows).T


def report(conn) -> None:
    rows = conn.execute("SELECT day, strategy, net FROM shadow_daily WHERE day >= %s ORDER BY day",
                        (LEAGUE_START.date(),)).fetchall()
    if not rows:
        print("no league data yet (first rows appear the day after LEAGUE_START)")
        return
    daily = pd.DataFrame(rows, columns=["day", "strategy", "net"]).pivot(index="day", columns="strategy",
                                                                           values="net").fillna(0.0)
    days = len(daily)
    table = pd.DataFrame({n: stats(daily[n]) for n in daily.columns}).T[["days", "net", "sharpe", "sharpe_se",
                                                                         "max_dd"]]
    pd.set_option("display.width", 160)
    print(f"Shadow league since {LEAGUE_START.date()}: {days} days (reviews at {REVIEW_DAYS} days)\n")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nswitch rule (only binding at a review):")
    print(rule(daily).to_string(float_format=lambda v: f"{v:.3f}"))
    due = [r for r in REVIEW_DAYS if days >= r]
    print(f"\nreview due: {due[-1]}-day review" if due else f"\nnext review in {REVIEW_DAYS[0] - days} days")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print, store nothing")
    ap.add_argument("--since", help="first day (default LEAGUE_START); only LEAGUE_START onward counts")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        if a.report:
            report(conn)
            return
        since = pd.Timestamp(a.since, tz="UTC") if a.since else LEAGUE_START
        if since < LEAGUE_START and not a.dry_run:
            sys.exit("days before LEAGUE_START are information only: use --dry-run")
        end = pd.Timestamp.now(tz="UTC").floor("D")  # complete days only
        if end <= since:
            logger.info("league starts {}: nothing to compute yet", since.date())
            return
        t0 = time.time()
        df = run(conn, s.symbols, s.models_dir, since, end)
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
        logger.info("{} rows, {} days, {:.0f}s, peak {:.0f} MB", len(df), df["day"].nunique(), time.time() - t0, peak_mb)
        if a.dry_run:
            piv = df.pivot(index="day", columns="strategy", values="net")
            print(piv.to_string(float_format=lambda v: f"{v:+.4f}"))
            print("\ntotal:", piv.sum().to_string(float_format=lambda v: f"{v:+.4f}"))
        else:
            store(conn, df)


if __name__ == "__main__":
    main()
