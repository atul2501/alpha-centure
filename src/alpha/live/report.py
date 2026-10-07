"""Phase 6 report: mainnet paper vs shadow backtest (research code on the same days) vs validation expectations.

    uv run python -m alpha.live.report

Engineering gate for the 15-30 day paper run (pre-registered, all must pass):
    uptime            equity marks for >= 99% of minutes since the paper start
    schedule          every rebalance bar has a REBALANCE decision (or a NO_TRADE with a stated reason)
    ledger            account rebuilt from fills + funding == last recorded equity (within $0.01 at the same marks)
    costs             realized execution cost per unit traded <= 1.25 x the backtest cost model
    maker share       maker-filled share of traded notional within 0.48-0.72 (model assumes 0.60)
    book depth        no fill had to go beyond the visible top-20 book
    tracking          paper vs shadow daily net: correlation >= 0.8 once >= 10 days exist
Profitability is reported with its uncertainty but is NOT decided by a 15-30 day run (too few rebalances).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from alpha.config import get_settings
from alpha.exec.account import Account
from alpha.research import signals as sig
from alpha.research.models import feature_frame
from alpha.research.panel import build_panel, cached_panel, wide
from alpha.research.phase4 import apply_band_and_stop
from alpha.research.portfolio_sim import simulate
from alpha.research.screen import symbol_costs
from alpha.strategy import p6

VALID_A_EXPECTED = {"sharpe": 1.52, "daily_mean": 0.445 / 457}  # P6 on VALID-A (pre-registered finalist)
VALID_B_OBSERVED = {"sharpe": -0.15, "daily_mean": -0.034 / 370}


def q(conn, sql, params=None) -> pd.DataFrame:
    cur = conn.execute(sql, params or ())
    cols = [d.name for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def paper_frames(conn):
    eq = q(conn, "SELECT ts, equity, fees, funding, gross_lev, positions FROM paper_equity ORDER BY ts")
    fills = q(conn, "SELECT f.*, o.kind FROM paper_fills f JOIN paper_orders o ON o.id = f.order_id ORDER BY f.id")
    dec = q(conn, "SELECT * FROM paper_decisions ORDER BY id")
    fund = q(conn, "SELECT * FROM paper_funding ORDER BY id")
    st = conn.execute("SELECT value FROM paper_state WHERE key = 'account'").fetchone()
    mon = conn.execute("SELECT value FROM paper_state WHERE key = 'monitor'").fetchone()
    print("strategy monitor:", (mon[0] or {}).get("reason", "not checked yet") if mon else "not checked yet")
    return eq, fills, dec, fund, (st[0] if st else None)


def stats(daily: pd.Series) -> dict:
    d = daily.dropna()
    if d.empty:
        return {"days": 0}
    eq = d.cumsum()
    wins, losses = d[d > 0].sum(), -d[d < 0].sum()
    sd, down = d.std(), d[d < 0].std()
    return {"days": len(d), "net": float(d.sum()), "mean_daily": float(d.mean()),
            "pf": float(wins / losses) if losses > 0 else np.inf, "hit_days": float((d > 0).mean()),
            "max_dd": float((eq.cummax() - eq).max()),
            "sharpe": float(d.mean() / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
            "sortino": float(d.mean() / down * np.sqrt(365)) if down and down > 0 else np.nan,
            "sharpe_se": float(np.sqrt(365 / len(d)))}  # ~ standard error of an annualized Sharpe estimate


def per_symbol_paper(eq: pd.DataFrame) -> pd.DataFrame:
    """Gross price P&L per symbol and by side from consecutive minute snapshots: qty_{t-1} * (mark_t - mark_{t-1})."""
    rows = []
    prev = {}
    for _, r in eq.iterrows():
        pos = r["positions"] or {}
        for s, p in pos.items():
            if s in prev and p.get("mark") is not None and prev[s]["mark"] is not None:
                pnl = prev[s]["qty"] * (p["mark"] - prev[s]["mark"])
                rows.append((s, "long" if prev[s]["qty"] > 0 else "short", pnl))
        prev = pos
    df = pd.DataFrame(rows, columns=["symbol", "side", "pnl"])
    return df


def p6_scores(X: pd.DataFrame, end: pd.Timestamp, models_dir: str) -> pd.DataFrame | None:
    """Ridge scores as the live engine had them: each bundle from its train_end until the next one. The first
    bundle also scores the earlier rows (warm-up for the vol target only; score the window after its train_end)."""
    bundles = sorted(Path(models_dir).glob("p6_ridge_*.joblib"))
    if not bundles:
        return None
    import joblib

    bs = [joblib.load(b) for b in bundles]
    times = X.index.get_level_values("time")
    parts = []
    for i, b in enumerate(bs):
        nxt = bs[i + 1].train_end if i + 1 < len(bs) else end
        m = (times >= (b.train_end if i else times.min())) & (times < nxt)
        if m.any():
            parts.append(p6.score(b, X[m]))
    return pd.concat(parts).sort_index()


def shadow_from_weights(panel: pd.DataFrame, w: pd.DataFrame, costs: pd.DataFrame, start: pd.Timestamp,
                        end: pd.Timestamp, band_fn=None) -> pd.DataFrame:
    """Pre-band weights -> 1% band (or band_fn(w, ret, vol, h)) -> P6 cost model + funding -> daily
    gross / funding / cost / turnover / net over [start, end)."""
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    w = band_fn(w, ret, vol, p6.H) if band_fn else apply_band_and_stop(w, ret, vol, p6.BAND, None, p6.H)
    t = w.index[(w.index >= start) & (w.index < end)]
    per_side = p6.MAKER_SHARE_ASSUMED * costs["maker_bps"] + (1 - p6.MAKER_SHARE_ASSUMED) * costs["taker_bps"]
    return simulate(w.loc[t], ret.loc[t], wide(panel, "funding_rate").loc[t], per_side).daily()


def shadow(conn, symbols, start: pd.Timestamp, end: pd.Timestamp, costs: pd.DataFrame) -> pd.Series:
    """Research path on the paper period: same models (by train_end), band, maker share, cost model."""
    panel = build_panel(conn, symbols, start=start - pd.Timedelta(days=p6.HISTORY_DAYS), end=end)
    X, _ = feature_frame(panel, sig.compute(panel))
    sc = p6_scores(X, end, get_settings().models_dir)
    if sc is None:
        return pd.Series(dtype=float)
    return shadow_from_weights(panel, p6.target_weights(panel, sc), costs, start, end)["net"]


def main() -> None:
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        eq, fills, dec, fund, st = paper_frames(conn)
        if eq.empty or st is None:
            print("no paper data yet: start alpha-paper (uv run python -m alpha.live.engine)")
            return
        start_eq = float(st["start_equity"])
        start = pd.Timestamp(st["started_at"]).tz_convert("UTC") if "started_at" in st else eq["ts"].min()
        end = pd.Timestamp.now(tz="UTC")
        eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
        e = eq.set_index("ts")["equity"]
        daily_eq = e.resample("D").last().dropna()
        paper_daily = (daily_eq.diff() / start_eq).dropna()  # in starting-equity terms, like the backtest
        costs = symbol_costs(conn, cached_panel(conn, s.symbols))  # same cost model as the backtests (DEV depth)
        sh = shadow(conn, s.symbols, start, end, costs)

    # ---- execution quality ----
    traded = (fills["qty"] * fills["price"]).sum() if not fills.empty else 0.0
    slip_usd = (fills["slippage_bps"] / 1e4 * fills["qty"] * fills["arrival_mid"]).sum() if not fills.empty else 0.0
    realized_cost_bps = (fills["fee"].sum() + slip_usd) / traded * 1e4 if traded else np.nan
    model_bps = float((p6.MAKER_SHARE_ASSUMED * costs["maker_bps"] + (1 - p6.MAKER_SHARE_ASSUMED) * costs["taker_bps"]).mean())
    maker_share = (fills.loc[fills["liquidity"] == "maker", "qty"] * fills.loc[fills["liquidity"] == "maker", "price"]).sum() / traded if traded else np.nan
    minutes = max(1, int((end - start).total_seconds() // 60))
    uptime = len(eq) / minutes
    reb_bars = [b for b in pd.date_range(start.ceil("h"), end - pd.Timedelta(hours=2), freq="h") if p6.is_rebalance(b)]
    seen = set(pd.to_datetime(dec.loc[dec["action"].isin(["REBALANCE", "NO_TRADE", "FLATTEN"]), "bar_time"], utc=True)) if not dec.empty else set()
    missed = [str(b) for b in reb_bars if b not in seen]
    acct = Account(start_eq)
    for _, r in fills.iterrows():
        acct.apply_fill(r["symbol"], int(r["side"]), float(r["qty"]), float(r["price"]), r["liquidity"])
    acct.cash -= fund["amount"].sum() if not fund.empty else 0.0
    last_pos = eq["positions"].iloc[-1] or {}
    marks = {k: v["mark"] for k, v in last_pos.items() if v.get("mark")}
    rebuilt = acct.cash + sum(p.qty * (marks.get(sym, p.entry) - p.entry) for sym, p in acct.positions.items() if p.qty)
    both = pd.concat({"paper": paper_daily, "shadow": sh}, axis=1).dropna()
    corr = both["paper"].corr(both["shadow"]) if len(both) >= 10 else np.nan

    checks = {
        "uptime >= 99%": uptime >= 0.99,
        "no missed rebalance": not missed,
        "ledger reconciles (< $0.01)": abs(rebuilt - eq["equity"].iloc[-1]) < 0.01,
        "cost <= 1.25 x model": bool(np.isnan(realized_cost_bps) or realized_cost_bps <= 1.25 * model_bps),
        "maker share 0.48-0.72": bool(np.isnan(maker_share) or 0.48 <= maker_share <= 0.72),
        "no beyond-book fills": bool(fills.empty or not fills["beyond_book"].any()),
        "tracking corr >= 0.8 (>= 10 days)": bool(np.isnan(corr) or corr >= 0.8),
    }

    print(f"=== Paper trading {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC  ({(end - start).days} days) ===")
    print(f"start equity {start_eq:,.2f}   now {eq['equity'].iloc[-1]:,.2f}   fees {eq['fees'].iloc[-1]:,.2f}   "
          f"funding paid {eq['funding'].iloc[-1]:,.2f}   gross lev now {eq['gross_lev'].iloc[-1]:.2f}x")
    print(f"decisions: {dec['action'].value_counts().to_dict() if not dec.empty else {}}   fills: {len(fills)}   "
          f"traded ${traded:,.0f}")
    print(f"execution: realized cost {realized_cost_bps:.2f} bps/unit traded vs model {model_bps:.2f}; maker share "
          f"{maker_share:.0%}; avg slippage vs mid {fills['slippage_bps'].mean() if not fills.empty else float('nan'):.2f} bps; "
          f"taker latency used {fills['latency_ms'].dropna().mean() if not fills.empty else float('nan'):.0f} ms")
    print("\n             paper            shadow backtest   VALID-A expected   VALID-B observed")
    sp, ss = stats(paper_daily), stats(sh[sh.index >= start.floor('D')] if not sh.empty else sh)
    for k in ("days", "net", "pf", "hit_days", "max_dd", "sharpe", "sortino"):
        exp = VALID_A_EXPECTED.get(k, "")
        vb = VALID_B_OBSERVED.get(k, "")
        print(f"{k:>10}  {sp.get(k, float('nan')):>14.4f}  {ss.get(k, float('nan')):>16.4f}  {exp!s:>17}  {vb!s:>16}")
    if sp.get("days"):
        print(f"(Sharpe standard error at {sp['days']} days ~ +-{sp['sharpe_se']:.1f}: profit is not decidable yet)")
    ps = per_symbol_paper(eq)
    if not ps.empty:
        print("\nper token gross price P&L ($):", ps.groupby("symbol")["pnl"].sum().round(2).to_dict())
        print("long vs short gross P&L ($):", ps.groupby("side")["pnl"].sum().round(2).to_dict())
    print("\nEngineering gate:")
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    if missed:
        print("  missed rebalance bars:", missed[:10])
    out = Path("data/experiments/paper_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checks": checks, "paper": sp, "shadow": ss, "realized_cost_bps": realized_cost_bps,
                               "model_cost_bps": model_bps, "maker_share": maker_share, "uptime": uptime},
                              default=str, indent=2))


if __name__ == "__main__":
    sys.exit(main())
