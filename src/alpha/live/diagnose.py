"""Where does the paper account lose? Direction hit rate vs payoff, per coin / side / cost item / strategy part.

    uv run python -m alpha.live.diagnose                 # paper period (since the account started)
    uv run python -m alpha.live.diagnose --days 180      # strategy-part analysis over a longer recent window

"Right direction" is counted per (72h rebalance period, coin): did the position's price P&L over the period end
positive. A trend / carry book is expected to be near 50% on that count; it makes money only if winners are larger
than losers (payoff ratio > 1), so every table prints hit rate, average win, average loss and payoff side by side.

Sections
    1  paper account   price P&L vs fees vs slippage vs funding; by coin and by side, with hit rate and payoff
    2  strategy parts  research path on the same window: ridge, rules, carry and the traded book, each held on the
                       72h schedule; standalone Sharpe / hit rate / payoff and approximate contribution to the book
    3  ridge forecast  rank IC and sign accuracy of the forecast vs the realized 72h move, overall and per coin
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from alpha.config import get_settings
from alpha.live.report import p6_scores, q, shadow_from_weights
from alpha.research import signals as sig
from alpha.research.models import feature_frame, target
from alpha.research.panel import build_panel, cached_panel, wide
from alpha.research.screen import symbol_costs
from alpha.strategy import p6

MIX = {"ridge": 0.5, "rules": 0.25, "carry": 0.25}


def period_of(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """72h rebalance period id (UTC-aligned, same schedule as p6.is_rebalance)."""
    hours = (pd.DatetimeIndex(ts) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
    return np.asarray(hours // p6.H)


def direction_stats(pnl: pd.Series) -> dict:
    """Hit rate and payoff of a set of (period, coin) P&Ls. Zero P&L (no position) is ignored."""
    x = pnl[pnl != 0].dropna()
    if x.empty:
        return {"n": 0}
    win, loss = x[x > 0], x[x < 0]
    avg_w = float(win.mean()) if len(win) else 0.0
    avg_l = float(-loss.mean()) if len(loss) else 0.0
    return {"n": len(x), "hit": float((x > 0).mean()), "avg_win": avg_w, "avg_loss": avg_l,
            "payoff": avg_w / avg_l if avg_l > 0 else np.inf, "net": float(x.sum()),
            "breakeven_hit": avg_l / (avg_w + avg_l) if avg_w + avg_l > 0 else np.nan}


def table(df: pd.DataFrame, by: str, col: str = "pnl") -> pd.DataFrame:
    rows = {k: direction_stats(g[col]) for k, g in df.groupby(by)}
    return pd.DataFrame(rows).T.sort_values("net")


def held_on_schedule(w: pd.DataFrame, every: int = p6.H) -> pd.DataFrame:
    """Weights as they would be held: taken at rebalance hours and kept until the next one."""
    hours = (w.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
    return w.where(pd.Series(np.asarray(hours % every == 0), index=w.index), axis=0).ffill().fillna(0.0)


def period_pnl(held: pd.DataFrame, ret: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Long frame (period, symbol, side, price, funding, pnl) from hourly held weights (fraction of equity).
    Weights decided at bar t earn bar t+1's return, as in portfolio_sim.simulate."""
    pos = held.shift(1).fillna(0.0)
    price = pos * np.expm1(ret.reindex_like(pos).fillna(0.0))
    fund = -pos * funding.reindex_like(pos).fillna(0.0)
    per = period_of(pos.index)
    p = price.groupby(per).sum().stack().rename("price")
    f = fund.groupby(per).sum().stack().rename("funding")
    side = np.sign(pos.groupby(per).mean()).stack().rename("side_sign")
    out = pd.concat([p, f, side], axis=1).reset_index()
    out.columns = ["period", "symbol", "price", "funding", "side_sign"]
    out = out[out["side_sign"] != 0]
    out["side"] = np.where(out["side_sign"] > 0, "long", "short")
    out["pnl"] = out["price"] + out["funding"]
    return out.drop(columns="side_sign")


# ---------------- 1. paper account ----------------

def paper_minutes(eq: pd.DataFrame) -> pd.DataFrame:
    """Per-minute gross price P&L per coin from consecutive equity snapshots: qty_{t-1} * (mark_t - mark_{t-1})."""
    rows, prev = [], {}
    for ts, pos in zip(eq["ts"], eq["positions"]):
        pos = pos or {}
        for s, p in pos.items():
            a = prev.get(s)
            if a and a.get("mark") is not None and p.get("mark") is not None:
                rows.append((ts, s, "long" if a["qty"] > 0 else "short", a["qty"] * (p["mark"] - a["mark"])))
        prev = pos
    return pd.DataFrame(rows, columns=["ts", "symbol", "side", "pnl"])


def paper_section(conn, start: pd.Timestamp) -> dict:
    eq = q(conn, "SELECT ts, equity, fees, funding, positions FROM paper_equity WHERE ts >= %s ORDER BY ts", (start,))
    if eq.empty:
        print("no paper equity marks in the window")
        return {}
    eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
    fills = q(conn, "SELECT symbol, side, price, qty, fee, liquidity, slippage_bps, arrival_mid FROM paper_fills "
                    "WHERE ts >= %s", (start,))
    fund = q(conn, "SELECT ts, symbol, amount FROM paper_funding WHERE ts >= %s", (start,))
    m = paper_minutes(eq)
    m["period"] = period_of(m["ts"])
    pp = m.groupby(["period", "symbol", "side"], as_index=False)["pnl"].sum()

    start_eq, end_eq = float(eq["equity"].iloc[0]), float(eq["equity"].iloc[-1])
    price_total = float(m["pnl"].sum())
    fees = float(fills["fee"].sum()) if not fills.empty else 0.0
    slip = float((fills["slippage_bps"] / 1e4 * fills["qty"] * fills["arrival_mid"]).sum()) if not fills.empty else 0.0
    funding = float(fund["amount"].sum()) if not fund.empty else 0.0
    traded = float((fills["qty"] * fills["price"]).sum()) if not fills.empty else 0.0
    print(f"\n=== 1. Paper account {eq['ts'].iloc[0]:%Y-%m-%d} -> {eq['ts'].iloc[-1]:%Y-%m-%d} "
          f"({(eq['ts'].iloc[-1] - eq['ts'].iloc[0]).days} days) ===")
    print(f"equity {start_eq:,.2f} -> {end_eq:,.2f}  ({end_eq / start_eq - 1:+.2%})")
    print(f"  price moves (mark to mark)   {price_total:+12,.2f}")
    print(f"  fees                         {-fees:+12,.2f}")
    print(f"  funding (received > 0)       {-funding:+12,.2f}")
    print(f"  (slippage vs mid, already inside price moves: {-slip:+,.2f}; traded ${traded:,.0f} = "
          f"{traded / start_eq:.1f}x equity)")
    print("\nDirection per (72h period, coin):", fmt(direction_stats(pp["pnl"])))
    print("\nBy coin (worst first; $):")
    by_coin = table(pp, "symbol")
    if not fills.empty:
        by_coin["fees"] = fills.groupby("symbol")["fee"].sum().reindex(by_coin.index).fillna(0.0)
    if not fund.empty:
        by_coin["funding_paid"] = fund.groupby("symbol")["amount"].sum().reindex(by_coin.index).fillna(0.0)
    print(by_coin.round(3).to_string())
    print("\nBy side ($):")
    print(table(pp, "side").round(3).to_string())
    per_day = m.groupby(m["ts"].dt.floor("D"))["pnl"].sum()
    if len(per_day):
        top = per_day.reindex(per_day.abs().sort_values(ascending=False).index[:5])
        print("\nBiggest price days ($):", {str(k.date()): round(v, 2) for k, v in top.items()},
              f" share of |P&L| from the 5 biggest days: {top.abs().sum() / per_day.abs().sum():.0%}")
    return {"equity_start": start_eq, "equity_end": end_eq, "price": price_total, "fees": fees, "funding_paid": funding,
            "slippage": slip, "direction": direction_stats(pp["pnl"]),
            "by_coin": by_coin.to_dict(orient="index"), "by_side": table(pp, "side").to_dict(orient="index")}


# ---------------- 2 + 3. strategy parts and ridge forecast ----------------

def sleeves(panel: pd.DataFrame, sc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = p6.target_weights(panel, sc)
    parts = {"ridge": base, "rules": p6.rules_weights(panel).reindex_like(base).fillna(0.0),
             "carry": p6.carry_weights(panel).reindex_like(base).fillna(0.0)}
    book = p6.book_weights(panel, sc)
    mix = sum(MIX[k] * parts[k] for k in parts)
    scale = (book.abs().sum(axis=1) / mix.abs().sum(axis=1).replace(0, np.nan)).fillna(0.0)
    contrib = {f"{k} (in book)": parts[k].mul(MIX[k] * scale, axis=0) for k in parts}
    return {**parts, "book": book, **contrib}


def summarize(pp: pd.DataFrame) -> dict:
    daily = pp.groupby("period")["pnl"].sum()  # per 72h period, fraction of equity
    sd = daily.std()
    return {**direction_stats(pp["pnl"]), "periods": len(daily),
            "sharpe": float(daily.mean() / sd * np.sqrt(365 * 24 / p6.H)) if sd and sd > 0 else np.nan,
            "price": float(pp["price"].sum()), "funding": float(pp["funding"].sum())}


def forecast_quality(panel: pd.DataFrame, sc: pd.DataFrame, start: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    ret = wide(panel, "ret")
    vol = ret.rolling(sig.VOL_WINDOW, min_periods=48).std()
    y = target(panel, vol, p6.H).unstack("symbol").reindex_like(ret)
    el = wide(panel, "eligible").fillna(False).astype(bool)
    hours = (ret.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
    t = ret.index[(np.asarray(hours % p6.H == 0)) & (ret.index >= start)]
    s = sc.reindex(index=t, columns=ret.columns).where(el.loc[t])
    r = y.loc[t].where(s.notna())
    ok = r.notna() & s.notna()
    ic = s.rank(axis=1).corrwith(r.rank(axis=1), axis=1).dropna()
    sign = (np.sign(s) == np.sign(r)).where(ok)
    per_coin = pd.DataFrame({"sign_right": sign.mean(), "n": ok.sum(),
                             "ic": s.corrwith(r, method="spearman")}).sort_values("sign_right")
    out = {"rebalances": len(ic), "rank_ic_mean": float(ic.mean()) if len(ic) else np.nan,
           "rank_ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 2 and ic.std() > 0 else np.nan,
           "ic_positive_share": float((ic > 0).mean()) if len(ic) else np.nan,
           "sign_right": float(sign.stack().mean()) if ok.any().any() else np.nan}
    return out, per_coin


def strategy_section(conn, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, models_dir: str) -> dict:
    panel = build_panel(conn, symbols, start=start - pd.Timedelta(days=p6.HISTORY_DAYS), end=end)
    X, _ = feature_frame(panel, sig.compute(panel))
    sc = p6_scores(X, end, models_dir)
    if sc is None:
        print("no ridge model bundles in", models_dir)
        return {}
    ret, funding = wide(panel, "ret"), wide(panel, "funding_rate")
    print(f"\n=== 2. Strategy parts, research path {start:%Y-%m-%d} -> {end:%Y-%m-%d} (no costs except 'book net') ===")
    res, coin_tabs = {}, {}
    for name, w in sleeves(panel, sc).items():
        pp = period_pnl(held_on_schedule(w), ret, funding)
        pp = pp[pp["period"] >= period_of(pd.DatetimeIndex([start]))[0]]
        res[name] = summarize(pp)
        coin_tabs[name] = pp.groupby("symbol")["pnl"].sum()
    costs = symbol_costs(conn, cached_panel(conn, symbols))
    net = shadow_from_weights(panel, p6.book_weights(panel, sc), costs, start, end)
    res["book"]["net_after_costs"] = float(net["net"].sum())
    res["book"]["cost"] = float(net["cost"].sum())
    print(pd.DataFrame(res).T[["n", "hit", "breakeven_hit", "avg_win", "avg_loss", "payoff", "net", "sharpe",
                               "price", "funding"]].to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"book after band + cost model: net {res['book']['net_after_costs']:+.4f}, cost {res['book']['cost']:.4f} "
          "(fraction of equity)")
    contrib = pd.DataFrame({k: v for k, v in coin_tabs.items() if "(in book)" in k or k == "book"}).fillna(0.0)
    print("\nPer coin net (fraction of equity), worst first:")
    print(contrib.sort_values("book").to_string(float_format=lambda v: f"{v:+.4f}"))

    fq, per_coin = forecast_quality(panel, sc, start)
    print("\n=== 3. Ridge forecast vs realized 72h move (eligible coins, rebalance hours) ===")
    print(fmt(fq))
    print(per_coin.to_string(float_format=lambda v: f"{v:.3f}"))
    return {"parts": res, "per_coin": contrib.to_dict(), "forecast": fq, "forecast_per_coin": per_coin.to_dict()}


def fmt(d: dict) -> str:
    return ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=None, help="strategy-part window (default: the paper period)")
    ap.add_argument("--no-strategy", action="store_true", help="paper account only (fast)")
    a = ap.parse_args()
    s = get_settings()
    end = pd.Timestamp.now(tz="UTC").floor("h")
    out = {}
    with psycopg.connect(s.database_url) as conn:
        st = conn.execute("SELECT value FROM paper_state WHERE key = 'account'").fetchone()
        paper_start = pd.Timestamp(st[0]["started_at"]).tz_convert("UTC") if st and "started_at" in st[0] else None
        if paper_start is not None:
            out["paper"] = paper_section(conn, paper_start)
        else:
            print("no paper account yet")
        if not a.no_strategy:
            start = end - pd.Timedelta(days=a.days) if a.days else (paper_start or end - pd.Timedelta(days=90))
            out["strategy"] = strategy_section(conn, s.symbols, start.floor("h"), end, s.models_dir)
    path = Path("data/experiments/diagnose.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, default=str, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
