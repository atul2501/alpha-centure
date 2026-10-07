"""reports/COMPARISON_OCT2025_SEP2026.md from data/experiments/compare_2025_10/books.json (plain-language report)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.tournament.experiments.compare import END, EQUITY, OUT, START

REPORT = Path("reports/COMPARISON_OCT2025_SEP2026.md")
ORDER = ["P6 (60% maker, as in paper)", "random_forest", "decision_tree", "linear_svm", "buy_hold"]
LABEL = {"P6 (60% maker, as in paper)": "P6 (your model)", "random_forest": "Random forest",
         "decision_tree": "Decision tree", "linear_svm": "Linear SVM", "buy_hold": "Buy-and-hold (model)"}
WHAT = {
    "P6 (your model)": "ridge regression on slow momentum (1-30 days), rebalanced every 72 h, long and short across the "
                       "23 coins, volatility-targeted to 20%/year (up to 3x), mostly limit orders",
    "Random forest": "200-tree random forest on 15-minute features (price, volume, volatility, momentum, trend, order "
                     "flow, funding, BTC/ETH context), one model per coin, retrained every quarter",
    "Decision tree": "a single shallow decision tree on the same features, per coin, retrained every quarter",
    "Linear SVM": "linear support-vector classifier (up / flat / down) on the same features, per coin",
    "Buy-and-hold (model)": "the tournament's 'always long' rule: it may sit out a quarter when the previous 90 days "
                            "lost money",
}


def usd(x):
    return "–" if x is None or not np.isfinite(x) else f"{'−' if x < 0 else ''}${abs(x) * EQUITY:,.0f}"


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def load() -> dict:
    b = json.loads((OUT / "books.json").read_text())
    return {k: {kk: (np.nan if vv is None else vv) for kk, vv in v.items()} for k, v in b.items()}


def verdicts(b: dict) -> dict[str, str]:
    out = {}
    best = max(ORDER, key=lambda k: b[k]["sharpe"] if np.isfinite(b[k]["sharpe"]) else -9)
    for k in ORDER:
        v = b[k]
        if v["net"] <= 0:
            out[k] = "Lost money after costs"
        elif k == best and np.isfinite(v.get("net_x2", np.nan)) and v["net_x2"] > 0:
            out[k] = "Best risk-adjusted, survives 2× costs"
        elif np.isfinite(v.get("net_x2", np.nan)) and v["net_x2"] <= 0:
            out[k] = "Profitable, but not at 2× costs"
        else:
            out[k] = "Profitable"
        if k == best and not out[k].startswith("Best") and v["net"] > 0:
            out[k] = "Best risk-adjusted; " + out[k].lower()
    return out


def write() -> Path:
    b = load()
    market = b.get("hold all 23 coins (market)", {})
    v = verdicts(b)
    ranked = sorted(ORDER, key=lambda k: -(b[k]["sharpe"] if np.isfinite(b[k]["sharpe"]) else -9))
    best = ranked[0]
    L = ["# Which model is best? P6 vs the top-4 tournament models", "",
         f"_23 coins · 1 Oct 2025 → 30 Sep 2026 · $30,000 starting capital · generated "
         f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_", "", "## The short answer", ""]
    if b[best]["net"] > 0:
        L.append(f"**{LABEL[best]}** did best over the last 12 months: {usd(b[best]['net'])} on $30,000 "
                 f"({pct(b[best]['net'])}), Sharpe {num(b[best]['sharpe'])}, worst drawdown {pct(b[best]['max_dd'])}.")
    else:
        L.append("**Every model lost money over the last 12 months.** The least bad was "
                 f"{LABEL[best]} ({usd(b[best]['net'])}).")
    p6 = b["P6 (60% maker, as in paper)"]
    others = [k for k in ORDER if k != "P6 (60% maker, as in paper)"]
    beat = [LABEL[k] for k in others if b[k]["sharpe"] > p6["sharpe"]]
    L += ["", ("P6 had the best return for the risk taken; none of the four tournament models beat it."
               if not beat else f"Beat P6 on risk-adjusted return: {', '.join(beat)}."),
          f"The market itself (holding all 23 coins equally) returned {pct(market.get('net'))} over the same year.", "",
          "## Ranking (best first)", "",
          "| Rank | Model | Profit on $30k | Return | Sharpe | Worst drawdown | Profit at 2× costs | Trades | "
          "Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for i, k in enumerate(ranked, 1):
        x = b[k]
        L.append(f"| {i} | **{LABEL[k]}** | {usd(x['net'])} | {pct(x['net'])} | {num(x['sharpe'])} | "
                 f"{pct(x['max_dd'])} | {usd(x.get('net_x2'))} | {int(x['trades']):,} | {v[k]} |")
    L += ["", "How to read it: **Sharpe** = return per unit of risk (above 1 is good, below 0 lost money). It is the "
          "fairest single number here because P6 uses up to 3× leverage and the others at most 1×. **Worst "
          "drawdown** = biggest fall from a high, as a share of the $30k. **Profit at 2× costs** = the same trades "
          "if fees and slippage were twice as high (a safety check).", "",
          "## Same risk, side by side", "",
          "If every model were sized to have the same day-to-day swings as P6, profits would have been:", "",
          "| Model | Profit at P6's risk level | Yearly volatility as run |", "|---|---|---|"]
    for k in ranked:
        L.append(f"| {LABEL[k]} | {usd(b[k]['vol_matched_net'])} | {pct(b[k]['vol_ann'])} |")
    L += ["", "## Details", "", "| | " + " | ".join(LABEL[k] for k in ORDER) + " |", "|---|" + "---|" * len(ORDER)]
    rows = [("Profit", lambda x: usd(x["net"])), ("Fees + slippage", lambda x: usd(x["fees"] + x["slippage"])),
            ("Funding (− = received)", lambda x: usd(x["funding"])), ("Sortino", lambda x: num(x["sortino"])),
            ("Calmar (return / drawdown)", lambda x: num(x["calmar"])), ("Winning days", lambda x: pct(x["win_days"], 0)),
            ("Profitable months", lambda x: f"{int(x['months_pos'])} of {int(x['months'])}"),
            ("Best month", lambda x: usd(x["best_month"])), ("Worst month", lambda x: usd(x["worst_month"])),
            ("Profit at 1.5× costs", lambda x: usd(x.get("net_x1_5"))),
            ("Long trades P&L", lambda x: usd(x.get("long"))), ("Short trades P&L", lambda x: usd(x.get("short"))),
            ("Coins profitable", lambda x: f"{int(x['coins_pos'])} of 23" if np.isfinite(x.get("coins_pos", np.nan))
             else "–")]
    for name, fn in rows:
        L.append(f"| {name} | " + " | ".join(fn(b[k]) for k in ORDER) + " |")
    L += ["", "P6's trade count is the number of coin position changes at its 72-hour rebalances; the others count "
          "round trips. P6's fees and slippage are one combined number.", "", "## Month by month", "",
          "| Month | " + " | ".join(LABEL[k] for k in ORDER) + " | Market |", "|---|" + "---|" * (len(ORDER) + 1)]
    months = sorted(set().union(*[b[k]["monthly"].keys() for k in ORDER]))
    for m in months:
        L.append(f"| {m} | " + " | ".join(usd(b[k]["monthly"].get(m, 0.0)) for k in ORDER)
                 + f" | {usd(market.get('monthly', {}).get(m, 0.0))} |")
    L += ["", "## Per coin (tournament models)", "", "| Coin | " + " | ".join(LABEL[k] for k in others) + " |",
          "|---|" + "---|" * len(others)]
    for c in sorted(b["random_forest"]["per_coin"]):
        L.append(f"| {c.replace('USDT', '')} | " + " | ".join(usd(b[k]["per_coin"].get(c, 0.0)) for k in others) + " |")
    L += ["", "## What each model is", ""] + [f"- **{k}**: {t}." for k, t in WHAT.items()]
    L += ["", "## How the test was run", "",
          "- Same 23 coins and the same 12 months for every model. P&L is added up on a $30,000 start, not compounded.",
          "- Tournament models: one model per coin, trained only on data before each quarter, settings chosen on the "
          "90 days before the quarter, then traded through it. Each coin gets $30,000 / 23 ≈ $1,304. Every trade "
          "pays taker fees (0.05% per side), half the spread (2 × the coin's live median), price impact from that "
          "coin's order book and a latency allowance. Funding is charged at every settlement.",
          "- P6: its own research engine with quarterly retraining, 60% limit orders as in paper trading. A taker-only "
          f"run is also shown below: {usd(b.get('P6 taker-only', {}).get('net'))}, Sharpe "
          f"{num(b.get('P6 taker-only', {}).get('sharpe'))}.",
          "- HYPE has too little history for the first two quarters; it trades from April 2026 in the tournament "
          "models.", "", "## Important caveats", "",
          "- **One year is short.** Four quarters can flatter or punish any model by luck. Use this as a hint, not "
          "proof.",
          "- **This window was already used.** It is the VALID-B period, where P6 was checked before. These results "
          "describe what happened. Picking the winner from this one year is itself a choice that can be lucky. Each "
          "model's run here is recorded in the lockbox (COMPARE-2025-10).",
          "- The tournament models were designed and tested on SOL (2021-2024) and moved to 23 coins unchanged; on SOL's "
          "own 2021-2024 test none of them passed the significance gates.",
          "- Buy-and-hold and the long-heavy models mostly reflect whether the market went up or down that year.", "",
          "## Recommendation", ""]
    if best.startswith("P6"):
        L.append("Keep **P6** in paper trading. Nothing in this comparison is a better replacement. The paper results "
                 "over the next months will tell whether P6 really makes money.")
    else:
        L.append(f"{LABEL[best]} had the better year. Before switching, run it in paper next to P6 for 8-12 weeks: "
                 "one good year on an already-used window is not enough to replace a model.")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")
    return REPORT
