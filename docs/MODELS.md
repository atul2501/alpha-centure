# Alpha Centure: every model and every branch

Status on 9 Oct 2026. This document is the one place that describes all models side by side; each strategy branch
describes only its own model. All results are **paper/replay results with modelled costs, not real trades**, and
none of the differences between the top models is statistically proven (see [How much to trust this](#how-much-to-trust-this)).

## Contents
1. [Branches](#branches)
2. [The shared pipeline](#the-shared-pipeline)
3. [The models](#the-models)
4. [Results, Jan 2021 → 7 Oct 2026](#results-jan-2021--7-oct-2026)
5. [How we got here (research history)](#how-we-got-here-research-history)
6. [How the models react to the market](#how-the-models-react-to-the-market)
7. [Small accounts: $300, $500, $1,000](#small-accounts-300-500-1000)
8. [How much to trust this](#how-much-to-trust-this)
9. [Switching the server to a branch](#switching-the-server-to-a-branch)
10. [Published reports](#published-reports)

---

## Branches

| Branch | Commit | Trades | Contains | Use |
|---|---|---|---|---|
| `main_v2` | `2cbdd14` | **P6** | P6 live engine + the **shadow league** (P6, R1, R5, R6, N1, BTC scored daily) | Current server version |
| `R5` | `2cbdd14` | P6 | Same commit as `main_v2` | Working branch |
| `main_v1` | `6e3a2af` | P6 | `main_v2` merged by PR #1 (before the latest fixes) | Older |
| `n1-profit` | `0ce100f` | **N1** | N1 only (league removed) | Ready to deploy |
| `r1-balance` | `fd0feeb` | **R1** | R1 only (league removed) | Ready to deploy |
| `v4-carry` | `85f7eed` | **V4_carry** | V4 only (league removed) | Ready to deploy; recommended for a $1,000 paper start |
| `v1-rules-heavy` | `d77f61d` | **V1_rules_heavy** | V1 only (league removed), marked research | Research |
| `n1-improve` | `d26b433` | N1 | N1 + the six-variant test (`alpha.research.n1_variants`) | Research |
| `model-reports` | this commit | – | This document only (branched from `main_v2`) | Documentation |
| `main`, `new_model` | – | – | Earlier history, not part of this work | – |

Only one branch can run on the server at a time (one paper account).

## The shared pipeline

Every model below uses the same machinery; only the "book" (how target positions are formed) differs.

| Step | What happens |
|---|---|
| Universe | 23 Binance USD-M perpetuals: BTC ETH SOL SUI TRX AAVE BNB XRP HYPE LINK ADA UNI LTC AVAX ATOM DOGE DOT NEAR OP ARB WLD CAKE POL |
| Data | Collector: 1m–1d candles, order flow, top of book, funding, premium, open interest, long/short ratios; history from data.binance.vision |
| Ridge forecast | 19 features per coin (1/2/4-week momentum, trend breakouts, funding, basis, OI, positioning, flow, Bitcoin's trend) → 72-hour vol-normalized return forecast. Ridge regression (alpha 100), retrained monthly live / quarterly in replays |
| Momentum rules | No trained model: cross-sectional momentum (rank by 1/2/4-week trend, long strongest, short weakest, dollar-neutral) + time-series momentum (2/4-week trend, 20-day breakout), 50/50 |
| Funding carry | Short coins whose 24h-average funding is high (crowded longs pay), long coins with low funding, dollar-neutral |
| Sizing | Inverse volatility, scaled to a **20% yearly volatility** target; caps **3x gross**, **0.5x per coin** |
| Schedule | Rebalance every **72 hours**; skip changes smaller than the **1% no-trade band** |
| Execution (paper) | Maker limit order at the best price for 20 minutes, then taker for the rest; modelled as 60% maker / 40% taker |
| Risk | −3% day → reduce-only for the day; **−30% drawdown → flatten everything and halt** |

## The models

| Model | Book | Idea | Branch |
|---|---|---|---|
| **P6** | Ridge forecast only | The original finalist: one trained model | `main_v2` |
| **R5** cost band | P6, but a coin changes only if its expected 72h move is > 1.5× its round-trip cost | Skip weak, expensive trades | league only |
| **R1** ensemble | 0.5 × P6 + 0.5 × momentum rules | Two views must agree to bet big; calmer | `r1-balance` |
| **R6** carry sleeve | 0.75 × P6 + 0.25 × funding carry | Earn funding, smoother | league only |
| **N1** | R1 scaled back up to the 20% vol target | R1's decisions at full risk | `n1-profit` |
| **V1** rules-heavy | vol-target(0.3 × ridge + 0.7 × rules) | Lean on the untrained rules | `v1-rules-heavy` |
| V2 ridge-heavy | vol-target(0.7 × ridge + 0.3 × rules) | Lean on the model | tested only |
| V3 cost band | N1 + R5's cost rule | Fewer weak trades | tested only |
| **V4** carry | vol-target(0.5 × ridge + 0.25 × rules + 0.25 × carry) | N1 plus funding income | `v4-carry` |
| V5 2% band | N1 with a 2% no-trade band | Less turnover | tested only |
| V6 15% risk | 0.75 × N1 | Same model, smaller size | tested only |
| BTC | 1× long Bitcoin | Market reference | – |

Example of how the blends combine positions (illustrative):

| Coin | Ridge | Rules | Carry | R1 | N1 (≈ R1 × 1.3) | V4 mix before scaling |
|---|---|---|---|---|---|---|
| SOL | +$4,000 | +$6,000 | −$2,000 | +$5,000 | ≈ +$6,500 | +$3,000 |
| DOGE | +$3,000 | −$3,000 | −$4,000 | $0 | $0 | −$250 |
| LINK | −$2,000 | $0 | +$2,000 | −$1,000 | ≈ −$1,300 | −$500 |

## Results, Jan 2021 → 7 Oct 2026

Same engine code as live, ridge retrained quarterly, frozen per-coin cost table, $30,000 start, 23 coins, after fees,
slippage and funding. Columns: end value | Sharpe | worst drop | worst month.

### Full period

| Model | $30k → | Sharpe | Worst drop | Worst month |
|---|---|---|---|---|
| **V4** | **$228,509** | **1.64** | 17.9% | −7.9% |
| V3 | $211,232 | 1.53 | 25.7% | −8.9% |
| **N1** | $210,035 | 1.54 | 26.1% | −8.1% |
| V5 | $205,664 | 1.52 | 25.9% | −8.5% |
| **V1** | $200,508 | 1.52 | 24.1% | −7.6% |
| V2 | $188,857 | 1.47 | 26.1% | −9.6% |
| **R1** | $131,192 | 1.44 | 18.5% | **−5.0%** |
| V6 | $130,637 | 1.53 | 20.1% | −5.9% |
| R5 | $128,138 | 1.24 | 19.2% | −11.9% |
| **P6** | $123,379 | 1.22 | 19.1% | −11.9% |
| R6 | $109,467 | 1.38 | **12.8%** | −7.6% |
| BTC | $46,850 | 0.42 | 79.0% | −37.6% |

### By period

| Model | Jan 2021 – Jun 2024 (DEV) | Jul 2024 – Sep 2025 (VALID-A) | Oct 2025 – Oct 2026 (last 12 mo) |
|---|---|---|---|
| P6 | $87,349 · 1.53 · DD 15.3% | $38,611 · 1.00 · DD 19.1% | $32,924 · 0.51 · DD 15.9% |
| R5 | $86,800 · 1.50 · DD 14.9% | $39,252 · 1.04 · DD 19.2% | $33,849 · 0.63 · DD 14.0% |
| R1 | $87,106 · 1.72 · DD 12.3% | $35,126 · 0.76 · DD 18.5% | $38,590 · 1.34 · DD 9.8% |
| R6 | $75,175 · 1.66 · DD 11.5% | $40,594 · 1.40 · DD 11.1% | $32,285 · 0.49 · DD 12.8% |
| N1 | $112,736 · 1.78 · DD 16.4% | $37,180 · 0.82 · DD 26.1% | $45,098 · 1.67 · DD 12.0% |
| V1 | $109,099 · 1.74 · DD 15.5% | $33,543 · 0.49 · DD 24.1% | **$49,312 · 2.05 · DD 9.1%** |
| V4 | **$121,372 · 1.88 · DD 14.0%** | **$42,281 · 1.26 · DD 17.9%** | $40,076 · 1.30 · DD 14.7% |
| V6 | $80,404 · 1.75 · DD 12.4% | $35,465 · 0.82 · DD 20.1% | $41,231 · 1.71 · DD 8.7% |
| BTC | $39,719 · 0.44 · DD 79.0% | $50,517 · 1.14 · DD 29.3% | $21,014 · −0.56 · DD 53.9% |

Each period starts again from $30,000.

### By year

| Year | P6 | R5 | R1 | R6 | N1 | V1 | V4 | BTC |
|---|---|---|---|---|---|---|---|---|
| 2021 | +33.8% | +36.4% | +56.5% | +41.5% | +64.3% | **+72.0%** | +60.9% | +22.9% |
| 2022 | **+5.2%** | +1.4% | +4.5% | +0.1% | −0.7% | −0.9% | +3.4% | −66.3% |
| 2023 | +67.4% | +73.5% | +39.0% | +47.1% | +63.3% | +44.8% | **+75.3%** | +135.6% |
| 2024 | +44.4% | +41.3% | +40.0% | +47.5% | +54.1% | +57.9% | **+72.0%** | +95.4% |
| 2025 | +4.9% | +5.1% | +7.0% | +7.7% | +16.9% | +12.8% | **+17.8%** | −10.8% |
| 2026 (to 7 Oct) | +15.2% | +20.0% | +28.5% | +10.2% | +45.9% | **+52.1%** | +29.0% | −8.2% |

### Why V4 beats N1 over the full period

| Jan 2021 → Oct 2026 (% of start equity) | N1 | V4 |
|---|---|---|
| Funding | −13.2% paid | **+2.2% earned** |
| Trading costs | **−9.4%** | −16.5% |
| Funding + costs | −22.6% | **−14.3%** |
| Turnover per year | **40×** | 70× |
| Days more than 15% below the high | 53 | **3** |
| Worst stretch (5 Jul → 6 Nov 2024) | −24.1%, recovered Feb 2025 | **−15.5%, recovered Nov 2024** |

V4 lost less in five of N1's six worst months. If real trading costs are 1.5× the model its lead is $10.1k, at
2× it is $2.6k: V4's drawdown advantage is more robust than its profit advantage.

## How we got here (research history)

| Step | Result |
|---|---|
| Old 4-token setup system (LONG/SHORT/PASS) | +47.8R in backtests, almost all from 2020–23, ~0R since 2024 → retired and deleted |
| Phase 2 signal screen | Only slow momentum (1–4 weeks, 24–72h rebalance) beat costs; reversal, flow, mean reversion, carry, basis, OI did not alone; LightGBM worse than its shuffled control |
| Phase 4 | P6 built (DEV Sharpe 1.64 in its own test); failed the strict overfitting gates (deflated Sharpe ≤ 0.66, PBO 0.28) |
| VALID-A (Jul 2024 – Sep 2025) | P6 +44.5%, Sharpe 1.52 → pass |
| VALID-B (Oct 2025 – Oct 2026) | P6 −3.4%, Sharpe −0.15 → **fail; verdict NO TRADE** (paper trading chosen anyway) |
| Overnight round (154 trials) | Trading 23 coins instead of 14 was the only winner → now live |
| Round 2 (161 trials) | No winner (PBO 0.61); R1, R5, R6 came closest |
| SOL 15m tournament | Stopped; nothing passed a gate |
| Shadow league (`main_v2`) | From 9 Oct 2026, P6/R1/R5/R6/N1 scored daily on new data; switch rule pre-registered (reviews 7 Jan / 7 Apr 2027: beat P6 on net, Newey-West t ≥ 2.5, drawdown ≤ P6 + 5 pts) |
| N1 improvement test (`n1-improve`) | Six pre-registered variants, chosen on DEV only; **V4 passed**, others did not; deflated Sharpe 0.73, PBO 0.79, t vs N1 0.33 |

## How the models react to the market

Last two years (Oct 2024 → Oct 2026: Bitcoin +81% then −34%):

| Average month | Bitcoin | P6 | N1 | R6 |
|---|---|---|---|---|
| Bitcoin rose (13 months) | +11.2% | +0.9% | +0.9% | +0.2% |
| Bitcoin fell (12 months) | −8.9% | +2.6% | +5.0% | +3.0% |

All models are long/short and nearly independent of Bitcoin (daily correlation −0.1 to −0.2). They earn most in
sell-offs (June 2026: Bitcoin −20%, P6 +15%, N1 +17%) and lag strong rallies (October 2024: Bitcoin +12%, models −4% to −7%).
None of them is a bull-market strategy.

## Small accounts: $300, $500, $1,000

Binance USD-M order rules (fetched 8 Oct 2026): minimum order value $5 for most coins, $20 for ETH, LINK, LTC,
$50 for BTC; quantity steps up to ≈ $85 (0.001 BTC) and ≈ $18 (0.1 AAVE). The engine skips orders that are too small
(`exec/oms.py`), so small accounts hold fewer coins.

Snapshot at the 4 Oct 2026 targets:

| Account | V4 coins held (of 23) | V4 exposure achieved | N1 coins held | N1 exposure achieved |
|---|---|---|---|---|
| $300 | 8 | 53% | 3 | 21% |
| $500 | 10 | 58% | 6 | 45% |
| $1,000 | 15 | 78% | 13 | 69% |
| $3,000 | 19 | 92% | 20 | 90% |
| $10,000 | 23 | 100% | 23 | 97% |

Replay of the last two years as a **$1,000 account** (positions rounded to Binance rules at every rebalance, sized with
the running balance):

| | Ideal | As a $1,000 account | Lost to rounding | Avg coins held |
|---|---|---|---|---|
| V4 | $1,777 (+77.7%) | **$1,734 (+73.4%)** | 4.3 pts | 17.5 |
| N1 | $1,903 (+90.3%) | **$1,829 (+82.9%)** | 7.4 pts | 16.3 |

- $300–$500: mostly rounding noise; not a faithful test of any model.
- $1,000: workable (85–90% of the strategy); Bitcoin is almost never held (smallest order ≈ $85).
- $3,000: ~90%; $10,000: the full strategy.
- In these two years N1 beat V4 at $1,000; V4's case rests on the full history and its smaller drawdowns. For a
  $1,000 start V4 is suggested because N1's worst drop (26%) came within 4 points of the −30% kill switch.

## How much to trust this

- **Not proven.** Best gap to a reference: N1 vs P6 t = 1.89, V4 vs N1 t = 0.33; strong evidence needs t ≥ 2.5 or a
  deflated Sharpe ≥ 0.95 (V4: 0.73). More than 320 variants have been tried on this history.
- **Seen data.** VALID-A and VALID-B were used to judge P6; R1/N1/V-variants were designed after round 2 results that
  included the last 12 months. Only data after 9 Oct 2026 is truly new.
- **Absolute levels.** This replay uses the league's stricter cost table and per-year restarts; P6's DEV result here is
  $87k vs $115k in round 2. Rankings are comparable, exact dollar figures less so.
- **Costs are modelled.** Real paper fills (maker share, slippage) will show whether the 60% maker assumption holds;
  high-turnover models (V4, R6) are most sensitive.
- **Real money** needs months of positive paper results after costs first.

## Switching the server to a branch

```bash
cd ~/alpha-centure && git fetch && git checkout <branch> && git pull   # n1-profit | r1-balance | v4-carry | v1-rules-heavy
bash deploy/setup_ec2.sh
sudo systemctl disable --now alpha-league.timer 2>/dev/null; sudo rm -f /etc/systemd/system/alpha-league.*   # single-model branches have no league
sudo systemctl daemon-reload && sudo systemctl restart alpha-paper alpha-dashboard
```

Fresh paper account at a different size (backs up and clears the paper ledger; market data untouched):
```bash
sudo systemctl stop alpha-paper
sudo -u postgres pg_dump -d alpha -t 'paper_*' -Fc -f ~/paper_backup_$(date +%Y%m%d).dump
sudo -u postgres psql -d alpha -c "TRUNCATE paper_fills, paper_orders, paper_decisions, paper_funding, paper_equity, paper_state;"
echo "PAPER_EQUITY=1000" | sudo tee -a /etc/alpha/.env
sudo systemctl start alpha-paper
```

Back to P6 with the shadow league: `git checkout main_v2`, then the same setup and restart (remove `PAPER_EQUITY` from
`/etc/alpha/.env` to return to $30,000 on a fresh ledger).

## Published reports

| Report | Link |
|---|---|
| P6 and the shadow league | https://claude.ai/artifact/RKCfABPT8Mefh7c2vbgjFW |
| P6 vs R5, Oct 2025 → Oct 2026 | https://claude.ai/artifact/9x4uCjTasLiN7dTVyAG6QC |
| P6, R1, N1, R5, R6 compared 2021–2026 | https://claude.ai/artifact/FexonkJGmhKW1U77ckLV8L |
| P6, N1, R6 over the last two years | https://claude.ai/artifact/S3Eo4Q11BZthdAvaeCE8w5 |
| N1 improvement test (six variants) | https://claude.ai/artifact/HdU2JLXNZYFoc2cUSSfUFm |
| V4 vs N1, and $300 / $500 accounts | https://claude.ai/artifact/Ndk2nfSWVQKdiJVbh1jDEG |
| A $1,000 account | https://claude.ai/artifact/3gSD7EZP8sTJEG3XdcksfJ |

The report links are private to the account that created them; share them from each page's Share menu.
