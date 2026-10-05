"""Strategy health monitor: an automatic circuit breaker with automatic resume.

Each day the strategy's own shadow P&L (the research path replayed on recent data with the models it had at each
time, same costs) is checked:
    PAUSE   when the shadow is more than PAUSE_DRAWDOWN below its peak over the last DD_DAYS
    RESUME  when the shadow's last SHARPE_DAYS annualized Sharpe is above RESUME_SHARPE again
The shadow keeps running while paused, so resuming never depends on trades the account did not make.

Chosen on DEV only (2021-01 -> 2024-06, P6): on/off rules on 90/180-day Sharpe cut net return from +140% to
+100-106% and raised drawdown (whipsaw); this drawdown breaker never triggered there (DEV max drawdown 16%), so it
costs nothing in normal conditions and only acts when losses are unusually deep. Not tuned on any later data.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from loguru import logger

from alpha.strategy import p6

PAUSE_DRAWDOWN = 0.20
RESUME_SHARPE = 0.5
SHARPE_DAYS = 60
DD_DAYS = 365
MIN_DAYS = 40


@dataclass
class MonitorState:
    active: bool = True
    reason: str = "warming up"
    since: str = ""
    drawdown: float = float("nan")
    sharpe: float = float("nan")
    checked: str = ""

    def to_json(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, d: dict | None) -> "MonitorState":
        if not d:
            return cls()
        return cls(**{k: (float("nan") if v is None and k in ("drawdown", "sharpe") else v) for k, v in d.items()})


def evaluate(daily: pd.Series, st: MonitorState, today: pd.Timestamp) -> MonitorState:
    """Pure state transition from the shadow's daily net returns (fraction of equity, index = day)."""
    d = daily[(daily.index > today - pd.Timedelta(days=DD_DAYS)) & (daily.index <= today)].dropna()
    new = MonitorState(st.active, st.reason, st.since, checked=str(today.date()))
    if len(d) < MIN_DAYS:
        new.reason = f"warming up ({len(d)} shadow days)"
        return new
    eq = pd.concat([pd.Series([0.0]), d.cumsum().reset_index(drop=True)])
    new.drawdown = float(eq.cummax().iloc[-1] - eq.iloc[-1])
    last = d.iloc[-SHARPE_DAYS:]
    new.sharpe = float(last.mean() / last.std() * np.sqrt(365)) if len(last) >= MIN_DAYS and last.std() > 0 else float("nan")
    if st.active and new.drawdown > PAUSE_DRAWDOWN:
        new.active, new.since = False, str(today.date())
        new.reason = f"paused: shadow drawdown {new.drawdown:.1%} > {PAUSE_DRAWDOWN:.0%}"
    elif not st.active and np.isfinite(new.sharpe) and new.sharpe > RESUME_SHARPE:
        new.active, new.since = True, str(today.date())
        new.reason = f"resumed: {SHARPE_DAYS}d shadow Sharpe {new.sharpe:.2f} > {RESUME_SHARPE}"
    elif st.active:
        new.reason = f"active: shadow drawdown {new.drawdown:.1%}, {SHARPE_DAYS}d Sharpe {new.sharpe:.2f}"
    else:
        new.reason = f"paused since {st.since}: {SHARPE_DAYS}d shadow Sharpe {new.sharpe:.2f} <= {RESUME_SHARPE}"
    return new


def ensure_monthly_bundles(conn: psycopg.Connection, symbols: list[str], models_dir: str, now: pd.Timestamp,
                           months: int = 14) -> None:
    """Models as the live engine would have had them: one per month start over the shadow window."""
    have = {b.stem for b in Path(models_dir).glob("p6_ridge_*.joblib")}
    first = (now - pd.DateOffset(months=months)).normalize().replace(day=1)
    for m in pd.date_range(first, now, freq="MS"):
        if f"p6_ridge_{m:%Y%m%d}" not in have:
            logger.info("monitor: training shadow model for {}", m.date())
            p6.train(conn, symbols, m).save(models_dir)


def shadow_daily(conn: psycopg.Connection, symbols: list[str], models_dir: str, now: pd.Timestamp) -> pd.Series:
    from alpha.live.report import shadow
    from alpha.research.panel import cached_panel
    from alpha.research.screen import symbol_costs

    ensure_monthly_bundles(conn, symbols, models_dir, now)
    costs = symbol_costs(conn, cached_panel(conn, symbols))
    return shadow(conn, symbols, now - pd.Timedelta(days=DD_DAYS + 10), now, costs)
