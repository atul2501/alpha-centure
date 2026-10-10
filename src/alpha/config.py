from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PERP_SUFFIX = ".P"
# 23 perps since 2026-10-06 (overnight review: Idea 2, rank and trade 23 coins; user-approved)
UNIVERSE = "BTCUSDT,ETHUSDT,SOLUSDT,SUIUSDT,TRXUSDT,AAVEUSDT,BNBUSDT,XRPUSDT,HYPEUSDT,LINKUSDT,ADAUSDT,UNIUSDT,LTCUSDT,AVAXUSDT,ATOMUSDT,DOGEUSDT,DOTUSDT,NEARUSDT,OPUSDT,ARBUSDT,WLDUSDT,CAKEUSDT,POLUSDT"


def perp_symbol(symbol: str) -> str:
    """Stored name of a USD-M perpetual series (candles, perp flow/book): BTCUSDT -> BTCUSDT.P."""
    return symbol if symbol.endswith(PERP_SUFFIX) else symbol + PERP_SUFFIX


@dataclass(frozen=True)
class Feed:
    """One candle series: how it is stored (db_symbol) and fetched (api_symbol on market)."""

    db_symbol: str
    api_symbol: str
    market: str  # spot | perp
    interval: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://localhost:5432/alpha"
    # Plain strings (comma separated) so env vars like SYMBOLS=BTCUSDT,ETHUSDT work as-is.
    symbols_csv: str = Field(UNIVERSE, validation_alias=AliasChoices("SYMBOLS", "symbols_csv"))
    # Spot candles are off: the system trades and models USD-M perpetuals only.
    spot_enabled: bool = False
    intervals_csv: str = Field("1m,5m,15m,1h,4h,1d,1w", validation_alias=AliasChoices("INTERVALS", "intervals_csv"))
    # USD-M perpetual candles (what long/short trades execute on). Stored as e.g. BTCUSDT.P
    perp_intervals_csv: str = Field("1m,5m,15m,1h,4h,1d", validation_alias=AliasChoices("PERP_INTERVALS", "perp_intervals_csv"))
    backfill_1m_days: int = 30  # 1m candles kept (and backfilled) for this many days
    backfill_start: str = "2017-08-01"
    futures_enabled: bool = True
    orderflow_enabled: bool = True
    book_tick_seconds: int = 1   # top-of-book sample rate (spread/slippage calibration)
    oi_poll_seconds: int = 60    # live open interest poll (the 5m history endpoint keeps only ~30 days)
    models_dir: str = "models"
    # Paper trading (alpha.live.engine). There is no live order path; 'paper' is the only accepted mode.
    trading_mode: str = "paper"
    paper_equity: float = 30_000.0
    # Strategy monitor (pause at 20% shadow drawdown, auto-resume). Off: neutral on DEV (never triggered) and it cost
    # -$3,500 on $30k over Oct 2025 -> Oct 2026 (paused near lows, resumed after rebounds). See alpha.strategy.monitor.
    monitor_enabled: bool = False
    # Dashboard (python -m dashboard.server): bind address / port, optional basic-auth password
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8501
    dashboard_password: str = ""

    spot_rest: str = "https://api.binance.com"
    futures_rest: str = "https://fapi.binance.com"
    spot_ws: str = "wss://stream.binance.com:9443"
    # Binance routes USD-M market streams under /market; the bare host accepts connections but sends nothing.
    futures_ws: str = "wss://fstream.binance.com/market"
    # Depth / bookTicker streams are only served on /public (they never arrive on /market).
    futures_ws_public: str = "wss://fstream.binance.com/public"

    @property
    def symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols_csv.split(",") if s.strip()]

    @property
    def intervals(self) -> list[str]:
        return [i.strip() for i in self.intervals_csv.split(",") if i.strip()] if self.spot_enabled else []

    @property
    def perp_intervals(self) -> list[str]:
        return [i.strip() for i in self.perp_intervals_csv.split(",") if i.strip()] if self.futures_enabled else []

    @property
    def all_intervals(self) -> list[str]:
        return list(dict.fromkeys(self.intervals + self.perp_intervals))

    def candle_feeds(self, market: str | None = None) -> list["Feed"]:
        feeds = [Feed(s, s, "spot", i) for i in self.intervals for s in self.symbols]
        feeds += [Feed(perp_symbol(s), s, "perp", i) for i in self.perp_intervals for s in self.symbols]
        return [f for f in feeds if market is None or f.market == market]

    @property
    def backfill_start_dt(self) -> datetime:
        return datetime.fromisoformat(self.backfill_start).replace(tzinfo=timezone.utc)


@lru_cache
def get_settings() -> Settings:
    return Settings()
