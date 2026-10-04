from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PERP_SUFFIX = ".P"


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
    symbols_csv: str = Field("BTCUSDT,ETHUSDT,SOLUSDT,SUIUSDT", validation_alias=AliasChoices("SYMBOLS", "symbols_csv"))
    intervals_csv: str = Field("1m,5m,15m,1h,4h,1d,1w", validation_alias=AliasChoices("INTERVALS", "intervals_csv"))
    # USD-M perpetual candles (what long/short trades execute on). Stored as e.g. BTCUSDT.P
    perp_intervals_csv: str = Field("5m,15m,1h,4h", validation_alias=AliasChoices("PERP_INTERVALS", "perp_intervals_csv"))
    backfill_1m_days: int = 365
    backfill_start: str = "2017-08-01"
    futures_enabled: bool = True
    orderflow_enabled: bool = True
    models_dir: str = "models"
    # experiment config used by alpha.trainer (see alpha.research.experiment): breakeven exits, calibrated per-setup EV,
    # cost gate, 3-window threshold selection, 2-year rolling training, EV sizing + exposure cap + daily loss stop
    production_config: str = "roll730_risk"

    spot_rest: str = "https://api.binance.com"
    futures_rest: str = "https://fapi.binance.com"
    spot_ws: str = "wss://stream.binance.com:9443"
    # Binance routes USD-M market streams under /market; the bare host accepts connections but sends nothing.
    futures_ws: str = "wss://fstream.binance.com/market"

    @property
    def symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols_csv.split(",") if s.strip()]

    @property
    def intervals(self) -> list[str]:
        return [i.strip() for i in self.intervals_csv.split(",") if i.strip()]

    @property
    def perp_intervals(self) -> list[str]:
        return [i.strip() for i in self.perp_intervals_csv.split(",") if i.strip()] if self.futures_enabled else []

    @property
    def all_intervals(self) -> list[str]:
        return list(dict.fromkeys(self.intervals + self.perp_intervals))

    def candle_feeds(self, market: str | None = None) -> list["Feed"]:
        feeds = [Feed(s, s, "spot", i) for i in self.intervals for s in self.symbols]
        feeds += [Feed(s + PERP_SUFFIX, s, "perp", i) for i in self.perp_intervals for s in self.symbols]
        return [f for f in feeds if market is None or f.market == market]

    @property
    def backfill_start_dt(self) -> datetime:
        return datetime.fromisoformat(self.backfill_start).replace(tzinfo=timezone.utc)


@lru_cache
def get_settings() -> Settings:
    return Settings()
