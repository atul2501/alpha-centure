"""History backfill from Binance's public dumps (data.binance.vision), USD-M perpetuals.

    uv run python -m alpha.binance.vision                       # klines, metrics, premium, bookDepth since 2020-01
    uv run python -m alpha.binance.vision --datasets aggTrades  # 1m flow from trades (very large downloads)
    uv run python -m alpha.binance.vision --report              # coverage per dataset x symbol

Resumable: every processed file is recorded in vision_files and skipped next time. Monthly files cover complete
months; the current month is filled from daily files. Months before a symbol's listing are skipped.

Dataset -> table:
    klines     candles (BTCUSDT.P, source 'vision'), one series per --intervals entry
    premium    premium_kline (plain symbol)
    metrics    futures_metrics (5m OI, long/short ratios, taker ratio; from ~Dec 2021)
    bookDepth  book_depth_5m (+-1..5% notional, last snapshot per 5 minutes; from ~2023)
    aggTrades  flow_1m (BTCUSDT.P): same definitions as the live aggTrade aggregator
"""

import argparse
import asyncio
import io
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from alpha.config import get_settings, perp_symbol
from alpha.db import DB

BASE = "https://data.binance.vision/data/futures/um"
EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
AGG_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
AGG_CHUNK = 2_000_000
MISSING_AFTER_DAYS = 3  # a 404 for a period older than this is final (recorded); newer ones are retried


@dataclass(frozen=True)
class Job:
    dataset: str      # klines | premium | metrics | bookDepth | aggTrades
    symbol: str       # plain, e.g. BTCUSDT
    period: str       # YYYY-MM (monthly file) or YYYY-MM-DD (daily file)
    interval: str = ""

    @property
    def daily(self) -> bool:
        return len(self.period) == 10

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.interval}" if self.interval else self.dataset

    @property
    def path(self) -> str:
        freq = "daily" if self.daily else "monthly"
        s, p, iv = self.symbol, self.period, self.interval
        return {
            "klines": f"{freq}/klines/{s}/{iv}/{s}-{iv}-{p}.zip",
            "premium": f"{freq}/premiumIndexKlines/{s}/{iv}/{s}-{iv}-{p}.zip",
            "metrics": f"daily/metrics/{s}/{s}-metrics-{p}.zip",
            "bookDepth": f"daily/bookDepth/{s}/{s}-bookDepth-{p}.zip",
            "aggTrades": f"{freq}/aggTrades/{s}/{s}-aggTrades-{p}.zip",
        }[self.dataset]

    @property
    def period_end(self) -> date:
        if self.daily:
            return date.fromisoformat(self.period)
        y, m = map(int, self.period.split("-"))
        return (date(y + m // 12, m % 12 + 1, 1)) - timedelta(days=1)


# ---------------------------------------------------------------------------------------------------------------
# pure parsers: CSV text -> DB rows (unit tested)

def _read_csv(f, names: list[str], **kw) -> pd.DataFrame:
    """Dumps have a header row in newer files and none in older ones."""
    head = f.read(64)
    f.seek(0)
    first = head.decode() if isinstance(head, bytes) else head
    has_header = not first[:1].isdigit()
    return pd.read_csv(f, header=0 if has_header else None, names=names, **kw)


def _ms(s: pd.Series) -> pd.Series:
    """Epoch -> ms; some newer dumps use microseconds."""
    v = s.astype("int64")
    return v.where(v < 10**14, v // 1000)


def _times(ms: pd.Series) -> list[datetime]:
    return list(pd.to_datetime(_ms(ms), unit="ms", utc=True).dt.to_pydatetime())


def parse_klines(f, db_symbol: str, interval: str) -> list[tuple]:
    df = _read_csv(f, KLINE_COLS, dtype=str)
    num = ["open", "high", "low", "close", "volume", "quote_volume"]
    cols = [df[c].str.strip() for c in num]
    taker = [df["taker_buy_volume"].str.strip(), df["taker_buy_quote_volume"].str.strip()]
    return list(zip([db_symbol] * len(df), [interval] * len(df), _times(df["open_time"]), _times(df["close_time"]),
                    *cols[:4], cols[4], cols[5], df["count"].astype(int).tolist(), *taker, ["vision"] * len(df)))


def parse_premium(f, symbol: str, interval: str) -> list[tuple]:
    df = _read_csv(f, KLINE_COLS)
    return list(zip([symbol] * len(df), [interval] * len(df), _times(df["open_time"]), _times(df["close_time"]),
                    df["open"].astype(float), df["high"].astype(float), df["low"].astype(float),
                    df["close"].astype(float)))


def parse_metrics(f, symbol: str) -> list[tuple]:
    df = pd.read_csv(f)
    ts = list(pd.to_datetime(df["create_time"], utc=True).dt.to_pydatetime())
    num = lambda c: [None if pd.isna(v) else float(v) for v in df[c]] if c in df else [None] * len(df)
    return list(zip([symbol] * len(df), ts, num("sum_open_interest"), num("sum_open_interest_value"),
                    num("count_toptrader_long_short_ratio"), num("sum_toptrader_long_short_ratio"),
                    num("count_long_short_ratio"), num("sum_taker_long_short_vol_ratio")))


def parse_book_depth(f, db_symbol: str) -> list[tuple]:
    df = pd.read_csv(f)
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
    df = df[df["percentage"].isin([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])]
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["bucket"] = df["timestamp"].dt.floor("5min")
    last = df.groupby("bucket")["timestamp"].transform("max")
    snap = df[df["timestamp"] == last].pivot_table(index=["bucket", "timestamp"], columns="percentage",
                                                   values="notional", aggfunc="last")
    snap = snap.reindex(columns=[-1, -2, -3, -4, -5, 1, 2, 3, 4, 5])
    out = []
    for (bucket, t), r in snap.iterrows():
        vals = [None if pd.isna(v) else float(v) for v in r.to_numpy()]
        out.append((db_symbol, bucket.to_pydatetime(), t.to_pydatetime(), *vals))
    return out


def agg_trades_to_flow(chunks, db_symbol: str) -> list[tuple]:
    """aggTrade chunks -> complete 1m flow rows (buyer_is_maker => aggressive sell, as in FlowBucket)."""
    parts = []
    for df in chunks:
        t = _ms(df["transact_time"])
        maker = df["is_buyer_maker"].astype(str).str.lower().eq("true").to_numpy()
        qty = df["quantity"].astype(float).to_numpy()
        quote = df["price"].astype(float).to_numpy() * qty
        g = pd.DataFrame({
            "minute": (t - t % 60_000).to_numpy(),
            "buy_vol": np.where(maker, 0.0, qty), "sell_vol": np.where(maker, qty, 0.0),
            "buy_quote": np.where(maker, 0.0, quote), "sell_quote": np.where(maker, quote, 0.0),
            "trades": 1, "max_q": quote,
        }).groupby("minute")
        parts.append(g.agg({"buy_vol": "sum", "sell_vol": "sum", "buy_quote": "sum", "sell_quote": "sum",
                            "trades": "sum", "max_q": "max"}))
    if not parts:
        return []
    m = pd.concat(parts).groupby(level=0).agg({"buy_vol": "sum", "sell_vol": "sum", "buy_quote": "sum",
                                               "sell_quote": "sum", "trades": "sum", "max_q": "max"})
    minutes = _times(pd.Series(m.index))
    return [(db_symbol, mt, r.buy_vol, r.sell_vol, r.buy_vol - r.sell_vol, r.buy_quote, r.sell_quote, int(r.trades),
             r.max_q, True) for mt, r in zip(minutes, m.itertuples())]


# ---------------------------------------------------------------------------------------------------------------
# planning + I/O

def months(start: date, end_exclusive: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while date(y, m, 1) < end_exclusive:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def days(start: date, end_inclusive: date) -> list[str]:
    n = (end_inclusive - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(max(0, n + 1))]


def plan(symbols: list[str], datasets: list[str], intervals: list[str], since: date, listed: dict[str, date],
         today: date) -> list[Job]:
    this_month = today.replace(day=1)
    yesterday = today - timedelta(days=1)
    jobs = []
    for sym in symbols:
        start = max(since, listed.get(sym, since))
        for ds in datasets:
            ivs = intervals if ds == "klines" else (["5m", "1h"] if ds == "premium" else [""])
            for iv in ivs:
                if ds in ("metrics", "bookDepth"):
                    periods = days(start, yesterday)
                else:
                    periods = months(start.replace(day=1), this_month) + days(max(start, this_month), yesterday)
                jobs += [Job(ds, sym, p, iv) for p in periods]
    return jobs


async def listing_dates(client: httpx.AsyncClient) -> dict[str, date]:
    r = await client.get(EXCHANGE_INFO)
    r.raise_for_status()
    return {s["symbol"]: datetime.fromtimestamp(s["onboardDate"] / 1000, timezone.utc).date()
            for s in r.json()["symbols"] if s.get("contractType") == "PERPETUAL"}


def process(job: Job, zpath: Path) -> tuple[str, list[tuple]]:
    """Unzip + parse one file (runs in a worker thread). Returns (table, rows)."""
    with zipfile.ZipFile(zpath) as z:
        name = z.namelist()[0]
        if job.dataset == "aggTrades":
            with z.open(name) as raw:
                f = io.BufferedReader(raw)
                head = f.peek(64)[:64].decode(errors="ignore")
                header = 0 if not head[:1].isdigit() else None
                chunks = pd.read_csv(f, header=header, names=AGG_COLS, usecols=[1, 2, 5, 6], chunksize=AGG_CHUNK)
                return "flow_1m", agg_trades_to_flow(chunks, perp_symbol(job.symbol))
        data = io.BytesIO(z.read(name))
    if job.dataset == "klines":
        return "candles", parse_klines(data, perp_symbol(job.symbol), job.interval)
    if job.dataset == "premium":
        return "premium_kline", parse_premium(data, job.symbol, job.interval)
    if job.dataset == "metrics":
        return "futures_metrics", parse_metrics(data, job.symbol)
    return "book_depth_5m", parse_book_depth(data, perp_symbol(job.symbol))


async def run_job(db: DB, client: httpx.AsyncClient, job: Job, tmp: Path, today: date) -> int:
    url = f"{BASE}/{job.path}"
    zpath = tmp / job.path.replace("/", "_")
    async with client.stream("GET", url) as r:
        if r.status_code == 404:
            if (today - job.period_end).days > MISSING_AFTER_DAYS:
                await db.upsert("vision_files", [(job.path, job.key, job.symbol, job.period, -1)])
            return 0
        r.raise_for_status()
        with open(zpath, "wb") as out:
            async for chunk in r.aiter_bytes(1 << 20):
                out.write(chunk)
    try:
        table, rows = await asyncio.to_thread(process, job, zpath)
    finally:
        zpath.unlink(missing_ok=True)
    for i in range(0, len(rows), 200_000):
        await db.bulk_upsert(table, rows[i:i + 200_000])
    await db.upsert("vision_files", [(job.path, job.key, job.symbol, job.period, len(rows))])
    return len(rows)


async def backfill(symbols: list[str], datasets: list[str], intervals: list[str], since: date, workers: int) -> None:
    settings = get_settings()
    db = await DB.connect(settings.database_url)
    await db.init_schema()
    today = datetime.now(timezone.utc).date()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=300), follow_redirects=True) as client:
        listed = await listing_dates(client)
        unknown = [s for s in symbols if s not in listed]
        if unknown:
            logger.warning("not USD-M perpetuals, skipped: {}", unknown)
        jobs = plan([s for s in symbols if s in listed], datasets, intervals, since, listed, today)
        done = {r["path"] for r in await db.pool.fetch("SELECT path FROM vision_files")}
        todo = [j for j in jobs if j.path not in done]
        # small files first so features become usable early; aggTrades last
        todo.sort(key=lambda j: (j.dataset == "aggTrades", j.daily is False, j.period))
        logger.info("{} files planned, {} already done, {} to fetch", len(jobs), len(jobs) - len(todo), len(todo))
        queue: asyncio.Queue[Job] = asyncio.Queue()
        for j in todo:
            queue.put_nowait(j)
        stats = {"files": 0, "rows": 0, "errors": 0}

        async def worker(tmp: Path) -> None:
            while not queue.empty():
                job = queue.get_nowait()
                for attempt in range(3):
                    try:
                        n = await run_job(db, client, job, tmp, today)  # await first: workers interleave
                        stats["rows"] += n
                        stats["files"] += 1
                        break
                    except Exception as e:
                        if attempt == 2:
                            stats["errors"] += 1
                            logger.error("{} failed: {}", job.path, e)
                        await asyncio.sleep(2 ** attempt)
                if stats["files"] % 200 == 0:
                    logger.info("{} / {} files, {} rows, {} errors", stats["files"], len(todo), stats["rows"],
                                stats["errors"])

        with tempfile.TemporaryDirectory(prefix="vision_") as tmp:
            await asyncio.gather(*(worker(Path(tmp)) for _ in range(workers)))
        logger.info("done: {} files, {} rows, {} errors", stats["files"], stats["rows"], stats["errors"])
    await db.close()


COVERAGE_SQL = """
SELECT dataset, symbol, min(period) FILTER (WHERE rows > 0) AS first, max(period) FILTER (WHERE rows > 0) AS last,
       count(*) FILTER (WHERE rows > 0) AS files, count(*) FILTER (WHERE rows < 0) AS missing, sum(greatest(rows, 0)) AS rows
FROM vision_files GROUP BY 1, 2 ORDER BY 1, 2
"""


async def report() -> None:
    db = await DB.connect(get_settings().database_url)
    rows = await db.pool.fetch(COVERAGE_SQL)
    await db.close()
    df = pd.DataFrame([dict(r) for r in rows])
    print(df.to_string(index=False) if not df.empty else "nothing ingested yet")


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="klines,premium,metrics,bookDepth")
    ap.add_argument("--symbols", default=",".join(s.symbols))
    ap.add_argument("--intervals", default=",".join(s.perp_intervals))
    ap.add_argument("--since", default="2020-01-01")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level: <7} | {message}")
    if a.report:
        asyncio.run(report())
        return
    asyncio.run(backfill([x.strip().upper() for x in a.symbols.split(",") if x.strip()],
                         [x.strip() for x in a.datasets.split(",") if x.strip()],
                         [x.strip() for x in a.intervals.split(",") if x.strip()],
                         date.fromisoformat(a.since), a.workers))


if __name__ == "__main__":
    main()
