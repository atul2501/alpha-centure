"""Idea 1 data + features: 1-minute order flow from Binance's 1m kline dumps, stored as Parquet (not Postgres).

    uv run python -m alpha.research.flow1m download      # 1m klines for the 23-coin pool -> data/research/flow1m/
    uv run python -m alpha.research.flow1m features      # 15-minute decision grid with flow + book features

A 1m kline carries the aggressive (taker) buy volume, so per minute: buy = taker_buy, sell = volume - taker_buy,
delta = buy - sell. That is the same aggregate the live aggTrade collector builds, without the 150 GB trade dumps.
Only the largest-single-trade size is missing. Files: one Parquet per symbol-month, resumable.

Features on a 15-minute grid, each known at the bar's close (only minutes that closed before it):
    flow_{5,15,60,240}   sum(delta) / sum(volume) over the last N minutes
    cvd_slope_60         change of cumulative delta over 60 min / volume, minus its 30-day mean
    buy_surprise_60      z of flow_60 against its own trailing 30 days
    vol_burst_15         volume of the last 15 min / mean 15-min volume over the trailing 7 days
    trades_burst_15      same for trade count
    book_imb             (bid - ask) / (bid + ask) notional within 1% of mid (book_depth_5m, 2023 on)
    book_imb_chg         change of book_imb over the last hour
Targets: forward log return over 1, 4 and 16 bars (15 min, 1 h, 4 h).
"""

import asyncio
import io
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from alpha.binance.vision import BASE, KLINE_COLS, listing_dates, months

ROOT = Path("data/research/flow1m")
POOL = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "TRXUSDT", "AAVEUSDT", "BNBUSDT", "XRPUSDT", "HYPEUSDT",
        "LINKUSDT", "ADAUSDT", "UNIUSDT", "LTCUSDT", "AVAXUSDT",
        "ATOMUSDT", "DOGEUSDT", "DOTUSDT", "NEARUSDT", "OPUSDT", "ARBUSDT", "WLDUSDT", "CAKEUSDT", "POLUSDT"]
START = date(2020, 1, 1)


def parse_1m(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        data = z.read(z.namelist()[0])
    first = data[:1].decode()
    df = pd.read_csv(io.BytesIO(data), header=None if first.isdigit() else 0, names=KLINE_COLS,
                     usecols=["open_time", "close", "volume", "count", "taker_buy_volume"])
    t = df["open_time"].astype("int64")
    t = t.where(t < 10**14, t // 1000)
    return pd.DataFrame({
        "time": pd.to_datetime(t, unit="ms", utc=True),
        "close": df["close"].astype("float64"), "volume": df["volume"].astype("float64"),
        "trades": df["count"].astype("int64"), "buy": df["taker_buy_volume"].astype("float64"),
    })


async def download(symbols: list[str] = POOL, workers: int = 6) -> None:
    today = datetime.now(timezone.utc).date()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=300), follow_redirects=True) as client:
        listed = await listing_dates(client)
        jobs = []
        for s in symbols:
            start = max(START, listed.get(s, START)).replace(day=1)
            for m in months(start, today.replace(day=1)):
                out = ROOT / s / f"{m}.parquet"
                if not out.exists() and not (ROOT / s / f"{m}.missing").exists():
                    jobs.append((s, m, out))
        logger.info("{} symbol-months to fetch", len(jobs))
        q: asyncio.Queue = asyncio.Queue()
        for j in jobs:
            q.put_nowait(j)
        done = {"n": 0}

        async def worker():
            while not q.empty():
                s, m, out = q.get_nowait()
                url = f"{BASE}/monthly/klines/{s}/1m/{s}-1m-{m}.zip"
                for attempt in range(4):
                    try:
                        r = await client.get(url)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if r.status_code == 404:
                            (ROOT / s / f"{m}.missing").touch()
                            break
                        r.raise_for_status()
                        df = await asyncio.to_thread(parse_1m, r.content)
                        df.to_parquet(out, index=False)
                        break
                    except Exception as e:
                        if attempt == 3:
                            logger.error("{} {} failed: {}", s, m, e)
                        await asyncio.sleep(2 ** attempt * 5)
                done["n"] += 1
                if done["n"] % 100 == 0:
                    logger.info("{} / {}", done["n"], len(jobs))

        await asyncio.gather(*(worker() for _ in range(workers)))
    logger.info("download done")


def load_1m(symbol: str) -> pd.DataFrame:
    files = sorted((ROOT / symbol).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("time").set_index("time").sort_index()
    return df


def features_15m(m1: pd.DataFrame, book: pd.DataFrame | None = None) -> pd.DataFrame:
    """15-minute grid features for one symbol (index = 15m bar open time; values known at bar close)."""
    m1 = m1.asfreq("1min")  # missing minutes -> NaN rows (exchange outages), never forward-filled into flow
    delta = 2 * m1["buy"] - m1["volume"]
    vol = m1["volume"]
    out = {}
    for n in (5, 15, 60, 240):
        out[f"flow_{n}"] = delta.rolling(n, min_periods=n // 2).sum() / vol.rolling(n, min_periods=n // 2).sum()
    cvd = delta.fillna(0).cumsum()
    slope = (cvd - cvd.shift(60)) / vol.rolling(60, min_periods=30).sum()
    out["cvd_slope_60"] = slope - slope.rolling(30 * 1440, min_periods=1440).mean()
    f60 = out["flow_60"]
    out["buy_surprise_60"] = (f60 - f60.rolling(30 * 1440, min_periods=1440).mean()) / \
        f60.rolling(30 * 1440, min_periods=1440).std()
    v15 = vol.rolling(15, min_periods=8).sum()
    out["vol_burst_15"] = v15 / v15.rolling(7 * 1440, min_periods=1440).mean()
    t15 = m1["trades"].rolling(15, min_periods=8).sum()
    out["trades_burst_15"] = t15 / t15.rolling(7 * 1440, min_periods=1440).mean()
    f = pd.DataFrame(out)
    # value at the last minute of each 15m bar = known at that bar's close
    g = f.resample("15min", label="left", closed="left").last()
    close = m1["close"].resample("15min", label="left", closed="left").last()
    g["close"] = close
    g["ret"] = np.log(close).diff()
    for h in (1, 4, 16):
        g[f"fwd_{h}"] = np.log(close.shift(-h) / close)
    if book is not None and not book.empty:
        b = book.set_index("snap_time").sort_index()
        imb = (b["bid_1"] - b["ask_1"]) / (b["bid_1"] + b["ask_1"])
        bar_close = g.index + pd.Timedelta(minutes=15)
        j = pd.merge_asof(pd.DataFrame({"t": bar_close}), imb.rename("book_imb").reset_index()
                          .rename(columns={"snap_time": "t"}), on="t", direction="backward",
                          tolerance=pd.Timedelta(minutes=10))
        g["book_imb"] = j["book_imb"].to_numpy()
        g["book_imb_chg"] = g["book_imb"] - g["book_imb"].shift(4)
    else:
        g["book_imb"] = np.nan
        g["book_imb_chg"] = np.nan
    return g


FEATS = ["flow_5", "flow_15", "flow_60", "flow_240", "cvd_slope_60", "buy_surprise_60", "burst_vol", "burst_trades",
         "book_imb", "book_imb_chg"]
FEAT_DIR = Path("data/research/flow15m")


def build_features(symbols: list[str] = POOL) -> None:
    """1m -> 15m features per symbol (one Parquet each; resumable)."""
    import psycopg

    from alpha.config import get_settings, perp_symbol

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(get_settings().database_url) as conn:
        for s in symbols:
            out = FEAT_DIR / f"{s}.parquet"
            if out.exists():
                continue
            m1 = load_1m(s)
            if m1.empty:
                continue
            cur = conn.execute("SELECT snap_time, bid_1, ask_1 FROM book_depth_5m WHERE symbol = %s ORDER BY snap_time",
                               (perp_symbol(s),))
            book = pd.DataFrame(cur.fetchall(), columns=["snap_time", "bid_1", "ask_1"])
            if not book.empty:
                book["snap_time"] = pd.to_datetime(book["snap_time"], utc=True)
                book[["bid_1", "ask_1"]] = book[["bid_1", "ask_1"]].astype(float)
            g = features_15m(m1, book)
            g["burst_vol"] = (g["vol_burst_15"] - 1) * np.sign(g["flow_15"])        # big volume, in flow's direction
            g["burst_trades"] = (g["trades_burst_15"] - 1) * np.sign(g["flow_15"])
            g.to_parquet(out)
            logger.info("features {}: {} bars", s, len(g))


def to_weights_bars(score: pd.DataFrame, vol: pd.DataFrame, el: pd.DataFrame, kind: str, every: int) -> pd.DataFrame:
    """Like portfolio_sim.to_weights but rebalancing every `every` bars of any grid (here 15 minutes)."""
    s = score.where(el).astype(float)
    raw = s / vol if kind == "ts" else s.sub(s.mean(axis=1), axis=0)
    raw = raw.where(el).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    on = (np.arange(len(w)) % every) == 0
    w = w.copy()
    w.iloc[~on] = np.nan
    return w.ffill().fillna(0.0).where(el, 0.0)


def screen(dev_end: pd.Timestamp, costs: pd.DataFrame, eligible_daily: pd.DataFrame) -> pd.DataFrame:
    """Idea 1 step a): gross edge per unit traded vs costs, per feature x TS/XS x horizon (15m, 1h, 4h), DEV only."""
    from alpha.research.portfolio_sim import newey_west_t, simulate

    frames = {s: pd.read_parquet(f) for s in POOL if (f := FEAT_DIR / f"{s}.parquet").exists()}
    W = lambda col: pd.DataFrame({s: g[col] for s, g in frames.items()}).sort_index()
    ret = W("ret")
    ret = ret[ret.index < dev_end]
    t = ret.index
    vol = ret.rolling(672, min_periods=96).std()
    el = eligible_daily.reindex(t.floor("D")).set_axis(t).reindex(columns=ret.columns).fillna(False).astype(bool)
    fund = pd.DataFrame(0.0, index=t, columns=ret.columns)  # holds of <= 4h: funding is second order, ignored
    per_side = (0.6 * costs["maker_bps"] + 0.4 * costs["taker_bps"]).reindex(ret.columns).fillna(costs["taker_bps"].max())
    rows = []
    for f in FEATS:
        sc = W(f).reindex(t)
        if sc.notna().sum().sum() == 0:
            continue
        rank = sc.where(el).rank(axis=1, pct=True)
        for kind, score in (("ts", sc.clip(-3, 3) if "flow" in f or "book" in f else sc.clip(-5, 5)),
                            ("xs", rank.sub(rank.mean(axis=1), axis=0))):
            for h in (1, 4, 16):
                w = to_weights_bars(score, vol, el & sc.notna(), kind, h)
                res = simulate(w, ret, fund, per_side)
                hr = res.hourly
                turn = hr["turnover"].sum()
                d = hr.groupby(hr.index.floor("D")).sum()
                yrs = d["net"].groupby(d.index.year).sum()
                edge = hr["gross"].sum() / turn * 1e4 if turn else np.nan
                cost = hr["cost"].sum() / turn * 1e4 if turn else np.nan
                t_nw = newey_west_t(d["net"], 5)
                rows.append({"feature": f, "kind": kind, "hold": {1: "15m", 4: "1h", 16: "4h"}[h],
                             "edge_bps": edge, "cost_bps": cost, "edge_to_cost": edge / cost if cost else np.nan,
                             "turnover_per_day": turn / max(1, len(d)), "net_ann": d["net"].mean() * 365,
                             "t_nw": t_nw, "years_pos": float((yrs > 0).mean()),
                             "passes": bool(edge >= 1.5 * cost and t_nw >= 2 and (yrs > 0).mean() >= 0.6),
                             "first_day": str(d.index.min().date())})
                logger.info("{} {} {}: edge {:.2f} vs cost {:.2f} bps", f, kind, h, edge, cost)
    return pd.DataFrame(rows)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    if cmd == "download":
        asyncio.run(download())
    elif cmd == "features":
        build_features()


if __name__ == "__main__":
    main()
