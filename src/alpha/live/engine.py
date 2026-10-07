"""Mainnet paper-trading engine: uv run python -m alpha.live.engine

Same strategy, risk and order logic that live trading would use; only order submission is simulated.
    data      mainnet websockets: depth20@100ms (/public) for the book, aggTrade + markPrice@1s (/market) for maker
              fills, marks and funding. Candles / funding / OI / premium come from the collector's tables.
    decide    hourly at HH:01:30 (after the collector has the closed bar and the 5m stats); trades only on the
              strategy's rebalance bars (every 72h, UTC-aligned) -> alpha.strategy.p6
    risk      caps (3x gross, 0.5x per coin), -30% drawdown kill switch (flatten + halt), -3%/day reduce-only
    execute   per coin: post at our side's best price for MAKER_TIMEOUT_S, remainder goes taker after the measured
              latency, walking the real book (alpha.exec.fillsim)
    ledger    paper_decisions / orders / fills / funding / equity (sql/006_paper.sql); the account is rebuilt from
              it on restart
There are no API keys and no order endpoints anywhere in this process. TRADING_MODE must be 'paper'.
"""

import asyncio
import sys
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
import psycopg
from loguru import logger

from alpha.binance.parse import ms_to_dt, now_ms
from alpha.binance.ws import combined_url, run_stream
from alpha.config import Settings, get_settings
from alpha.db import DB
from alpha.exec.account import Account
from alpha.exec.fillsim import Book, MakerOrder, taker_fill
from alpha.exec.oms import (DAILY_LOSS_STOP, KILL_DRAWDOWN, LotFilter, RiskState, margin_ratio, plan_orders,
                            round_step)
from alpha.strategy import p6
from alpha.strategy.monitor import MonitorState, evaluate, shadow_daily

MAKER_TIMEOUT_S = 20 * 60
BOOK_STALE_MS = 10_000
DECIDE_AFTER_CLOSE_S = 90
ORDER_ACK_MS = 30.0        # added to measured market-data latency for the simulated order round trip
RETRAIN_AFTER_DAYS = 30
EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"


async def lot_filters(symbols: list[str]) -> dict[str, LotFilter]:
    async with httpx.AsyncClient(timeout=30) as c:
        info = (await c.get(EXCHANGE_INFO)).json()
    out = {}
    for s in info["symbols"]:
        if s["symbol"] not in symbols:
            continue
        f = {x["filterType"]: x for x in s["filters"]}
        lot = f.get("MARKET_LOT_SIZE") or f["LOT_SIZE"]
        out[s["symbol"]] = LotFilter(float(lot["stepSize"]), float(lot["minQty"]),
                                     float(f.get("MIN_NOTIONAL", {}).get("notional", 5.0)))
    return out


class PaperEngine:
    def __init__(self, settings: Settings, db: DB):
        if settings.trading_mode != "paper":
            raise RuntimeError("alpha.live.engine only runs in TRADING_MODE=paper")
        self.s, self.db = settings, db
        self.symbols = settings.symbols
        self.books = {s: Book(s) for s in self.symbols}
        self.marks: dict[str, float] = {}
        self.funding_next: dict[str, tuple[int, float]] = {}   # symbol -> (next settlement ms, latest rate)
        self.makers: dict[int, tuple[MakerOrder, float]] = {}   # order id -> (order, arrival mid)
        self.account = Account(settings.paper_equity)
        self.risk = RiskState(peak_equity=settings.paper_equity)
        self.filters: dict[str, LotFilter] = {}
        self.bundle: p6.RidgeBundle | None = None
        self.latency_ms = 150.0
        self.busy = asyncio.Lock()
        self.monitor = MonitorState()

    # ---------------- restore ----------------

    async def restore(self) -> None:
        st = await self.db.pool.fetchval("SELECT value FROM paper_state WHERE key = 'account'")
        if st is None:
            await self._state("account", {"start_equity": self.s.paper_equity, "started_at": datetime.now(timezone.utc).isoformat()})
            st = {"start_equity": self.s.paper_equity}
        self.account = Account(float(st["start_equity"]))
        for r in await self.db.pool.fetch("SELECT symbol, side, qty, price, liquidity FROM paper_fills ORDER BY id"):
            self.account.apply_fill(r["symbol"], r["side"], r["qty"], r["price"], r["liquidity"])
        for r in await self.db.pool.fetch("SELECT amount FROM paper_funding ORDER BY id"):
            self.account.cash -= r["amount"]
            self.account.funding += r["amount"]
        self.monitor = MonitorState.from_json(await self.db.pool.fetchval(
            "SELECT value FROM paper_state WHERE key = 'monitor'"))
        rs = await self.db.pool.fetchval("SELECT value FROM paper_state WHERE key = 'risk'")
        if rs:
            self.risk = RiskState(peak_equity=rs["peak_equity"], halted=rs["halted"], reason=rs.get("reason", ""))
        await self.db.pool.execute("UPDATE paper_orders SET status = 'cancelled', closed_at = now() WHERE status = 'open'")
        logger.info("restored: cash {:.2f}, positions {}, halted {}", self.account.cash,
                    {k: round(v.qty, 6) for k, v in self.account.positions.items() if v.qty}, self.risk.halted)

    async def heartbeat(self, **extra) -> None:
        """Engine status for the dashboard's workflow page."""
        bar = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
        hours = (bar - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)
        next_bar = bar + pd.Timedelta(hours=(p6.H - hours % p6.H) % p6.H or p6.H)
        prev = await self.db.pool.fetchval("SELECT value FROM paper_state WHERE key = 'engine'") or {}
        await self._state("engine", {**prev, "model": p6.NAME, "rebalance_hours": p6.H,
                                     "model_train_end": str(self.bundle.train_end.date()) if self.bundle else None,
                                     "latency_ms": round(self.latency_ms, 1), "open_maker_orders": len(self.makers),
                                     "next_rebalance_decision": str(next_bar + pd.Timedelta(hours=1, seconds=DECIDE_AFTER_CLOSE_S)),
                                     "kill_drawdown": KILL_DRAWDOWN, "daily_loss_stop": DAILY_LOSS_STOP,
                                     "heartbeat": datetime.now(timezone.utc).isoformat(), **extra})

    async def _state(self, key: str, value: dict) -> None:
        await self.db.pool.execute("INSERT INTO paper_state (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE "
                                   "SET value = EXCLUDED.value", key, value)

    # ---------------- market data ----------------

    async def on_book(self, stream: str, d: dict) -> None:
        self.books[d["s"]].update(d["b"], d["a"], d["E"], now_ms())

    async def on_market(self, stream: str, d: dict) -> None:
        kind = stream.split("@", 2)[1].lower()
        sym = d["s"]
        if kind == "markprice":
            self.marks[sym] = float(d["p"])
            if d.get("T"):
                self.funding_next[sym] = (int(d["T"]), float(d["r"]))
            return
        if kind == "aggtrade" and self.makers:
            price, qty, bim = float(d["p"]), float(d["q"]), bool(d["m"])
            for oid, (o, mid) in list(self.makers.items()):
                if o.symbol != sym:
                    continue
                f = o.on_trade(price, qty, bim)
                if f:
                    await self._record_fill(oid, sym, o.side, f.price, f.qty, "maker", mid, None, False)
                    if o.remaining <= 1e-12:
                        await self._close_order(oid, "filled", o.filled)
                        self.makers.pop(oid, None)

    # ---------------- ledger ----------------

    async def _record_fill(self, oid: int, sym: str, side: int, price: float, qty: float, liq: str,
                           mid: float, latency: float | None, beyond: bool) -> None:
        fee = self.account.apply_fill(sym, side, qty, price, liq)
        slip = side * (price / mid - 1) * 1e4
        await self.db.pool.execute(
            "INSERT INTO paper_fills (order_id, symbol, side, price, qty, liquidity, fee, arrival_mid, slippage_bps, "
            "latency_ms, beyond_book) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            oid, sym, side, price, qty, liq, fee, mid, slip, latency, beyond)
        await self.db.pool.execute("UPDATE paper_orders SET filled = filled + $2 WHERE id = $1", oid, qty)
        logger.info("fill {} {} {} @ {} ({}, slip {:.2f} bps)", sym, "BUY" if side > 0 else "SELL", qty, price, liq, slip)

    async def _new_order(self, decision_id: int, sym: str, side: int, kind: str, price: float | None, qty: float,
                         mid: float, queue: float | None) -> int:
        return await self.db.pool.fetchval(
            "INSERT INTO paper_orders (decision_id, symbol, side, kind, price, qty, status, arrival_mid, queue_ahead) "
            "VALUES ($1,$2,$3,$4,$5,$6,'open',$7,$8) RETURNING id", decision_id, sym, side, kind, price, qty, mid, queue)

    async def _close_order(self, oid: int, status: str, filled: float) -> None:
        await self.db.pool.execute("UPDATE paper_orders SET status = $2, closed_at = now() WHERE id = $1", oid, status)

    async def _decision(self, bar_time, action: str, reason: str, targets: dict | None, current: dict | None) -> int:
        eq = self.account.equity(self.marks)
        return await self.db.pool.fetchval(
            "INSERT INTO paper_decisions (bar_time, model, action, reason, equity, targets, current) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id", bar_time, p6.NAME, action, reason, eq, targets, current)

    # ---------------- execution ----------------

    async def taker(self, decision_id: int, sym: str, side: int, qty: float, mid: float) -> None:
        await asyncio.sleep(self.latency_ms / 1000)   # decision -> exchange: book state after the delay
        book = self.books[sym]
        if not book.ok:
            logger.warning("no book for {}, taker skipped", sym)
            return
        f = taker_fill(book, side, qty)
        oid = await self._new_order(decision_id, sym, side, "taker", None, qty, mid, None)
        await self._record_fill(oid, sym, side, f.price, f.qty, "taker", mid, self.latency_ms, f.beyond_book)
        await self._close_order(oid, "filled", f.qty)

    async def execute(self, decision_id: int, orders, urgent: bool = False) -> None:
        mine = []
        for o in orders:
            book = self.books[o.symbol]
            mid = book.mid if book.ok else self.marks[o.symbol]
            if urgent or not book.ok:
                await self.taker(decision_id, o.symbol, o.side, o.qty, mid)
                continue
            price = book.bids[0][0] if o.side > 0 else book.asks[0][0]
            queue = book.queue_at(o.side, price)
            oid = await self._new_order(decision_id, o.symbol, o.side, "maker", price, o.qty, mid, queue)
            m = MakerOrder(oid, o.symbol, o.side, price, o.qty, queue, now_ms(), now_ms() + MAKER_TIMEOUT_S * 1000)
            self.makers[oid] = (m, mid)
            mine.append(oid)
        if not mine:
            return
        await asyncio.sleep(MAKER_TIMEOUT_S)
        for oid in mine:
            m, mid = self.makers.pop(oid, (None, None))
            if m is None:
                continue
            await self._close_order(oid, "expired" if m.filled == 0 else "partial_expired", m.filled)
            f = self.filters.get(m.symbol, LotFilter(0.0, 0.0, 5.0))
            rest = round_step(m.remaining, f.step)
            if rest > 0 and rest * self.marks.get(m.symbol, 0) >= f.min_notional:
                await self.taker(decision_id, m.symbol, m.side, rest, mid)

    # ---------------- decisions ----------------

    def _positions(self) -> dict[str, float]:
        return {s: p.qty for s, p in self.account.positions.items() if p.qty}

    def _weights(self) -> dict[str, float]:
        eq = self.account.equity(self.marks)
        return {s: round(v / eq, 5) for s, v in self.account.notional(self.marks).items()} if eq > 0 else {}

    async def hourly(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        bar = now.floor("h") - pd.Timedelta(hours=1)
        current = self._weights()
        if self.risk.halted:
            await self._decision(bar, "NO_TRADE", f"halted: {self.risk.reason}", None, current)
            return
        if self.s.monitor_enabled and not self.monitor.active:
            await self._decision(bar, "NO_TRADE", f"monitor {self.monitor.reason}", None, current)
            return
        if not p6.is_rebalance(bar):
            nxt = bar + pd.Timedelta(hours=p6.H - ((bar - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)) % p6.H)
            await self._decision(bar, "HOLD", f"not a rebalance bar (next {nxt:%Y-%m-%d %H:%M} UTC)", None, current)
            return
        stale = [s for s, b in self.books.items() if not b.ok or now_ms() - b.recv_ms > BOOK_STALE_MS]
        if len(stale) > len(self.symbols) // 2:
            await self._decision(bar, "NO_TRADE", f"stale books: {stale}", None, current)
            return
        try:
            last, w, _ = await asyncio.to_thread(self._targets, now)
        except Exception as e:
            logger.exception("target computation failed")
            await self._decision(bar, "NO_TRADE", f"strategy error: {e}", None, current)
            return
        if last != bar:
            await self._decision(bar, "NO_TRADE", f"data not ready: last closed bar {last}", None, current)
            return
        targets = {s: round(float(v), 5) for s, v in w.items() if abs(v) > 1e-6}
        eq = self.account.equity(self.marks)
        orders = plan_orders(targets, self._positions(), self.marks, eq, self.filters, p6.BAND,
                             reduce_only=self.risk.daily_stop)
        reason = f"{len(orders)} orders" + (" (daily loss stop: reduce only)" if self.risk.daily_stop else "")
        did = await self._decision(bar, "REBALANCE", reason, targets, current)
        logger.info("rebalance {}: {} orders, targets {}", bar, len(orders), targets)
        asyncio.create_task(self.execute(did, orders))

    def _targets(self, now: pd.Timestamp):
        with psycopg.connect(self.s.database_url) as conn:
            return p6.live_targets(conn, self.symbols, self.bundle, now)

    async def ensure_model(self) -> None:
        b = p6.RidgeBundle.latest(self.s.models_dir)
        month = pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
        if b is None or (month - b.train_end).days >= RETRAIN_AFTER_DAYS - 1:
            logger.info("training ridge on data up to {}", month)

            def _train():
                with psycopg.connect(self.s.database_url) as conn:
                    nb = p6.train(conn, self.symbols, month)
                nb.save(self.s.models_dir)
                return nb
            b = await asyncio.to_thread(_train)
        self.bundle = b
        logger.info("model trained up to {}", b.train_end)

    async def minutely(self) -> None:
        if not self.marks:
            return
        t = now_ms()
        for sym, (T, rate) in list(self.funding_next.items()):
            if t >= T and sym in self.marks:
                qty = self.account.pos(sym).qty
                if qty:
                    inserted = await self.db.pool.fetchval(
                        "INSERT INTO paper_funding (ts, symbol, qty, mark, rate, amount) VALUES ($1,$2,$3,$4,$5,$6) "
                        "ON CONFLICT (symbol, ts) DO NOTHING RETURNING id",
                        ms_to_dt(T), sym, qty, self.marks[sym], rate, qty * self.marks[sym] * rate)
                    if inserted:
                        self.account.apply_funding(sym, self.marks[sym], rate)
                self.funding_next[sym] = (T + 8 * 3600_000, rate)  # replaced by the stream's next value
        eq = self.account.equity(self.marks)
        self.risk.observe(eq, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        notional = self.account.notional(self.marks)
        mr = margin_ratio(notional, eq)
        await self.db.pool.execute(
            "INSERT INTO paper_equity (ts, equity, cash, unrealized, gross_lev, net_lev, margin_ratio, fees, funding, "
            "positions) VALUES (date_trunc('minute', now()),$1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT (ts) DO NOTHING",
            eq, self.account.cash, self.account.unrealized(self.marks), self.account.gross_leverage(self.marks),
            sum(notional.values()) / eq if eq > 0 else 0.0, mr, self.account.fees, self.account.funding,
            {s: {"qty": p.qty, "entry": p.entry, "mark": self.marks.get(s)} for s, p in self.account.positions.items() if p.qty})
        await self._state("risk", {"peak_equity": self.risk.peak_equity, "halted": self.risk.halted,
                                   "reason": self.risk.reason, "day_start_equity": self.risk.day_start_equity,
                                   "daily_stop": self.risk.daily_stop})
        await self.heartbeat()
        if self.risk.halted and self._positions() and not self.busy.locked():
            async with self.busy:
                logger.error("KILL SWITCH: {} -> flattening", self.risk.reason)
                did = await self._decision(pd.Timestamp.now(tz="UTC").floor("h"), "FLATTEN", self.risk.reason, {}, self._weights())
                orders = plan_orders({}, self._positions(), self.marks, eq, self.filters, 0.0)
                await self.execute(did, orders, urgent=True)
        elif mr > 0.5:
            logger.warning("margin ratio {:.2f}", mr)

    async def check_monitor(self) -> None:
        """Daily: replay the strategy's shadow P&L and pause / resume on the monitor's rules."""
        now = pd.Timestamp.now(tz="UTC")

        def _shadow():
            with psycopg.connect(self.s.database_url) as conn:
                return shadow_daily(conn, self.symbols, self.s.models_dir, now)
        daily = await asyncio.to_thread(_shadow)
        before = self.monitor.active
        self.monitor = evaluate(daily, self.monitor, now.normalize())
        await self._state("monitor", self.monitor.to_json())
        logger.info("monitor: {}", self.monitor.reason)
        if before and not self.monitor.active:
            async with self.busy:
                did = await self._decision(now.floor("h"), "PAUSE", self.monitor.reason, {}, self._weights())
                orders = plan_orders({}, self._positions(), self.marks, self.account.equity(self.marks),
                                     self.filters, 0.0)
                await self.execute(did, orders)
        elif not before and self.monitor.active:
            await self._decision(now.floor("h"), "RESUME", self.monitor.reason + " (trades from the next rebalance bar)",
                                 None, self._weights())

    async def _every_day(self) -> None:
        while True:
            try:
                await self.check_monitor()
            except Exception:
                logger.exception("monitor check failed")
            now = time.time()
            await asyncio.sleep(86400 - now % 86400 + 600)  # 00:10 UTC

    async def measure_latency(self) -> None:
        p95 = await self.db.pool.fetchval("SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY p95_ms) FROM ws_latency "
                                          "WHERE stream = 'depth20' AND minute > now() - interval '1 hour'")
        if p95:
            self.latency_ms = float(p95) + ORDER_ACK_MS

    # ---------------- run ----------------

    async def run(self) -> None:
        await self.restore()
        self.filters = await lot_filters(self.symbols)
        await self.ensure_model()
        await self.measure_latency()
        await self.heartbeat(started_at=datetime.now(timezone.utc).isoformat())
        book_streams = [f"{s.lower()}@depth20@100ms" for s in self.symbols]
        mkt_streams = [f"{s.lower()}@{k}" for s in self.symbols for k in ("aggTrade", "markPrice@1s")]
        tasks = [
            asyncio.create_task(run_stream("paper_book", combined_url(self.s.futures_ws_public, book_streams), self.on_book)),
            asyncio.create_task(run_stream("paper_market", combined_url(self.s.futures_ws, mkt_streams), self.on_market)),
            asyncio.create_task(self._every_minute()),
            asyncio.create_task(self._every_hour()),
        ]
        if self.s.monitor_enabled:
            tasks.append(asyncio.create_task(self._every_day()))
        else:
            await self._state("monitor", {"active": True, "reason": "disabled (MONITOR_ENABLED=false): always on, "
                                          "-30% kill switch only", "checked": None})
        await asyncio.gather(*tasks)

    async def _every_minute(self) -> None:
        while True:
            await asyncio.sleep(60 - time.time() % 60 + 1)
            try:
                await self.minutely()
            except Exception:
                logger.exception("minutely failed")

    async def _every_hour(self) -> None:
        while True:
            now = time.time()
            await asyncio.sleep(3600 - now % 3600 + DECIDE_AFTER_CLOSE_S)
            try:
                await self.measure_latency()
                await self.ensure_model()
                await self.hourly()
                await self.heartbeat(last_hourly=datetime.now(timezone.utc).isoformat())
            except Exception:
                logger.exception("hourly failed")


async def main() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    s = get_settings()
    db = await DB.connect(s.database_url)
    await db.init_schema()
    try:
        await PaperEngine(s, db).run()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
