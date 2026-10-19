"""Alpha Centure dashboard: a small JSON API + one static page. Read-only, except the Coins tab's requests (check a
coin, start its shadow test, promote it), which need DASHBOARD_PASSWORD set.

    uv run python -m dashboard.server            # http://127.0.0.1:8501

Settings (environment or .env):
    DASHBOARD_HOST      bind address (default 127.0.0.1; 0.0.0.0 to open it to the network)
    DASHBOARD_PORT      default 8501
    DASHBOARD_PASSWORD  optional: when set, every request needs HTTP basic auth (any user name, this password).
                        Without it a public dashboard is protected only by the security group (allow your IP only).

Results are cached on the server so any number of open browsers share one query: workflow 1 s, ledger 5 s,
market 10 s, and the expensive data-audit scans 5 minutes.
"""

import asyncio
import base64
import secrets
import threading
import time
from pathlib import Path

import psycopg
import uvicorn
from loguru import logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from alpha import coins
from alpha.config import get_settings
from dashboard import data

STATIC = Path(__file__).parent / "static"
TTL = {"coins": 5.0, "workflow": 1.0, "ledger": 5.0, "market": 10.0, "market_heavy": 300.0, "candles": 10.0}
BARS = (100, 200, 500, 1000)

_settings = get_settings()
_lock = threading.Lock()
_conn: psycopg.Connection | None = None
_cache: dict[tuple, tuple[float, object, float]] = {}


def _db() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed or _conn.broken:
        _conn = psycopg.connect(_settings.database_url, autocommit=True, **_settings.ledger_conn)
    return _conn


def _fresh(key: tuple) -> bool:
    hit = _cache.get(key)
    # never recompute sooner than 2x the last computation took: a slow query can't keep the database busy
    return bool(hit) and time.monotonic() - hit[0] < max(TTL[key[0]], 2 * hit[2])


def cached(key: tuple, fn):
    """One computation per TTL, shared by every viewer; queries run one at a time on one connection."""
    if _fresh(key):
        return _cache[key][1]
    with _lock:
        if _fresh(key):
            return _cache[key][1]
        t0 = time.monotonic()
        try:
            value = fn(_db())
        except psycopg.OperationalError:
            global _conn
            _conn = None
            value = fn(_db())
        _cache[key] = (time.monotonic(), value, time.monotonic() - t0)
        return value


# Live view: one LISTEN connection for the whole server; every open browser gets the latest engine snapshot pushed
# (Server-Sent Events) as soon as it arrives, about once a second.
_live: dict = {"v": 0, "data": None}
_listener: threading.Thread | None = None


def _listen() -> None:
    while True:
        try:
            with psycopg.connect(_settings.database_url, autocommit=True, **_settings.ledger_conn) as c:
                c.execute("LISTEN paper_live")
                for n in c.notifies():
                    _live["data"], _live["v"] = n.payload, _live["v"] + 1
        except Exception as e:
            logger.warning("live listener: {}; reconnecting", e)
            time.sleep(5)


def _ensure_listener() -> None:
    global _listener
    with _lock:
        if _listener is None or not _listener.is_alive():
            _listener = threading.Thread(target=_listen, name="live-listen", daemon=True)
            _listener.start()


async def api_live(request: Request):
    _ensure_listener()

    async def stream():
        seen, quiet = -1, 0.0
        yield "retry: 3000\n\n"
        while not await request.is_disconnected():
            if _live["v"] != seen and _live["data"]:
                seen, quiet = _live["v"], 0.0
                yield f"data: {_live['data']}\n\n"
            elif quiet > 15:  # keep proxies from closing an idle stream
                quiet = 0.0
                yield ": ping\n\n"
            await asyncio.sleep(0.2)
            quiet += 0.2

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def check_bind(host: str, password: str | None) -> None:
    if host not in ("127.0.0.1", "localhost", "::1") and not password:
        logger.warning("dashboard on {} without a password: anyone who can reach the port can see it; "
                       "limit port access to your IP in the security group", host)


class BasicAuth(BaseHTTPMiddleware):
    def __init__(self, app, password: str | None):
        super().__init__(app)
        self.password = password

    async def dispatch(self, request: Request, call_next):
        if self.password:
            auth = request.headers.get("authorization", "")
            ok = False
            if auth.lower().startswith("basic "):
                try:
                    _, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                    ok = secrets.compare_digest(pw, self.password)
                except (ValueError, UnicodeDecodeError):
                    ok = False
            if not ok:
                return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Alpha Centure"'})
        return await call_next(request)


def api_workflow(request: Request):
    return JSONResponse(cached(("workflow",), lambda c: data.workflow(c, _settings)))


def api_ledger(request: Request):
    try:
        return JSONResponse(cached(("ledger",), data.ledger))
    except Exception as e:  # show the cause on the page instead of a bare 500
        logger.exception("ledger failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, 500)


def api_market(request: Request):
    light = cached(("market",), lambda c: data.market_light(c, _settings))
    heavy = cached(("market_heavy",), lambda c: data.market_heavy(c, _settings))
    return JSONResponse({**light, "heavy": heavy})


def api_candles(request: Request):
    symbols = {f.db_symbol for f in _settings.candle_feeds()}
    symbol, interval = request.query_params.get("symbol", ""), request.query_params.get("interval", "")
    try:
        bars = int(request.query_params.get("bars", "200"))
    except ValueError:
        bars = -1
    if symbol not in symbols or interval not in _settings.all_intervals or bars not in BARS:
        return JSONResponse({"error": "unknown symbol, interval or bar count"}, 400)
    return JSONResponse(cached(("candles", symbol, interval, bars), lambda c: data.candles(c, symbol, interval, bars)))


def api_coins(request: Request):
    return JSONResponse(cached(("coins",), lambda c: {"coins": coins.listing(c), "base": _settings.symbols,
                                                       "shadow_min_days": coins.SHADOW_MIN_DAYS,
                                                       "purge_days": coins.PURGE_AFTER_DAYS,
                                                       "writable": bool(_settings.dashboard_password)}))


async def api_coins_action(request: Request):
    """POST {"action": "check" | "shadow" | "live" | "rejected", "symbol": "INJUSDT"}."""
    if not _settings.dashboard_password:
        return JSONResponse({"error": "set DASHBOARD_PASSWORD to allow changes from the dashboard"}, 403)
    if request.headers.get("content-type", "").split(";")[0] != "application/json":  # no cross-site form posts
        return JSONResponse({"error": "JSON only"}, 415)
    try:
        body = await request.json()
        action, symbol = str(body.get("action", "")), str(body.get("symbol", "")).strip().upper()
        with _lock:
            if action == "check":
                symbol = coins.request_check(_db(), symbol, coins.trading_symbols(_db(), _settings.symbols))
            else:
                coins.move(_db(), symbol, action)
            _cache.pop(("coins",), None)
    except (ValueError, AttributeError) as e:
        return JSONResponse({"error": str(e)}, 400)
    logger.info("coins: {} {}", action, symbol)
    return JSONResponse({"ok": True, "symbol": symbol})


def index(request: Request):
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


def create_app(password: str | None = None) -> Starlette:
    return Starlette(routes=[
        Route("/", index),
        Route("/api/workflow", api_workflow),
        Route("/api/ledger", api_ledger),
        Route("/api/market", api_market),
        Route("/api/candles", api_candles),
        Route("/api/coins", api_coins),
        Route("/api/live", api_live),
        Route("/api/coins/action", api_coins_action, methods=["POST"]),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ], middleware=[Middleware(BasicAuth, password=password)])


def main() -> None:
    host, password = _settings.dashboard_host, _settings.dashboard_password or None
    check_bind(host, password)
    uvicorn.run(create_app(password), host=host, port=_settings.dashboard_port, log_level="warning")


if __name__ == "__main__":
    main()
