"""Alpha Centure dashboard: a small JSON API + one static page. Read-only.

    uv run python -m dashboard.server            # http://127.0.0.1:8501

Environment:
    DASHBOARD_HOST      bind address (default 127.0.0.1; use 0.0.0.0 only together with a password)
    DASHBOARD_PORT      default 8501
    DASHBOARD_PASSWORD  when set, every request needs HTTP basic auth (any user name, this password)

Results are cached on the server so any number of open browsers share one query: workflow 1 s, ledger 5 s,
market 10 s, and the expensive data-audit scans 5 minutes.
"""

import base64
import os
import secrets
import threading
import time
from pathlib import Path

import psycopg
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from alpha.config import get_settings
from dashboard import data

STATIC = Path(__file__).parent / "static"
TTL = {"workflow": 1.0, "ledger": 5.0, "market": 10.0, "market_heavy": 300.0, "candles": 10.0}
BARS = (100, 200, 500, 1000)

_settings = get_settings()
_lock = threading.Lock()
_conn: psycopg.Connection | None = None
_cache: dict[tuple, tuple[float, object, float]] = {}


def _db() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed or _conn.broken:
        _conn = psycopg.connect(_settings.database_url, autocommit=True)
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


def check_bind(host: str, password: str | None) -> None:
    if host not in ("127.0.0.1", "localhost", "::1") and not password:
        raise SystemExit(f"refusing to serve the dashboard on {host} without DASHBOARD_PASSWORD "
                         "(set it in /etc/alpha/.env, or use 127.0.0.1 with an SSH tunnel)")


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
    return JSONResponse(cached(("ledger",), data.ledger))


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


def index(request: Request):
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


def create_app(password: str | None = None) -> Starlette:
    return Starlette(routes=[
        Route("/", index),
        Route("/api/workflow", api_workflow),
        Route("/api/ledger", api_ledger),
        Route("/api/market", api_market),
        Route("/api/candles", api_candles),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ], middleware=[Middleware(BasicAuth, password=password)])


def main() -> None:
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    password = os.environ.get("DASHBOARD_PASSWORD") or None
    check_bind(host, password)
    uvicorn.run(create_app(password), host=host, port=int(os.environ.get("DASHBOARD_PORT", "8501")), log_level="warning")


if __name__ == "__main__":
    main()
