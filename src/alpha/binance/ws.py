"""Resilient combined-stream websocket runner."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable

from loguru import logger
from websockets.asyncio.client import connect

MAX_CONNECTION_AGE_S = 23 * 3600  # Binance drops connections at 24h; reconnect on our own terms first


async def run_stream(
    name: str,
    url: str,
    on_message: Callable[[str, dict], Awaitable[None]],
    on_connect: Callable[[], Awaitable[None]] | None = None,
    on_disconnect: Callable[[str], Awaitable[None]] | None = None,
    idle_timeout: float | None = 90,
) -> None:
    """Connects to a combined stream URL forever. on_message(stream_name, data) per event.

    idle_timeout: reconnect if no message arrives for this many seconds. Catches streams that stay
    connected but deliver nothing (e.g. a wrong endpoint), which would otherwise fail silently."""
    backoff = 1
    while True:
        reason = "closed"
        try:
            async with connect(url, ping_interval=20, ping_timeout=20, max_size=2**22, open_timeout=15) as ws:
                logger.info("[{}] connected", name)
                backoff = 1
                if on_connect:
                    await on_connect()
                started = time.monotonic()
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), idle_timeout)
                    except TimeoutError:
                        reason = f"no data for {idle_timeout}s"
                        break
                    msg = json.loads(raw)
                    try:
                        await on_message(msg["stream"], msg["data"])
                    except Exception:
                        logger.exception("[{}] handler error", name)
                    if time.monotonic() - started > MAX_CONNECTION_AGE_S:
                        reason = "max age reached"
                        break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        logger.warning("[{}] disconnected ({}), reconnecting in {}s", name, reason, backoff)
        if on_disconnect:
            try:
                await on_disconnect(reason)
            except Exception:
                logger.exception("[{}] on_disconnect error", name)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


def combined_url(base: str, streams: list[str]) -> str:
    return f"{base}/stream?streams={'/'.join(streams)}"
