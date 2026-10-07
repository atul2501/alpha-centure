"""Resource tracking per fit (CPU, RSS, Apple-GPU memory, time, model size) and a RAM watchdog for the queue."""

import io
import os
import pickle
import time
from contextlib import contextmanager

import psutil

MIN_FREE_GB = 0.8  # macOS keeps ~80% "used" (cache, compressor), so an absolute free-memory floor is used


def gpu_mb() -> float | None:
    try:
        import sys

        if "torch" not in sys.modules:
            return None
        import torch

        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 2**20
    except Exception:
        return None
    return None


@contextmanager
def track():
    p = psutil.Process(os.getpid())
    p.cpu_percent(None)
    t0 = time.perf_counter()
    out = {}
    yield out
    out["seconds"] = time.perf_counter() - t0
    out["cpu_percent"] = p.cpu_percent(None)
    out["rss_mb"] = p.memory_info().rss / 2**20
    out["gpu_mb"] = gpu_mb()


def model_bytes(model) -> int | None:
    try:
        b = io.BytesIO()
        pickle.dump(model, b)
        return b.tell()
    except Exception:
        return None


def ram_ok() -> bool:
    return psutil.virtual_memory().available / 2**30 >= MIN_FREE_GB


def wait_for_ram(max_wait_s: float = 900) -> None:
    t0 = time.time()
    while not ram_ok() and time.time() - t0 < max_wait_s:
        time.sleep(5)
