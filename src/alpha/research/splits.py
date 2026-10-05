"""Time splits and the validation lockbox.

    DEV      2020-01-01 -> 2024-07-01   all research, purged walk-forward
    VALID-A  2024-07-01 -> 2025-10-01   never seen by the new system: <= 3 finalists, each run once
    VALID-B  2025-10-01 -> paper start  run once at the very end (mildly contaminated: the old 4-token system's
                                        aggregate results there were looked at); the only history HYPE has
    PAPER    mainnet paper trading      the truly unseen test

The lockbox is a ledger file in data/experiments: open_lockbox() refuses a second run of the same candidate on the
same split, and more than MAX_FINALISTS candidates on VALID-A. It cannot stop someone deleting the file; it stops
accidents and makes every look at held-out data visible.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEV_START = pd.Timestamp("2020-01-01", tz="UTC")
DEV_END = pd.Timestamp("2024-07-01", tz="UTC")
VALID_A_END = pd.Timestamp("2025-10-01", tz="UTC")
MAX_FINALISTS = 3
LEDGER = Path("data/experiments/lockbox.jsonl")


@dataclass(frozen=True)
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None  # None = up to now (VALID-B ends when paper trading starts)

    def mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        m = index >= self.start
        return m & (index < self.end) if self.end is not None else m


DEV = Split("DEV", DEV_START, DEV_END)
VALID_A = Split("VALID-A", DEV_END, VALID_A_END)
VALID_B = Split("VALID-B", VALID_A_END, None)
SPLITS = {s.name: s for s in (DEV, VALID_A, VALID_B)}


def dev_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for research (index = signal/decision time)."""
    return df[DEV.mask(df.index)]


class LockboxError(RuntimeError):
    pass


def _ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def open_lockbox(candidate: str, split: str, ledger: Path = LEDGER) -> None:
    """Record a look at held-out data, or refuse it. Call before evaluating `candidate` on VALID-A / VALID-B."""
    if split not in ("VALID-A", "VALID-B"):
        raise ValueError(f"lockbox only guards held-out splits, got {split!r}")
    rows = _ledger(ledger)
    if any(r["candidate"] == candidate and r["split"] == split for r in rows):
        raise LockboxError(f"{candidate} was already evaluated on {split}; held-out data is used once")
    seen = {r["candidate"] for r in rows if r["split"] == split}
    if split == "VALID-A" and len(seen) >= MAX_FINALISTS:
        raise LockboxError(f"VALID-A already used by {MAX_FINALISTS} finalists: {sorted(seen)}")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps({"candidate": candidate, "split": split,
                            "at": datetime.now(timezone.utc).isoformat()}) + "\n")


def n_trials(log: Path = Path("data/experiments/log.jsonl")) -> int:
    """How many configurations have been evaluated so far (input to the deflated Sharpe ratio)."""
    return len(_ledger(log)) if log.exists() else 0
