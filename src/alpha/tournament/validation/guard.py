"""Held-out guard: tournament code may only see DEV bars (< DEV_END). VALID-A/B are consumed; the only true OOS
is forward shadow trading of frozen finalists (alpha.tournament.shadow), which is the one caller allowed to pass
allow_forward=True."""

import pandas as pd

from alpha.research.splits import DEV_END


class HeldOutAccess(RuntimeError):
    pass


def check_end(end: pd.Timestamp | None, allow_forward: bool = False) -> pd.Timestamp:
    """The exclusive end bound a loader may use."""
    if allow_forward:
        return end if end is not None else pd.Timestamp.now(tz="UTC")
    if end is None:
        return DEV_END
    end = pd.Timestamp(end)
    if end > DEV_END:
        raise HeldOutAccess(f"requested data up to {end}, beyond DEV_END {DEV_END}; VALID-A/B are consumed")
    return end


def assert_dev(index: pd.DatetimeIndex, what: str = "frame") -> None:
    if len(index) and index.max() >= DEV_END:
        raise HeldOutAccess(f"{what} contains bars >= DEV_END ({index.max()})")
