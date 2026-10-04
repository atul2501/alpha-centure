"""Raw strategy report (no ML, no regime): uv run python -m alpha.research.raw"""

import sys

import pandas as pd
import psycopg
from loguru import logger

from alpha.config import get_settings
from alpha.research.dataset import build_dataset, summarize

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        ds = build_dataset(conn, s)
    print(f"\n{len(ds)} labeled setups from {ds.index.min()} to {ds.index.max()}\n")
    print("== by strategy x timeframe ==")
    print(summarize(ds, ["strategy", "tf"]).to_string())
    print("\n== by strategy x symbol ==")
    print(summarize(ds, ["strategy", "symbol"]).to_string())
    ds.to_parquet("data/setups_raw.parquet")


if __name__ == "__main__":
    main()
