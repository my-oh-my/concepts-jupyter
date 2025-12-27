"""
Data loading utilities for Stooq.
"""

import io
import time

import pandas as pd
import requests
from joblib import Parallel, delayed  # type: ignore[import-untyped]

from src.config import RAW_DATA_DIR


def get_from_stooq(symbol: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch data from Stooq with local caching.
    """
    cache_file = RAW_DATA_DIR / f"{symbol.lower()}.csv"

    if use_cache and cache_file.exists():
        data = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        # Filter by date range
        return data.loc[start_date:end_date]  # type: ignore[misc]

    # Define the URL for the HTTP GET request
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Read the CSV content from the response
            csv_data = response.content.decode("utf-8")
            csv_file_like = io.StringIO(csv_data)
            data = pd.read_csv(csv_file_like, index_col="Date", parse_dates=True)

            # Cache the raw data
            if use_cache:
                data.to_csv(cache_file)

            return data.loc[start_date:end_date]  # type: ignore[misc]

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch {symbol}: {e}")
                return pd.DataFrame()
            time.sleep(2**attempt)  # Exponential backoff

    return pd.DataFrame()


def load_multiple_symbols(symbols: list, start_date: str, end_date: str, n_jobs: int = -1) -> list:
    """
    Load multiple symbols in parallel.
    """

    def _fetch(s: str) -> dict:
        return {"symbol_name": s, "data": get_from_stooq(s, start_date, end_date)}

    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_fetch)(symbol) for symbol in symbols)
    return list(results) if results is not None else []
