from typing import Optional

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]


def fetch_market_data(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Fetches historical market data for a given stock symbol.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL').
        period (str): The time period for the data (e.g., '1y', '5y', 'max').
        interval (str): The data interval (e.g., '1d', '1wk', '1mo').

    Returns:
        Optional[pd.DataFrame]: A DataFrame with the fetched data or None if an error occurs.
    """
    try:
        # Fetch the data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        # Handle empty data
        if data.empty:
            print(f"Warning: No data found for symbol '{symbol}' with period '{period}' and interval '{interval}'.")
            return None

        # Reset the index to make 'Date' a column
        data.reset_index(inplace=True)
        return data  # type: ignore[no-any-return]

    except Exception as err:  # pylint: disable=broad-exception-caught
        print(f"Error fetching data for symbol '{symbol}': {err}")
        return None
