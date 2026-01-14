"""
Data processing and transformation logic for path-based analysis.
"""

from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def reshape_to_daily_matrix(
    df: pd.DataFrame, price_col: str = "Close"
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[date]]:
    """
    Reshapes raw OHLC 1h data into a (Date x Hour) pivot table.
    Identifies and returns the 'Live' (incomplete) day if present.

    Args:
        df (pd.DataFrame): DataFrame with Datetime index and OHLC columns.
        price_col (str): The column name to use for price values. Defaults to "Close".

    Returns:
        Tuple: (daily_matrix_df, live_day_returns, live_day_date)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    working_df = df.copy()
    if not isinstance(working_df.index, pd.DatetimeIndex):
        working_df.index = pd.to_datetime(working_df.index)

    # Cast to DatetimeIndex for MyPy to recognize .date and .hour attributes
    dt_index = working_df.index
    working_df["Date"] = dt_index.date
    working_df["Hour"] = dt_index.hour

    # Pivot to create a matrix where rows are Dates and columns are Hours (0..23)
    daily_matrix_df = working_df.pivot(index="Date", columns="Hour", values=price_col)

    last_date = daily_matrix_df.index[-1]
    last_row = daily_matrix_df.iloc[-1]

    live_day_returns = None
    live_day_date = None

    if last_row.isna().any():
        live_day_prices = last_row.dropna().values
        if len(live_day_prices) > 0:
            # We need p0 (previous day's last known close)
            # Row -2 is the previous day relative to row -1
            prev_day_row = daily_matrix_df.iloc[-2]
            if not np.isnan(prev_day_row[23]):
                p0_live = prev_day_row[23]
                live_day_returns = np.log(live_day_prices / p0_live)
                live_day_date = last_date

        # Remove the incomplete day from the dataframe
        daily_matrix_df = daily_matrix_df.drop(last_date)

    # Filter for complete days (must have exactly 24 hours)
    daily_matrix_df = daily_matrix_df.dropna()

    return daily_matrix_df, live_day_returns, live_day_date


def calculate_daily_log_returns(daily_matrix_df: pd.DataFrame) -> Tuple[np.ndarray, List[date]]:
    """
    Calculates log returns for each day relative to the previous day's close (hour 23).

    Args:
        daily_matrix_df (pd.DataFrame): Shape (N, 24) with price values.

    Returns:
        Tuple: (series_matrix, valid_dates)
    """
    dates = daily_matrix_df.index
    valid_paths = []
    valid_dates = []

    price_matrix = daily_matrix_df.values

    # Start from 1 because we need i-1 for P0
    for i in range(1, len(price_matrix)):
        current_day_prices = price_matrix[i]
        prev_day_prices = price_matrix[i - 1]

        date_curr = dates[i]
        date_prev = dates[i - 1]

        # Use 1 day threshold to ensure relative continuity
        if (date_curr - date_prev) <= timedelta(days=1):
            p0 = prev_day_prices[-1]
            log_returns_path = np.log(current_day_prices / p0)

            valid_paths.append(log_returns_path)
            valid_dates.append(date_curr)

    return np.array(valid_paths), valid_dates
