from datetime import date
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd


def reshape_to_daily_matrix(
    df: pd.DataFrame,
    price_col: str = "Close",
    tz: str = "UTC",
    session_start: int = 0,
    session_end: int = 23,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[date]]:
    """
    Reshapes raw OHLC 1h data into a (Date x Hour) pivot table, aware of sessions and timezones.
    Identifies and returns the 'Live' (incomplete) day if present.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # refreshing index reference
    idx = cast(pd.DatetimeIndex, df.index)  # type: ignore[redundant-cast]

    # Handle timezone conversion once and filter session in one go
    working_df = df.copy()
    working_df.index = idx.tz_localize("UTC").tz_convert(tz) if idx.tz is None else idx.tz_convert(tz)

    # refresh index reference
    idx = cast(pd.DatetimeIndex, working_df.index)  # type: ignore[redundant-cast]
    working_df["Date"] = idx.date
    working_df["Hour"] = idx.hour

    session_df = working_df[(working_df["Hour"] >= session_start) & (working_df["Hour"] <= session_end)]
    daily_matrix_df = session_df.pivot(index="Date", columns="Hour", values=price_col)

    if daily_matrix_df.empty:
        return pd.DataFrame(), None, None

    last_date = daily_matrix_df.index[-1]
    last_row = daily_matrix_df.iloc[-1]
    live_ret, live_date = None, None

    # Identify live (incomplete) day
    if last_row.isna().any() or len(last_row.dropna()) < (session_end - session_start + 1):
        complete_days = daily_matrix_df.dropna()
        if not last_row.dropna().empty and not complete_days.empty:
            live_ret = np.log(last_row.dropna().values / complete_days.iloc[-1][session_end])
            live_date = cast(date, last_date)

        daily_matrix_df = daily_matrix_df.drop(last_date)

    return daily_matrix_df.dropna(), live_ret, live_date


def calculate_daily_log_returns(daily_matrix_df: pd.DataFrame, _session_end: int = 23) -> Tuple[np.ndarray, List[date]]:
    """
    Calculates log returns for each day relative to the previous session's close.
    """
    if daily_matrix_df.empty:
        return np.array([]), []

    dates = daily_matrix_df.index
    valid_paths, valid_dates = [], []
    price_matrix = daily_matrix_df.values

    for i in range(1, len(price_matrix)):
        # p0 is the last column (session_end) of the previous day
        p0 = price_matrix[i - 1][-1]
        valid_paths.append(np.log(np.array(price_matrix[i]) / p0))
        valid_dates.append(cast(date, dates[i]))

    return np.array(valid_paths), valid_dates
