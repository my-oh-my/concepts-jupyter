"""
Statistical analysis utilities for market data.
"""

from itertools import groupby
from typing import Any, Dict, List

import pandas as pd
from scipy.signal import argrelmax, argrelmin


def transform_from_raw(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add RoR column to the dataframe."""
    local_df = dataframe.copy()
    local_df["RoR"] = local_df.Close.pct_change()
    return local_df


def transform_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reset index and insert Date as a column."""
    local_df = dataframe.copy()
    local_df.insert(0, "Date", local_df.index)
    local_df.index = pd.RangeIndex(len(local_df))
    return local_df


def clean_extremes(collection: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean consecutive extremes on the same side."""

    def process_group(group: List[Dict[str, Any]], extreme_type: str) -> Dict[str, Any]:
        if extreme_type == "at_low":
            return min(group, key=lambda x: x["extreme_value"])
        return max(group, key=lambda x: x["extreme_value"])

    waves = [process_group(list(g), k) for k, g in groupby(collection, key=lambda x: x["side"])]
    return waves


def get_extremes(dataframe: pd.DataFrame, extreme_candles_range: int) -> List[Dict[str, Any]]:
    """Identify local high and low extremes."""
    # Find local extremes
    minima_idx_dirty = argrelmin(dataframe["Low"].values, order=extreme_candles_range)[0]  # type: ignore[arg-type]
    minima_candidates = [
        {
            "idx": idx,
            "date_time": dataframe.iloc[idx]["Date"],
            "side": "at_low",
            "extreme_value": dataframe.iloc[idx]["Low"],
        }
        for idx in minima_idx_dirty
    ]

    maxima_idx_dirty = argrelmax(dataframe["High"].values, order=extreme_candles_range)[0]  # type: ignore[arg-type]
    maxima_candidates = [
        {
            "idx": idx,
            "date_time": dataframe.iloc[idx]["Date"],
            "side": "at_high",
            "extreme_value": dataframe.iloc[idx]["High"],
        }
        for idx in maxima_idx_dirty
    ]

    # Combine indices and sort them
    extremes_cleaned = clean_extremes(sorted(minima_candidates + maxima_candidates, key=lambda x: x["idx"]))
    return extremes_cleaned


def get_ror(current_value: float, previous_value: float) -> float:
    """Calculate rate of return."""
    if previous_value == 0:
        return 0.0
    return (current_value - previous_value) / previous_value
