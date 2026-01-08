# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Medoid Path & Probability Bands Analysis
#
# This notebook demonstrates the **Medoid** approach for identifying the "most likely path" within a collection
# of time series.
#
# We will:
# 1. Download **Ethereum (ETH-USD)** 1-hour data.
# 2. Transform the continuous stream into a collection of **daily paths** (24 hours per day).
# 3. Determine the **Medoid Path** among these daily paths.
# 4. Calculate **Probability Bands** (Statistical Channel) around the Medoid.
# 5. Visualize the results using **Plotly**.

# %%
import os
import sys
from datetime import timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

# Add the src directory to the system path to allow importing modules
# pylint: disable=wrong-import-position
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Import the data loader
from src.dataio.yfinance_loader import fetch_market_data  # pylint: disable=wrong-import-position

# %% [markdown]
# ## 1. Core Algorithm: Medoid & Probability Channel
#
# We define the function `get_most_likely_channel` which:
# - Calculates the Euclidean distance matrix between all paths.
# - Identifies the Medoid (the path with the minimum sum of squared distances to all other paths).
# - Calculates percentiles for the probability bands.


# %%
def get_most_likely_channel(input_matrix: np.ndarray, probability: float = 0.75) -> Dict[str, Any]:
    """
    Finds the Medoid path and calculating the probability channel.

    Args:
        input_matrix (np.ndarray): Shape (n_series, time_steps)
        probability (float): The coverage probability for the bands (e.g., 0.75 for 75%).

    Returns:
        dict: Contains medoid path, lower bound, upper bound, and the index of the medoid.
    """
    # Step 1: Find the Medoid (The Most Likely single path)
    # input_matrix shape: (n_series, time_steps)
    # We use broadcasting to compute differences: (N, 1, T) - (1, N, T) -> (N, N, T)
    diffs = input_matrix[:, np.newaxis, :] - input_matrix[np.newaxis, :, :]

    # Euclidean distance between each pair of paths
    dist_matrix = np.sqrt(np.sum(np.square(diffs), axis=2))

    # The Medoid minimizes the sum of distances to all other points
    medoid_idx = np.argmin(np.sum(dist_matrix, axis=1))
    most_likely_path = input_matrix[medoid_idx]

    # Step 2: Calculate the Statistical Channel (Probability Range)
    # We calculate the distribution at each time step t across all N series
    tail = (1.0 - probability) / 2.0
    lower_bound = np.percentile(input_matrix, tail * 100, axis=0)
    upper_bound = np.percentile(input_matrix, (1.0 - tail) * 100, axis=0)

    return {"medoid": most_likely_path, "lower": lower_bound, "upper": upper_bound, "index": medoid_idx}


def find_nearest_path(partial_path: np.ndarray, history_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Finds the nearest neighbor in the history matrix for a given partial path.

    Args:
        partial_path (np.ndarray): Shape (k,) - The first k hours of a day.
        history_matrix (np.ndarray): Shape (N, 24) - The historical dataset.

    Returns:
        dict: Contains the matched full path, its index, and the distance.
    """
    k = len(partial_path)
    if k == 0:
        raise ValueError("Partial path cannot be empty.")

    # Slice history to the same length as partial_path
    history_slice = history_matrix[:, :k]

    # Calculate Euclidean distance between partial_path and all historical segments
    # Shape: (N,)
    distances = np.sqrt(np.sum(np.square(history_slice - partial_path), axis=1))

    # Find the index of the minimum distance
    best_match_idx = np.argmin(distances)
    best_match_path = history_matrix[best_match_idx]

    return {
        "matched_path": best_match_path,
        "index": best_match_idx,
        "distance": distances[best_match_idx],
    }


# %% [markdown]
# ## 2. Data Acquisition & Processing
#
# We fetch 1-hour data for ETH-USD. We aim for ~2 years of data to get a robust sample of daily behaviors.
#
# **Transformation Logic:**
# - We split the data into 24-hour chunks (Days).
# - For each day, we calculate the cumulative Log-Returns starting from a common reference point `P0`.
# - `P0` is defined as the *Close price of the previous day* (hour 23 of day D-1).
# - The path for Day D at hour h (0-23) is: $ \ln(P_{D,h} / P_{D-1, 23}) $.

# %%
# 1. Fetch Data
SYMBOL = "ETH-USD"
PERIOD = "730d"  # ~2 Years
INTERVAL = "1h"

print(f"Fetching {SYMBOL} data for the last {PERIOD}...")
df = fetch_market_data(SYMBOL, PERIOD, INTERVAL)

if df is None or df.empty:
    raise ValueError(f"No data fetched for {SYMBOL}")

# The loader returns a DataFrame with a 'Date' or 'Datetime' column (reset index).
# We need to set it back to the index for our time-based processing.
# Typically for 1h data, 'Datetime' is the column name.
DATE_COLUMN = "Datetime" if "Datetime" in df.columns else "Date"
if DATE_COLUMN in df.columns:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df.set_index(DATE_COLUMN, inplace=True)

# Ensure simple index if MultiIndex columns are present (just in case)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Keep only Close price
df = df[["Close"]].copy()
df.dropna(inplace=True)

print(f"Loaded {len(df)} data points.")

# %%
# 2. Reshape into Daily Paths

# Add date and hour info
# Cast index to DatetimeIndex to ensure .date and .hour properties are accessible for MyPy
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

df["Date"] = df.index.date
df["Hour"] = df.index.hour

# Pivot to create a matrix where rows are Dates and columns are Hours (0..23)
# This might have missing values if some hours are missing
daily_matrix_df = df.pivot(index="Date", columns="Hour", values="Close")

# Filter for complete days (must have exactly 24 hours)
daily_matrix_df = daily_matrix_df.dropna()
print(f"Data reshaped. Complete days found: {len(daily_matrix_df)}")

# 3. Calculate Log-Returns from Previous Close (P0)

# We need the previous day's close for each day in our filtered list.
# Since we might have gaps after filtering, we look back at the original df or shift the pivoted one
# if days are consecutive. A robust way is to re-index the pivoted DF to ensure we can access row i-1.

dates = daily_matrix_df.index
valid_paths = []
valid_dates = []

price_matrix = daily_matrix_df.values  # Shape (N, 24)

# We iterate from index 1 because we need index 0 as the 'previous day' for P0 reference
for i in range(1, len(price_matrix)):
    current_day_prices = price_matrix[i]
    prev_day_prices = price_matrix[i - 1]

    # Check if dates are consecutive (optional, but good for "Previous Close" logic)
    # If dates are not consecutive, using the previous row's close might be inaccurate if the gap is large.
    # However, for general "gap" analysis, comparing to the last known close is often acceptable.
    # Let's verify temporal continuity.
    date_curr = dates[i]
    date_prev = dates[i - 1]

    if (date_curr - date_prev) <= timedelta(days=1):
        # P0 is the last hour (index 23) of the previous row
        p0 = prev_day_prices[-1]

        # Log Returns relative to P0
        # path[t] = ln( P_t / p0 )
        log_returns_path = np.log(current_day_prices / p0)

        valid_paths.append(log_returns_path)
        valid_dates.append(date_curr)

series_matrix = np.array(valid_paths)
print(f"Final Series Matrix shape: {series_matrix.shape} (Days x Hours)")

# %% [markdown]
# ## 3. Analysis
# We run the Medoid algorithm on our collection of log-return paths.

# %%
PROBABILITY = 0.75
result = get_most_likely_channel(series_matrix, probability=PROBABILITY)

medoid_path = result["medoid"]
lower_path = result["lower"]
upper_path = result["upper"]
medoid_date = valid_dates[result["index"]]

print(f"Medoid Index: {result['index']}")
print(f"Most Representative Date: {medoid_date}")

# %% [markdown]
# ## 4. Visualization
# We use Plotly to create an interactive chart showing:
# 1. **All Paths**: Faint, to show density.
# 2. **The Medoid**: Highlighted, representing the "center" of behavior.
# 3. **Probability Bands**: The statistical "normal" range.

# %%
time_steps = np.arange(24)

fig = go.Figure()

# 1. Plot all background paths (Sampling a subset if too many for performance, e.g., max 500)
MAX_PATHS_TO_PLOT = 500
indices = np.arange(len(series_matrix))
if len(indices) > MAX_PATHS_TO_PLOT:
    np.random.shuffle(indices)
    indices = indices[:MAX_PATHS_TO_PLOT]

for idx in indices:
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=series_matrix[idx],
            mode="lines",
            line={"color": "rgba(150, 150, 150, 0.1)", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        )
    )

# 2. Plot Probability Bands (Filled Area)
# Upper Bound
fig.add_trace(
    go.Scatter(x=time_steps, y=upper_path, mode="lines", line={"width": 0}, showlegend=False, name="Upper Limit")
)

# Lower Bound (with fill)
fig.add_trace(
    go.Scatter(
        x=time_steps,
        y=lower_path,
        mode="lines",
        line={"width": 0},
        fill="tonexty",  # Fill to the trace before it (Upper Limit)
        fillcolor="rgba(0, 200, 255, 0.2)",
        showlegend=True,
        name=f"{int(PROBABILITY*100)}% Probability Band",
    )
)

# 3. Plot Medoid Path
fig.add_trace(
    go.Scatter(
        x=time_steps, y=medoid_path, mode="lines", line={"color": "blue", "width": 4}, name="Medoid (Most Likely Path)"
    )
)

fig.update_layout(
    title=f"Medoid Path Analysis: ETH-USD Daily 1H Profiles (Sample of {len(series_matrix)} days)",
    xaxis_title="Hour of Day (0-23 UTC)",
    yaxis_title="Log Return (vs Prev Day Close)",
    template="plotly_white",
    hovermode="x unified",
)

fig.show()

# %% [markdown]
# ## 5. Partial Path Matching (Forecasting / Nearest Neighbor)
#
# Here we simulate a real-world scenario:
# - We have observed the starting $k$ hours of a new day.
# - We want to find the **most similar day** from history to predict how the rest of the day might unfold.
#
# **Procedure:**
# 1. Select a random "Target Day" from our dataset to act as the "New Day".
# 2. Remove this Target Day from the "Historical Library" (to avoid matching with itself).
# 3. For $k$ in [2, 6, 12, 18, 23]:
#     - Take the first $k$ hours of the Target Day.
#     - Find the nearest neighbor in the Library (Euclidean distance on first $k$ steps).
#     - Plot the Input vs. the Matched Outcome.

# %%
# 1. Select a random target index
np.random.seed(42)  # For reproducibility in this demo
target_idx = np.random.randint(0, len(series_matrix))

target_full_path = series_matrix[target_idx]
target_date = valid_dates[int(target_idx)]

print(f"Target Day Selected: {target_date} (Index {target_idx})")

# 2. Create the Library (History excluding the target)
# We basically just mask out the target index
library_indices = np.arange(len(series_matrix)) != target_idx
library_matrix = series_matrix[library_indices]
library_dates = [d for i, d in enumerate(valid_dates) if i != target_idx]

# 3. Run Matching for different Partial Lengths
hours_to_test = [6, 12, 18, 23]

# Create a subplot for each test case

fig_match = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[f"Input: {h} Hours" for h in hours_to_test],
    shared_yaxes=True,
    x_title="Hour of Day",
    y_title="Log Return",
)

row_col_map = [(1, 1), (1, 2), (2, 1), (2, 2)]

for hours_val, (r, c) in zip(hours_to_test, row_col_map):
    # a. Slice the input
    partial_input = target_full_path[:hours_val]

    # b. Find nearest neighbor
    match_result = find_nearest_path(partial_input, library_matrix)
    matched_full_path = match_result["matched_path"]
    matched_idx_in_lib = match_result["index"]
    matched_date_str = library_dates[matched_idx_in_lib]

    # c. Plotting

    # Trace 1: The Matched Historical Path (Full Day)
    # We plot this first so it's in the background relative to the input
    fig_match.add_trace(
        go.Scatter(
            x=time_steps,
            y=matched_full_path,
            mode="lines",
            line={"color": "blue", "width": 2, "dash": "dot"},
            name=f"Matched: {matched_date_str}" if hours_val == hours_to_test[0] else None,
            showlegend=(hours_val == hours_to_test[0]),
        ),
        row=r,
        col=c,
    )

    # Trace 2: The 'Future' of the Target Path (Ground Truth) - Optional, for validation
    # Let's plot the full target path in faint grey to see how good the prediction was
    fig_match.add_trace(
        go.Scatter(
            x=time_steps,
            y=target_full_path,
            mode="lines",
            line={"color": "grey", "width": 1},
            name="Actual Full Path" if hours_val == hours_to_test[0] else None,
            showlegend=(hours_val == hours_to_test[0]),
        ),
        row=r,
        col=c,
    )

    # Trace 3: The Input Partial Path (What we observed)
    fig_match.add_trace(
        go.Scatter(
            x=time_steps[:hours_val],
            y=partial_input,
            mode="lines+markers",
            line={"color": "black", "width": 3},
            marker={"size": 6},
            name="Observed Input" if hours_val == hours_to_test[0] else None,
            showlegend=(hours_val == hours_to_test[0]),
        ),
        row=r,
        col=c,
    )

fig_match.update_layout(
    title=f"Partial Path Matching: Simulating Forecasting for {target_date}",
    height=800,
    width=1000,
    template="plotly_white",
    hovermode="x unified",
)

fig_match.show()
