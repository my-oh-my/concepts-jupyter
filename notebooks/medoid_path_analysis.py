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
# Medoid Path Analysis
#
# This notebook demonstrates the **Medoid** approach for identifying the "most likely path" within a collection
# of time series.
#
# We will:
# 1. Download **Ethereum (ETH-USD)** 1-hour data.
# 2. Transform the continuous stream into a collection of **daily paths** (24 hours per day).
# 3. Determine the **Medoid Path** among these daily paths to find the "center" of behavior.
# 4. Calculate **Probability Bands** (Statistical Channel) around the Medoid.
# 5. Visualize the results using **Plotly**.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore

from src.analysis.path_analysis import calculate_medoid_channel
from src.analysis.transformation import calculate_daily_log_returns, reshape_to_daily_matrix
from src.dataio.yfinance_loader import fetch_market_data

# %% [markdown]
# ## 1. Core Algorithm: Medoid & Probability Channel
#
# We utilize modular functions from `src/analysis` to:
# - Calculate the Euclidean distance matrix between all paths.
# - Identify the Medoid (the path with the minimum sum of squared distances to all other paths).
# - Calculate percentiles for the probability bands.
# - Perform Nearest Neighbor matching for partial days.
#
#
# (Core functions moved to src.analysis.path_analysis)
# calculate_medoid_channel, find_nearest_neighbors, etc. are now imported.


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
# 2. Daily Matrix & Returns logic extracted to src.analysis.transformation
daily_matrix_df, LIVE_DAY_SERIES, LIVE_DAY_DATE = reshape_to_daily_matrix(df)
series_matrix, valid_dates = calculate_daily_log_returns(daily_matrix_df)

print(f"Data reshaped. Complete days found: {len(daily_matrix_df)}")
print(f"Final Series Matrix shape: {series_matrix.shape} (Days x Hours)")

# %% [markdown]
# ## 3. Analysis
# We run the Medoid algorithm on our collection of log-return paths.

# %%
PROBABILITY = 0.75
result = calculate_medoid_channel(series_matrix, probability=PROBABILITY)

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

# (Partial Path Matching functionality moved to notebooks/path_analog_matching.py)
