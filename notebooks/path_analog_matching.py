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
# # Path Analog Matching Analysis (Forecasting)
#
# This notebook focuses on **Partial Path Matching** (Nearest Neighbor Search) to predict the outcome
# of the current day based on the most similar historical patterns.
#
# It utilizes:
# 1. **Multi-day lookback windows** for context-aware matching.
# 2. **Future session extension** to see the historical aftermath.
# 3. **Future leakage prevention** for robust backtesting.


# %%
# pylint: disable=duplicate-code
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from src.analysis.path_analysis import find_nearest_neighbors, get_lookback_vector, get_lookforward_vector
from src.analysis.transformation import calculate_daily_log_returns, reshape_to_daily_matrix
from src.dataio.yfinance_loader import fetch_market_data

# %% [markdown]
# ## 0. Global Settings

# %%
# Price Method: "Close" or "OHLC_Avg"
# OHLC_Avg calculates (Open + High + Low + Close) / 4
PRICE_METHOD = "OHLC_Avg"


# %% [markdown]
# ## 1. Data Acquisition & Processing

# %%
SYMBOL = "ETH-USD"
PERIOD = "730d"
INTERVAL = "1h"

print(f"Fetching {SYMBOL} data...")
df = fetch_market_data(SYMBOL, PERIOD, INTERVAL)

if df is None or df.empty:
    raise ValueError(f"No data fetched for {SYMBOL}")

# Process Datetime index
DATE_COLUMN = "Datetime" if "Datetime" in df.columns else "Date"
if DATE_COLUMN in df.columns:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df.set_index(DATE_COLUMN, inplace=True)

# Ensure simple index if MultiIndex columns are present (just in case)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Apply selected price method (configured in section 2)
if PRICE_METHOD == "OHLC_Avg":
    df["OHLC_Avg"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    PRICE_COL = "OHLC_Avg"
else:
    PRICE_COL = "Close"

df = df[[PRICE_COL]].copy()
df.dropna(inplace=True)

# Transform into Daily Returns matrix
daily_matrix_df, LIVE_DAY_SERIES, LIVE_DAY_DATE = reshape_to_daily_matrix(df, price_col=PRICE_COL)
series_matrix, valid_dates = calculate_daily_log_returns(daily_matrix_df)

print(f"Data reshaped ({PRICE_METHOD}). Complete days: {len(daily_matrix_df)}")
print(f"Series Matrix shape: {series_matrix.shape}")

# %% [markdown]
# ## 2. Partial Path Matching Configuration

# %%
# Configuration
# ---------------------------------------------------------
# Define the hours you want to check for partial matching
HOURS_TO_TEST = [6, 12, 18, 23]

# Number of nearest neighbors (top matches) to find
NUM_MATCHES = 3

# Matching Window: Number of days to use for matching context (Current + History).
MATCHING_WINDOW_DAYS = 2

# Future Sessions: Number of full days to show after the matched day.
FUTURE_SESSIONS = 1

# Define a specific target date to analyze (YYYY-MM-DD or None for Live/Last).
TARGET_DATE = None
# ---------------------------------------------------------

# Maintain copies for matching
library_matrix = series_matrix.copy()
library_dates = list(valid_dates)

# Variables to hold our target info
TARGET_PATH_FULL = None
TARGET_PATH_PARTIAL = None
TARGET_DATE_STR = ""
IS_LIVE_MODE = False
TARGET_IDX_IN_FULL_SERIES = -1

# %% [markdown]
# ## 3. Matching Execution & Plotting

# %%
# 1. Determine Target Logic
if TARGET_DATE:
    print(f"Mode: Historic Specific Date ({TARGET_DATE})")
    t_date_obj = pd.to_datetime(TARGET_DATE).date()
    date_indices = [i for i, d in enumerate(valid_dates) if d == t_date_obj]

    if not date_indices:
        raise ValueError(f"Date {TARGET_DATE} not found in valid history.")

    TARGET_IDX_IN_FULL_SERIES = date_indices[0]
    target_indices_to_exclude = list(range(TARGET_IDX_IN_FULL_SERIES, len(series_matrix)))
    TARGET_PATH_FULL = series_matrix[TARGET_IDX_IN_FULL_SERIES]
    TARGET_DATE_STR = str(valid_dates[TARGET_IDX_IN_FULL_SERIES])
else:
    if LIVE_DAY_SERIES is not None:
        print(f"Mode: LIVE Data ({LIVE_DAY_DATE})")
        TARGET_PATH_PARTIAL = LIVE_DAY_SERIES
        TARGET_DATE_STR = f"{LIVE_DAY_DATE} (LIVE)"
        IS_LIVE_MODE = True
        HOURS_TO_TEST = [len(LIVE_DAY_SERIES)]
        TARGET_IDX_IN_FULL_SERIES = len(series_matrix)
        target_indices_to_exclude = []
    else:
        print("Mode: Using LAST complete day.")
        TARGET_IDX_IN_FULL_SERIES = len(series_matrix) - 1
        target_indices_to_exclude = [TARGET_IDX_IN_FULL_SERIES]
        TARGET_PATH_FULL = series_matrix[TARGET_IDX_IN_FULL_SERIES]
        TARGET_DATE_STR = str(valid_dates[TARGET_IDX_IN_FULL_SERIES])

# 2. Setup Plot Grid
RANK_STYLES = {
    1: {"color": "blue", "dash": "dot", "width": 2},
    2: {"color": "rgba(0, 0, 255, 0.6)", "dash": "dashdot", "width": 2},
    "default": {"color": "rgba(0, 0, 255, 0.3)", "dash": "dash", "width": 1},
}

NUM_PLOTS = len(HOURS_TO_TEST)
COLS = min(NUM_PLOTS, 2)
ROWS = math.ceil(NUM_PLOTS / COLS)

fig_match = make_subplots(
    rows=ROWS,
    cols=COLS,
    subplot_titles=[f"Input: {h} Hours (Window: {MATCHING_WINDOW_DAYS} Days)" for h in HOURS_TO_TEST],
    shared_yaxes=True,
    x_title="Hour of Day (Relative)",
)

# 3. Main Loop
for plot_idx, hours_val in enumerate(HOURS_TO_TEST):
    r, c = (plot_idx // COLS) + 1, (plot_idx % COLS) + 1

    # Prepare Target Vector
    if IS_LIVE_MODE and TARGET_PATH_PARTIAL is not None:
        target_current_partial = TARGET_PATH_PARTIAL
    elif TARGET_PATH_FULL is not None:
        target_current_partial = TARGET_PATH_FULL[:hours_val]
    else:
        continue

    LOOKBACK_WINDOW = MATCHING_WINDOW_DAYS - 1
    target_Lookback_vec = get_lookback_vector(series_matrix, TARGET_IDX_IN_FULL_SERIES, LOOKBACK_WINDOW)

    if target_Lookback_vec is None:
        continue

    full_target_vector = np.concatenate([target_Lookback_vec, target_current_partial])

    # Build Library
    candidate_vectors, candidate_indices = [], []
    for idx_lib, day_path in enumerate(series_matrix):
        if idx_lib in target_indices_to_exclude:
            continue
        lib_lookback = get_lookback_vector(series_matrix, idx_lib, LOOKBACK_WINDOW)
        if lib_lookback is None:
            continue
        candidate_vectors.append(np.concatenate([lib_lookback, day_path[:hours_val]]))
        candidate_indices.append(idx_lib)

    library_search_matrix = np.array(candidate_vectors)
    if len(library_search_matrix) == 0:
        continue

    # Find Matches
    top_matches = find_nearest_neighbors(full_target_vector, library_search_matrix, n_neighbors=NUM_MATCHES)

    # Plotting Horizon
    HISTORY_HOURS, FUTURE_HOURS = LOOKBACK_WINDOW * 24, FUTURE_SESSIONS * 24
    plot_x_axis = np.arange(-HISTORY_HOURS, 24 + FUTURE_HOURS)
    show_legend = plot_idx == 0

    # Trace 1: Historical Matches
    for match in top_matches:
        match_rank = match["rank"]
        real_idx_lib = candidate_indices[match["index"]]

        hist_part = get_lookback_vector(series_matrix, real_idx_lib, LOOKBACK_WINDOW)
        curr_part = series_matrix[real_idx_lib]
        fut_part = get_lookforward_vector(series_matrix, real_idx_lib, FUTURE_SESSIONS)

        plot_data = np.concatenate([hist_part, curr_part, fut_part])
        current_trace_x = plot_x_axis[: len(plot_data)]
        style = RANK_STYLES.get(match_rank, RANK_STYLES["default"])

        fig_match.add_trace(
            go.Scatter(
                x=current_trace_x,
                y=plot_data,
                mode="lines",
                line={"color": style["color"], "width": style["width"], "dash": style["dash"]},
                name=f"Rank {match_rank}: {library_dates[real_idx_lib]}",
                showlegend=show_legend,
            ),
            row=r,
            col=c,
        )

    # Trace 2: Target Ground Truth (Grey)
    if not IS_LIVE_MODE:
        target_fut = get_lookforward_vector(series_matrix, TARGET_IDX_IN_FULL_SERIES, FUTURE_SESSIONS)
        target_plot_data = np.concatenate([target_Lookback_vec, TARGET_PATH_FULL, target_fut])
        fig_match.add_trace(
            go.Scatter(
                x=plot_x_axis[: len(target_plot_data)],
                y=target_plot_data,
                mode="lines",
                line={"color": "grey", "width": 1},
                name="Actual Full Path",
                showlegend=show_legend,
            ),
            row=r,
            col=c,
        )

    # Trace 3: Observed Input (Black)
    obs_x_axis = np.arange(-HISTORY_HOURS, -HISTORY_HOURS + len(full_target_vector))
    fig_match.add_trace(
        go.Scatter(
            x=obs_x_axis,
            y=full_target_vector,
            mode="lines+markers",
            line={"color": "black", "width": 3},
            marker={"size": 4},
            name="Observed Input",
            showlegend=show_legend,
        ),
        row=r,
        col=c,
    )

    # Session Borders
    for hour_mark in range(-HISTORY_HOURS, 24 + FUTURE_HOURS, 24):
        fig_match.add_vline(x=hour_mark, line_width=1, line_dash="solid", line_color="black", opacity=0.3, row=r, col=c)
    fig_match.add_vline(x=0, line_width=2, line_dash="solid", line_color="black", row=r, col=c)

fig_match.update_layout(
    title=f"Path Analog Matching ({MATCHING_WINDOW_DAYS}-Day Window, {PRICE_METHOD}) for {TARGET_DATE_STR}",
    height=max(500, 500 * ROWS),
    width=1200,
    template="plotly_white",
    hovermode="x unified",
    yaxis_title=f"Log Return ({PRICE_METHOD})",
)
fig_match.show()
