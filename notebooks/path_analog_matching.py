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
from typing import Any, Dict, List, cast

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
# 1. Data Source Settings
SYMBOL = "DNP.WA"
PERIOD = "730d"
INTERVAL = "1h"

# 2. Price Method Settings
# Options: "Close", "OHLC_Avg" (uses average of O+H+L+C)
PRICE_METHOD = "OHLC_Avg"

# 3. Session Settings
# Use 'UTC' for Crypto or specific timezones like 'America/New_York' for SPY
TIMEZONE = "Europe/Warsaw"
SESSION_START = 9  # 09:30-10:30 is the first full 1h candle in many feeds, or 9:00 for pre-open
SESSION_END = 17  # 15:00-16:00 is the last candle. Set to 15 for 16:00 close.

# 4. Matching Algorithm Settings
# Define the hours you want to check for partial matching
# (Should be within the session length)
HOURS_TO_TEST = [3, 5, 7]

# Number of nearest neighbors (top matches) to find
NUM_MATCHES = 3

# Matching Window: Number of days to use for matching context (Current + History).
MATCHING_WINDOW_DAYS = 10

# Future Sessions: Number of full days to show after the matched day.
FUTURE_SESSIONS = 3

# 5. Target Settings
# Define a specific target date to analyze (YYYY-MM-DD or None for Live/Last).
TARGET_DATE = None


# %% [markdown]
# ## 1. Data Acquisition & Processing

# %%
print(f"Fetching {SYMBOL} data...")
df_raw = fetch_market_data(SYMBOL, PERIOD, INTERVAL)

if df_raw is None or df_raw.empty:
    raise ValueError(f"No data fetched for {SYMBOL}")

# Process Datetime index
DATE_COLUMN = "Datetime" if "Datetime" in df_raw.columns else "Date"
if DATE_COLUMN in df_raw.columns:
    df_raw[DATE_COLUMN] = pd.to_datetime(df_raw[DATE_COLUMN])
    df_raw.set_index(DATE_COLUMN, inplace=True)

# Ensure simple index if MultiIndex columns are present (just in case)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

# Apply selected price method (configured in section 2)
if PRICE_METHOD == "OHLC_Avg":
    df_raw["OHLC_Avg"] = (df_raw["Open"] + df_raw["High"] + df_raw["Low"] + df_raw["Close"]) / 4
    PRICE_COL = "OHLC_Avg"
else:
    PRICE_COL = "Close"

# Store full OHLC for later candle plotting
df_full = df_raw[["Open", "High", "Low", "Close", PRICE_COL]].copy()
df_full.dropna(inplace=True)

# Transform into Daily Returns matrix
daily_matrix_df, LIVE_DAY_SERIES, LIVE_DAY_DATE = reshape_to_daily_matrix(
    df_full, price_col=PRICE_COL, tz=TIMEZONE, session_start=SESSION_START, session_end=SESSION_END
)

SESSION_LENGTH = SESSION_END - SESSION_START + 1

series_matrix, valid_dates = calculate_daily_log_returns(daily_matrix_df, _session_end=SESSION_END)

print(f"Data reshaped ({PRICE_METHOD}). Session: {SESSION_START}:00-{SESSION_END}:00 ({TIMEZONE})")
print(f"Complete days: {len(daily_matrix_df)}")
print(f"Series Matrix shape: {series_matrix.shape}")

# %% [markdown]
# ## 2. Partial Path Matching Configuration

# %%
# Configuration
# ---------------------------------------------------------
# Local processing variables
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
# Adjust HOURS_TO_TEST if it exceeds session length
HOURS_TO_TEST = [h for h in HOURS_TO_TEST if h <= SESSION_LENGTH]
if not HOURS_TO_TEST and not TARGET_DATE and LIVE_DAY_SERIES is not None:
    HOURS_TO_TEST = [len(LIVE_DAY_SERIES)]

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
    x_title="Time of Day (Relative to Target)",
)

# Shared reference for candle chart (collected during loop)
MATCH_DATA_FOR_CANDLES: List[Dict[str, Any]] = []

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
    HISTORY_HOURS, FUTURE_HOURS = LOOKBACK_WINDOW * SESSION_LENGTH, FUTURE_SESSIONS * SESSION_LENGTH
    plot_x_axis = np.arange(-HISTORY_HOURS, SESSION_LENGTH + FUTURE_HOURS)

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

        # Store for Candle Chart (only for the last analyzed 'hours_val' plot)
        if plot_idx == len(HOURS_TO_TEST) - 1:
            MATCH_DATA_FOR_CANDLES.append(
                {"rank": match_rank, "date": library_dates[real_idx_lib], "idx": real_idx_lib, "style": style}
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
    obs_x_axis = plot_x_axis[: len(full_target_vector)]
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
    for i in range(-LOOKBACK_WINDOW, FUTURE_SESSIONS + 2):
        hour_mark = i * SESSION_LENGTH
        fig_match.add_vline(x=hour_mark, line_width=1, line_dash="solid", line_color="black", opacity=0.3, row=r, col=c)
    fig_match.add_vline(x=0, line_width=2, line_dash="solid", line_color="black", row=r, col=c)

fig_match.update_layout(
    title=f"Path Analog Matching ({MATCHING_WINDOW_DAYS}-Day Window, {PRICE_METHOD}) for {TARGET_DATE_STR}",
    height=max(500, 500 * ROWS),
    width=1200,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Relative Session Hour (0 = Target Start)",
    yaxis_title=f"Log Return ({PRICE_METHOD})",
)
fig_match.show()

# %% [markdown]
# ## 4. Candle Chart Visualization
# This section displays the actual candles for the matched historical paths, normalized by their starting price.

# %%
# 1. Setup Candle Figure
fig_candles = go.Figure()

# Plotting Horizon
HISTORY_DAYS = LOOKBACK_WINDOW

for match in MATCH_DATA_FOR_CANDLES:
    rank = match["rank"]
    match_date = match["date"]
    style = match["style"]

    try:
        # Find index of match_date in our valid_dates list
        match_idx_in_history = [i for i, d in enumerate(valid_dates) if d == match_date][0]

        # Collect full path indices
        current_day_pos = daily_matrix_df.index.get_loc(match_date)
        start_pos = current_day_pos - LOOKBACK_WINDOW  # type: ignore
        end_pos = current_day_pos + FUTURE_SESSIONS + 1  # type: ignore
        block_dates = daily_matrix_df.index[start_pos:end_pos]

        # Get OHLC data for these dates, localized to the target timezone
        match_df = df_full.copy()
        m_idx = cast(pd.DatetimeIndex, match_df.index)
        if m_idx.tz is None:
            match_df.index = m_idx.tz_localize("UTC").tz_convert(TIMEZONE)
        else:
            match_df.index = m_idx.tz_convert(TIMEZONE)

        # refresh index reference
        m_idx = match_df.index

        # Filter for session hours and dates
        match_df["Hour"] = m_idx.hour
        match_df = match_df[(m_idx.date >= block_dates[0]) & (m_idx.date <= block_dates[-1])]
        match_df = match_df[(match_df["Hour"] >= SESSION_START) & (match_df["Hour"] <= SESSION_END)]

        # Find P0 for normalization (last price of the session before the block)
        prev_session_day = daily_matrix_df.index[start_pos - 1]
        p0_price = daily_matrix_df.loc[prev_session_day, SESSION_END]

        # Normalize
        for col in ["Open", "High", "Low", "Close"]:
            match_df[col] = match_df[col] / p0_price

        # Map to relative session hours
        # We need a way to map (Day, Hour) to a linear session index
        def get_rel_session_hour(dt: Any, b_dates: Any = block_dates) -> int:
            day_idx = (b_dates == dt.date()).argmax() - LOOKBACK_WINDOW
            hour_in_session = dt.hour - SESSION_START
            return int(day_idx * SESSION_LENGTH + hour_in_session)

        match_df["Relative_Hours"] = match_df.index.map(get_rel_session_hour)

        fig_candles.add_trace(
            go.Candlestick(
                x=match_df["Relative_Hours"],
                open=match_df["Open"],
                high=match_df["High"],
                low=match_df["Low"],
                close=match_df["Close"],
                name=f"Rank {rank}: {match_date}",
                opacity=0.8 if rank == 1 else 0.4,
                increasing_line_color=style["color"],
                decreasing_line_color="grey",
                showlegend=True,
            )
        )
    except (ValueError, IndexError, KeyError):
        print(f"Error plotting candles for {match_date}")

# 2. Add Target Path as candles
try:
    target_date_val = valid_dates[TARGET_IDX_IN_FULL_SERIES] if not IS_LIVE_MODE else LIVE_DAY_DATE

    # Collect target days
    target_pos = daily_matrix_df.index.get_loc(target_date_val)
    start_pos_t = target_pos - LOOKBACK_WINDOW  # type: ignore
    end_pos_t = target_pos + FUTURE_SESSIONS + 1  # type: ignore
    target_block_dates = daily_matrix_df.index[start_pos_t:end_pos_t]

    target_candle_df = df_full.copy()
    t_idx = cast(pd.DatetimeIndex, target_candle_df.index)
    if t_idx.tz is None:
        target_candle_df.index = t_idx.tz_localize("UTC").tz_convert(TIMEZONE)
    else:
        target_candle_df.index = t_idx.tz_convert(TIMEZONE)

    # refresh index reference
    t_idx = target_candle_df.index
    target_candle_df["Hour"] = t_idx.hour
    t_d_mask = (t_idx.date >= target_block_dates[0]) & (t_idx.date <= target_block_dates[-1])
    target_candle_df = target_candle_df[t_d_mask]
    target_candle_df = target_candle_df[
        (target_candle_df["Hour"] >= SESSION_START) & (target_candle_df["Hour"] <= SESSION_END)
    ]

    if not target_candle_df.empty:
        # Find P0 for Target
        prev_target_session_day = daily_matrix_df.index[start_pos_t - 1]
        target_p0_price = daily_matrix_df.loc[prev_target_session_day, SESSION_END]

        # Normalize
        for col in ["Open", "High", "Low", "Close"]:
            target_candle_df[col] = target_candle_df[col] / target_p0_price

        def get_rel_target_session_hour(dt: Any, tb_dates: Any = target_block_dates) -> int:
            day_idx = (tb_dates == dt.date()).argmax() - LOOKBACK_WINDOW
            hour_in_session = dt.hour - SESSION_START
            return int(day_idx * SESSION_LENGTH + hour_in_session)

        target_candle_df["Relative_Hours"] = target_candle_df.index.map(get_rel_target_session_hour)

        fig_candles.add_trace(
            go.Candlestick(
                x=target_candle_df["Relative_Hours"],
                open=target_candle_df["Open"],
                high=target_candle_df["High"],
                low=target_candle_df["Low"],
                close=target_candle_df["Close"],
                name=f"TARGET: {target_date_val}",
                increasing_line_color="black",
                decreasing_line_color="red",
                line_width=2,
                showlegend=True,
            )
        )
except (ValueError, IndexError, KeyError):
    print(f"Error plotting target candles: {target_date_val}")

# 3. Final Layout
for i in range(-LOOKBACK_WINDOW, FUTURE_SESSIONS + 2):
    hour_mark = i * SESSION_LENGTH
    fig_candles.add_vline(x=hour_mark, line_width=1, line_dash="solid", line_color="black", opacity=0.3)
fig_candles.add_vline(x=0, line_width=2, line_dash="solid", line_color="black")

fig_candles.update_layout(
    title=f"Normalized Candle Comparison for {SYMBOL} ({TARGET_DATE_STR})",
    xaxis_title="Relative Session Hour (0 = Target Start)",
    yaxis_title="Relative Price (P / P0)",
    template="plotly_white",
    height=700,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
)
fig_candles.show()
