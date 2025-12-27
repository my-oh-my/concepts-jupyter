# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks///py:percent
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
# Looking for instruments outperfoming some benchmark
# - considering instruments with certain correlation coefficient based on RoR,
# - considering outperfomers

# %%
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]

from src.analysis.stats import get_extremes, get_ror, transform_from_raw, transform_index

# Modular imports from src
from src.config import ALL_SYMBOLS, BENCHMARK_NAME, SWIG80_SYMBOLS
from src.dataio.stooq import load_multiple_symbols

# %%
# Set display options to show more rows and columns
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# %%
# Define dates
START_DATE = "2018-01-01"

# Get the current date
current_date = datetime.now().date()
END_DATE = current_date.strftime("%Y-%m-%d")
# END_DATE = '2025-12-23'

# %%
EXTREME_CANDLE_RANGE = 5

# %%
VOLUME_FLAG_THRESHOLD = 0.95

# %%
# Collect data in parallel
data = load_multiple_symbols(ALL_SYMBOLS, START_DATE, END_DATE)

# Process data to add RoR
for item in data:
    if not item["data"].empty:
        item["data"] = transform_from_raw(item["data"])

# %%
ror_df = pd.concat(
    [element["data"][["RoR"]].rename(columns={"RoR": element["symbol_name"]}) for element in data], axis=1
)
ror_df.fillna(0.0, inplace=True)

# %%
corr_matrix = ror_df.corr()

# %%
to_benchmark_coefficients = corr_matrix["WIG20"].sort_values(ascending=False)
# Symbols as the index and returns the values of Correlation Coefficient
benchmark_coefficients_df = pd.DataFrame(pd.Series(to_benchmark_coefficients, name="Corr"))

# %%
# Create a Plotly bar chart
fig = go.Figure(
    data=go.Bar(
        x=benchmark_coefficients_df.index,
        y=benchmark_coefficients_df["Corr"],
        marker_color="blue",  # Default color for bars
    )
)

# Customize the layout
fig.update_layout(
    title="Performance",
    xaxis_title="Symbol",
    yaxis_title="Correlation coefficient",
    yaxis={"gridcolor": "lightgray"},  # Adds grid lines on y-axis
    template="plotly_white",  # Optional: cleaner background
)

# %%
# Stack the correlation matrix to make it easier to sort
stacked_corr = corr_matrix.stack()

# Sort the correlation coefficients in descending order (ignoring self-correlation)
sorted_corr = stacked_corr[stacked_corr != 1].sort_values(ascending=False)
sorted_corr[:20]

# %%
# sorted_corr['WIG20']

# %%
# Performance ranking

# %%
# close_df = pd.concat(
#     [element["data"][["Close"]].rename(columns={"Close": element["symbol_name"]}) for element in data], axis=1
# )
# close_df.fillna(0.0, inplace=True)
# close_df = transform_index(close_df)

# %%
benchmark_df = next((item["data"] for item in data if item["symbol_name"] == BENCHMARK_NAME), None)
benchmark_df = transform_index(benchmark_df)  # type: ignore[arg-type]
benchmark_df[-10:]

# %%
extremes = get_extremes(benchmark_df, EXTREME_CANDLE_RANGE)
extremes[-10:]

# %%
previous_extreme = extremes[-1]
previous_extreme

# %%
market_side = previous_extreme["side"]

# %%
previous_extreme_date = extremes[-1]["date_time"]
after_extreme_date = benchmark_df.iloc[previous_extreme["idx"] + 1]["Date"]
# previous_extreme_date + pd.tseries.offsets.BDay(1)

# %%
recent_extreme_lowest_idx = benchmark_df["Low"][previous_extreme["idx"] :].idxmin()
recent_extreme_lowest_date = benchmark_df.loc[recent_extreme_lowest_idx, "Date"]
recent_extreme_lowest = benchmark_df.loc[recent_extreme_lowest_idx, "Low"]
(recent_extreme_lowest_idx, recent_extreme_lowest_date, recent_extreme_lowest)

# %%
recent_extreme_highest_idx = benchmark_df["Low"][previous_extreme["idx"] :].idxmax()
recent_extreme_highest_date = benchmark_df.loc[recent_extreme_highest_idx, "Date"]
recent_extreme_highest = benchmark_df.loc[recent_extreme_highest_idx, "High"]
(recent_extreme_highest_idx, recent_extreme_highest_date, recent_extreme_highest)

# %%
recent_extreme_idx = max(recent_extreme_lowest_idx, recent_extreme_highest_idx)
recent_extreme_date = benchmark_df.loc[recent_extreme_idx, "Date"]
recent_extreme = recent_extreme_lowest if market_side == "at_high" else recent_extreme_highest

# %%
recent_extreme

# %%
benchmark_ror = get_ror(recent_extreme, previous_extreme["extreme_value"])  # type: ignore[arg-type]
benchmark_ror

# %%
performance_data = {
    element["symbol_name"]: [
        get_ror(element["data"].loc[recent_extreme_date]["Low"], element["data"].loc[previous_extreme_date]["High"])
    ]
    for element in data
}

# Transpose the DataFrame to make symbols the index and returns the values
performance_df = pd.DataFrame(performance_data, index=["RoR"]).T
performance_df.sort_values(by="RoR", inplace=True)

# %%
# Create a list of colors based on the condition
colors = ["red" if symbol == BENCHMARK_NAME else "blue" for symbol in performance_df.index]

# Create the Plotly bar chart
fig = go.Figure(
    data=go.Bar(x=performance_df.index, y=performance_df["RoR"], marker_color=colors)  # Apply custom colors
)

# Customize layout
fig.update_layout(
    title="Performance",
    xaxis_title="Symbol",
    yaxis_title="Rate of Return",
    yaxis={"gridcolor": "lightgray"},  # Add grid lines
    template="plotly_white",  # Optional: for a cleaner look
)

# Show the plot
fig.show()

# %%
overview_df = pd.concat([benchmark_coefficients_df, performance_df], join="inner", ignore_index=False, axis=1)
overview_df = overview_df[overview_df["Corr"] >= 0.6]

# %%
benchmark_correlated_symbols = list(overview_df.index)
type(benchmark_correlated_symbols)

# %%
# Create numerical values for the x-axis (for labels)
x_values = overview_df.index

# Create a 3D scatter plot using plotly
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x_values,  # String index used for x-axis labels
            y=overview_df["Corr"],  # First column as y-axis
            z=overview_df["RoR"],  # Second column as z-axis
            mode="markers",
            marker={
                "size": 10,
                "color": overview_df["RoR"],  # Optional: color by Z-values for visual effect
                "colorscale": "Viridis",
                "opacity": 0.8,
            },
        )
    ]
)

# Update layout for labels and appearance
fig.update_layout(
    title="Corr to RoR",
    # width=800,
    # height=800,
    scene={"xaxis_title": "Symbol (X)", "yaxis_title": "Corr (Y)", "zaxis_title": "RoR (Z)"},
)

# Show the interactive plot
fig.show()

# %% [markdown]
# Rate of Returns daily since benchmark extreme date

# %%
# Get input for Cumulative RoR calculation
high_df = pd.concat(
    [
        element["data"][["High"]].rename(columns={"High": element["symbol_name"]})
        for element in data
        if element["symbol_name"] in benchmark_correlated_symbols
    ],
    axis=1,
)
low_df = pd.concat(
    [
        element["data"][["Low"]].rename(columns={"Low": element["symbol_name"]})
        for element in data
        if element["symbol_name"] in benchmark_correlated_symbols
    ],
    axis=1,
)
since_benchmark_extreme_input_df = (
    pd.concat([high_df.loc[previous_extreme_date].to_frame().T, low_df.loc[after_extreme_date:]])
    if market_side == "at_high"
    else pd.concat([low_df.loc[previous_extreme_date].to_frame().T, high_df.loc[after_extreme_date:]])
)

# %%
# Calculate daily returns
since_benchmark_extreme_ror_df = since_benchmark_extreme_input_df.pct_change()

# %%
# Calculate cumulative return (cumulative product of daily returns)
since_benchmark_extreme_cum_ror_df = (1 + since_benchmark_extreme_ror_df).cumprod() * 100
since_benchmark_extreme_cum_ror_df.iloc[0] = 100

# %%
# Create a Plotly figure
fig = go.Figure()

# Add a line for each stock
for stock in since_benchmark_extreme_cum_ror_df.columns:
    fig.add_trace(
        go.Scatter(
            x=since_benchmark_extreme_cum_ror_df.index,
            y=since_benchmark_extreme_cum_ror_df[stock],
            mode="lines",
            name=stock,  # Legend label
        )
    )

# Customize the layout
fig.update_layout(
    title="Rate of Return since benchmark extreme date",
    xaxis_title="Date",
    yaxis_title="Rate of Return",
    xaxis={"showgrid": True},  # Add grid lines for better readability
    yaxis={"showgrid": True},
    template="plotly_white",  # Optional: for a clean background
)

# Show the plot
fig.show()

# %% [markdown]
# Volume analysis

# %%
dict(since_benchmark_extreme_cum_ror_df.iloc[-1].sort_values(ascending=False))

# %%
volume_data = {
    element["symbol_name"]: (
        np.sum(element["data"]["Volume"] <= element["data"]["Volume"].iloc[-1]) / len(element["data"])
    )
    for element in data
}

filtered_volume_data = {key: value for key, value in volume_data.items() if key in SWIG80_SYMBOLS}

# Transpose the DataFrame to make symbols the index and returns the values
volume_df = pd.DataFrame([filtered_volume_data], index=["Volume_point"]).T
volume_df.sort_values(by="Volume_point", ascending=False, inplace=True)

filter_volume_df = volume_df[volume_df["Volume_point"] > VOLUME_FLAG_THRESHOLD]

# %%
# Create a list of colors based on the condition
colors = ["red" if symbol == BENCHMARK_NAME else "blue" for symbol in volume_df.index]

# Create the Plotly bar chart
fig = go.Figure(
    data=go.Bar(
        x=filter_volume_df.index, y=filter_volume_df["Volume_point"], marker_color=colors  # Apply custom colors
    )
)

# Customize layout
fig.update_layout(
    title="Volume",
    xaxis_title="Symbol",
    yaxis_title="Point",
    yaxis={"gridcolor": "lightgray"},  # Add grid lines
    template="plotly_white",  # Optional: for a cleaner look
)

# Show the plot
fig.show()
