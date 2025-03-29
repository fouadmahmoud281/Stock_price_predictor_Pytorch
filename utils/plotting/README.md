# Plotting Module

This module contains functions for creating interactive visualizations of stock data and model predictions.

## Functions

1. **`plot_stock_data(df)`**
   - Creates an interactive candlestick chart with moving averages and volume bars.
   - **Parameters**:
     - `df` (DataFrame): A Pandas DataFrame containing stock data (Open, High, Low, Close, Volume).
   - **Returns**:
     - A Plotly figure object.

2. **`plot_prediction_results(dates, actual, predicted, train_size, future_dates=None, future_preds=None)`**
   - Creates an interactive chart comparing actual vs. predicted stock prices, with optional future predictions.
   - **Parameters**:
     - `dates` (list): List of dates corresponding to the data.
     - `actual` (array-like): Actual stock prices.
     - `predicted` (array-like): Predicted stock prices.
     - `train_size` (int): Size of the training dataset.
     - `future_dates` (list, optional): Dates for future predictions.
     - `future_preds` (array-like, optional): Future predicted prices.
   - **Returns**:
     - A Plotly figure object.

## Usage
These functions are used in the main Streamlit app to visualize stock data and model predictions interactively.