# Data Fetching Module

This module contains functions for fetching stock data and retrieving company information from Yahoo Finance.

## Functions

1. **`fetch_stock_data(symbol, start, end)`**
   - Fetches historical stock data for a given symbol within a specified date range.
   - **Parameters**:
     - `symbol` (str): Stock ticker symbol (e.g., "AAPL").
     - `start` (datetime): Start date for the data.
     - `end` (datetime): End date for the data.
   - **Returns**:
     - A Pandas DataFrame containing the stock's historical data (Open, High, Low, Close, Volume).

2. **`get_company_info(symbol)`**
   - Retrieves basic information about a company, such as its name, sector, and industry.
   - **Parameters**:
     - `symbol` (str): Stock ticker symbol.
   - **Returns**:
     - A dictionary with keys `name`, `sector`, and `industry`.

## Usage
These functions are used in the main Streamlit app to fetch and display stock data and company details.