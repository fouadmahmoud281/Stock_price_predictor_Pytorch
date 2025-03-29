# Data Preprocessing Module

This module provides utility functions for preparing stock data for machine learning models.

## Functions

1. **`create_sequences(data, seq_length)`**
   - Creates sequences of data for time series prediction.
   - **Parameters**:
     - `data` (array-like): The input data (e.g., closing prices).
     - `seq_length` (int): The length of each sequence.
   - **Returns**:
     - Two NumPy arrays: `xs` (input sequences) and `ys` (target values).

2. **`scale_data(data)`**
   - Scales the input data using MinMaxScaler to normalize values between 0 and 1.
   - **Parameters**:
     - `data` (array-like): The input data to scale.
   - **Returns**:
     - A tuple containing the scaled data and the scaler object.

## Usage
These functions are used to preprocess stock data before feeding it into machine learning models like LSTM and GRU.