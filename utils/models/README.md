# Models Module

This module defines PyTorch-based machine learning models for stock price prediction.

## Classes

1. **`LinearRegressionModel`**
   - A simple linear regression model for predicting stock prices based on time.
   - **Methods**:
     - `forward(x)`: Performs a forward pass through the model.

2. **`LSTMModel`**
   - A Long Short-Term Memory (LSTM) neural network for capturing complex temporal patterns in stock data.
   - **Parameters**:
     - `input_dim`: Number of input features.
     - `hidden_dim`: Number of hidden units in the LSTM layers.
     - `num_layers`: Number of LSTM layers.
     - `output_dim`: Number of output features.
     - `dropout`: Dropout rate for regularization.
   - **Methods**:
     - `forward(x)`: Performs a forward pass through the model.

3. **`GRUModel`**
   - A Gated Recurrent Unit (GRU) neural network, a simpler alternative to LSTM.
   - **Parameters**:
     - Same as `LSTMModel`.
   - **Methods**:
     - `forward(x)`: Performs a forward pass through the model.

## Usage
These models are instantiated and trained in the main Streamlit app to predict stock prices.