# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import yfinance as yf

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS
local_css("styles.css")

# Title of the app
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📈 Simple Stock Price Predictor 📈</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Predict stock prices using PyTorch and historical data.</p>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.markdown("<h2 style='color: #FF9800;'>📊 Stock Selection</h2>", unsafe_allow_html=True)
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL for Apple)", "AAPL", key="stock_symbol")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"), key="start_date")
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"), key="end_date")

if stock_symbol:
    try:
        # Fetch stock data using yfinance
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            st.error("❌ No data found for the given stock symbol and date range.")
        else:
            # Display stock data
            st.markdown("<h3 style='color: #2196F3;'>📋 Historical Data</h3>", unsafe_allow_html=True)
            st.dataframe(stock_data.tail())

            # Extract closing prices
            closing_prices = stock_data["Close"].values
            st.markdown(f"<p style='font-size: 14px;'>Closing Prices Shape: <code>{closing_prices.shape}</code></p>", unsafe_allow_html=True)

            # Convert to PyTorch tensor
            closing_prices_tensor = torch.tensor(closing_prices, dtype=torch.float32)

            # Reshape for processing
            X = torch.arange(len(closing_prices_tensor), dtype=torch.float32).unsqueeze(1)
            y = closing_prices_tensor.unsqueeze(1)

            # Split into training and testing data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Define a simple linear regression model
            class LinearRegressionModel(torch.nn.Module):
                def __init__(self):
                    super(LinearRegressionModel, self).__init__()
                    self.linear = torch.nn.Linear(1, 1)

                def forward(self, x):
                    return self.linear(x)

            # Initialize model, loss function, and optimizer
            model = LinearRegressionModel()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

            # Train the model
            epochs = 1000
            progress_bar = st.progress(0)
            for epoch in range(epochs):
                # Forward pass
                predictions = model(X_train)
                loss = criterion(predictions, y_train)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                progress_bar.progress((epoch + 1) / epochs)

            # Evaluate the model
            with torch.no_grad():
                predicted = model(X_test)

            # Display results
            st.markdown("<h3 style='color: #E91E63;'>📈 Predicted vs Actual Prices</h3>", unsafe_allow_html=True)
            results_df = pd.DataFrame({
                "Actual": y_test.squeeze().numpy(),
                "Predicted": predicted.squeeze().numpy()
            })
            st.line_chart(results_df)

            # Show model parameters
            st.markdown("<h3 style='color: #9C27B0;'>⚙️ Model Parameters</h3>", unsafe_allow_html=True)
            for name, param in model.named_parameters():
                st.markdown(f"<p style='font-size: 14px;'><strong>{name}:</strong> {param.data}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {e}")
else:
    st.info("ℹ️ Please enter a stock symbol to get started.")
