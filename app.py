# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import time

# Page configuration and settings
st.set_page_config(
    page_title="Advanced Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Default styling if CSS file is not found
        st.markdown("""
        <style>
        .main-header {text-align: center; color: #4CAF50; font-size: 3rem; margin-bottom: 0.5rem;}
        .sub-header {text-align: center; font-size: 1.2rem; color: #888; margin-bottom: 2rem;}
        .section-header {color: #2196F3; font-size: 1.8rem; margin-top: 2rem; margin-bottom: 1rem;}
        .card {background-color: #f9f9f9; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1.5rem;}
        .metric-card {text-align: center; background-color: #f0f8ff; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
        .metric-value {font-size: 1.8rem; font-weight: bold; color: #0066cc;}
        .metric-label {font-size: 0.9rem; color: #666;}
        .highlight {background-color: #ffffcc; padding: 0.2rem; border-radius: 3px;}
        .footer {text-align: center; margin-top: 3rem; color: #888; font-size: 0.8rem;}
        </style>
        """, unsafe_allow_html=True)

# Try to load custom CSS
try:
    local_css("styles.css")
except:
    pass

# App header
st.markdown("<h1 class='main-header'>📈 Advanced Stock Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Predict stock prices using machine learning models and historical data.</p>", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("<h2 style='color: #FF9800;'>📊 Configuration</h2>", unsafe_allow_html=True)

# Stock selection
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL for Apple)", "AAPL")
# Time period
time_period_options = {
    "1 Month": 30,
    "3 Months": 90, 
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "5 Years": 1825,
    "Custom Range": "custom"
}
selected_period = st.sidebar.selectbox("Select Time Period", list(time_period_options.keys()))

# Handle custom date range
if selected_period == "Custom Range":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("today"))
else:
    # Calculate dates based on selected period
    days = time_period_options[selected_period]
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=days)

# Model selection
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #9C27B0;'>🧠 Model Settings</h3>", unsafe_allow_html=True)
model_type = st.sidebar.selectbox(
    "Select Model Type", 
    ["Linear Regression", "LSTM", "GRU"]
)

# Training parameters
if model_type in ["LSTM", "GRU"]:
    sequence_length = st.sidebar.slider("Sequence Length (Days)", 5, 60, 20)
    hidden_size = st.sidebar.slider("Hidden Units", 8, 128, 32)
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, 2)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)

train_test_split = st.sidebar.slider("Training/Testing Split", 0.5, 0.9, 0.8, 0.05)
epochs = st.sidebar.slider("Training Epochs", 100, 5000, 1000, 100)
learning_rate = st.sidebar.select_slider(
    "Learning Rate", 
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.001
)

# Advanced options
st.sidebar.markdown("---")
show_advanced = st.sidebar.checkbox("Show Advanced Options", False)
if show_advanced:
    optimization_method = st.sidebar.selectbox(
        "Optimization Method", 
        ["SGD", "Adam", "RMSprop", "Adagrad"]
    )
    batch_size = st.sidebar.selectbox(
        "Batch Size",
        [8, 16, 32, 64, 128, "Full Batch"]
    )
    prediction_days = st.sidebar.slider("Future Prediction Days", 1, 30, 7)
    loss_function = st.sidebar.selectbox(
        "Loss Function",
        ["MSE", "MAE", "Huber Loss"]
    )
else:
    optimization_method = "Adam"
    batch_size = "Full Batch"
    prediction_days = 7
    loss_function = "MSE"

# Run button
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

# Footer in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='footer'>Developed with ❤️ using PyTorch & Streamlit<br>©2025</div>", 
    unsafe_allow_html=True
)

# Helper functions
def fetch_stock_data(symbol, start, end):
    """Fetch stock data from Yahoo Finance with caching"""
    return yf.download(symbol, start=start, end=end)

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Model definitions
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class GRUModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = torch.nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def evaluate_model(y_true, y_pred):
    """Calculate and return evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE": mape
    }

def plot_stock_data(df):
    """Create an interactive stock price chart with volume"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                      vertical_spacing=0.1, subplot_titles=('Price', 'Volume'),
                      row_heights=[0.7, 0.3])
    
    # Add candlestick chart for OHLC
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'].rolling(window=20).mean(),
            line=dict(color='orange', width=1),
            name="20-day MA"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'].rolling(window=50).mean(),
            line=dict(color='green', width=1),
            name="50-day MA"
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{stock_symbol} Stock Price',
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_prediction_results(dates, actual, predicted, train_size, future_dates=None, future_preds=None):
    """Create an interactive chart showing actual vs predicted values"""
    fig = go.Figure()
    
    # Add training data markers
    fig.add_trace(
        go.Scatter(
            x=dates[:train_size],
            y=actual[:train_size],
            mode='markers',
            marker=dict(size=4, color='lightgray'),
            name='Training Data'
        )
    )
    
    # Add actual test data
    fig.add_trace(
        go.Scatter(
            x=dates[train_size:],
            y=actual[train_size:],
            mode='lines+markers',
            marker=dict(size=6),
            line=dict(width=2, color='blue'),
            name='Actual Prices'
        )
    )
    
    # Add predicted test data
    fig.add_trace(
        go.Scatter(
            x=dates[train_size:],
            y=predicted,
            mode='lines+markers',
            marker=dict(size=6),
            line=dict(width=2, color='red', dash='dash'),
            name='Predicted Prices'
        )
    )
    
    # Add future predictions if available
    if future_dates is not None and future_preds is not None:
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_preds,
                mode='lines+markers',
                marker=dict(size=8, symbol='star'),
                line=dict(width=3, color='purple'),
                name='Future Predictions'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Actual vs Predicted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Main content
if stock_symbol:
    try:
        # Display loading animation while fetching data
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
            if stock_data.empty:
                st.error("❌ No data found for the given stock symbol and date range.")
                st.stop()
            
            # Get company info
            try:
                ticker = yf.Ticker(stock_symbol)
                company_info = ticker.info
                company_name = company_info.get('longName', stock_symbol)
                sector = company_info.get('sector', 'N/A')
                industry = company_info.get('industry', 'N/A')
            except:
                company_name = stock_symbol
                sector = "N/A"
                industry = "N/A"
        
        # Company information card
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {company_name} ({stock_symbol})")
            st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")
        
        with col2:
            # Fix for the Series comparison error - use scalar values
            if len(stock_data) >= 2:
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2]
                price_change = float(current_price) - float(prev_price)
                price_change_pct = (price_change / float(prev_price)) * 100
                
                price_color = "green" if price_change >= 0 else "red"
                change_symbol = "↑" if price_change >= 0 else "↓"
                
                st.markdown(f"#### Current Price")
                st.markdown(f"<span style='color:{price_color}; font-size:24px; font-weight:bold;'>${current_price:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:{price_color};'>{change_symbol} ${abs(price_change):.2f} ({price_change_pct:.2f}%)</span>", unsafe_allow_html=True)
            else:
                st.markdown("#### Current Price")
                st.markdown("Insufficient data to calculate price change")
        
        with col3:
            st.markdown("#### Trading Period")
            st.markdown(f"**From:** {start_date.strftime('%Y-%m-%d')}")
            st.markdown(f"**To:** {end_date.strftime('%Y-%m-%d')}")
            st.markdown(f"**Days:** {len(stock_data)}")
        
        # Display interactive stock chart
        st.markdown("<h3 class='section-header'>📊 Historical Data Visualization</h3>", unsafe_allow_html=True)
        stock_chart = plot_stock_data(stock_data)
        st.plotly_chart(stock_chart, use_container_width=True)
        
        # Display some basic statistics
        st.markdown("<h3 class='section-header'>📋 Key Statistics</h3>", unsafe_allow_html=True)
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            highest_price = float(stock_data['High'].max())
            initial_price = float(stock_data['Close'].iloc[0])
            st.metric("Highest Price", f"${highest_price:.2f}", f"{((highest_price / initial_price) - 1) * 100:.1f}%")
        
        with stat_col2:
            lowest_price = float(stock_data['Low'].min())
            st.metric("Lowest Price", f"${lowest_price:.2f}", f"{((lowest_price / initial_price) - 1) * 100:.1f}%")
        
        with stat_col3:
            avg_volume = float(stock_data['Volume'].mean())
            st.metric("Avg. Volume", f"{avg_volume:,.0f}")
        
        with stat_col4:
            volatility = stock_data['Close'].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        
        with st.expander("View Historical Data Table"):
            st.dataframe(stock_data)
                
        # Run analysis when button is clicked
        if run_button:
            st.markdown("<h3 class='section-header'>🧮 Price Prediction Analysis</h3>", unsafe_allow_html=True)
            
            # Data preparation
            closing_prices = stock_data["Close"].values
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(closing_prices.reshape(-1, 1))
            
            # Prepare data based on model type
            if model_type == "Linear Regression":
                # For linear regression, we use simple X (time index) and y (price)
                X = torch.arange(len(scaled_prices), dtype=torch.float32).unsqueeze(1)
                y = torch.tensor(scaled_prices, dtype=torch.float32)
                
                # Split data
                split_idx = int(train_test_split * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Initialize model
                model = LinearRegressionModel()
                
            else:  # LSTM or GRU
                # Create sequences
                X_seq, y_seq = create_sequences(scaled_prices, sequence_length)
                
                # Convert to PyTorch tensors
                X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
                y_seq_tensor = torch.tensor(y_seq, dtype=torch.float32)
                
                # Split data
                split_idx = int(train_test_split * len(X_seq_tensor))
                X_train, X_test = X_seq_tensor[:split_idx], X_seq_tensor[split_idx:]
                y_train, y_test = y_seq_tensor[:split_idx], y_seq_tensor[split_idx:]
                
                # Initialize model
                if model_type == "LSTM":
                    model = LSTMModel(
                        input_dim=1, 
                        hidden_dim=hidden_size, 
                        num_layers=num_layers, 
                        output_dim=1, 
                        dropout=dropout
                    )
                else:  # GRU
                    model = GRUModel(
                        input_dim=1, 
                        hidden_dim=hidden_size, 
                        num_layers=num_layers, 
                        output_dim=1, 
                        dropout=dropout
                    )
            
            # Set up loss function
            if loss_function == "MSE":
                criterion = torch.nn.MSELoss()
            elif loss_function == "MAE":
                criterion = torch.nn.L1Loss()
            else:  # Huber Loss
                criterion = torch.nn.SmoothL1Loss()
            
            # Set up optimizer
            if optimization_method == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            elif optimization_method == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optimization_method == "RMSprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            else:  # Adagrad
                optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
            
            # Training loop
            losses = []
            progress_text = st.empty()
            progress_bar = st.progress(0)
            loss_chart_placeholder = st.empty()
            
            for epoch in range(epochs):
                # Forward pass
                if model_type == "Linear Regression":
                    predictions = model(X_train)
                    loss = criterion(predictions, y_train)
                else:
                    predictions = model(X_train)
                    loss = criterion(predictions, y_train.unsqueeze(1))
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track progress
                losses.append(loss.item())
                
                # Update progress every 10 epochs (for performance)
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                    progress_text.text(f"Training: Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
                    progress_bar.progress((epoch + 1) / epochs)
                    
                    # Update loss chart periodically
                    if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(losses)
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('Training Loss Curve')
                        ax.grid(True)
                        loss_chart_placeholder.pyplot(fig)
            
            # Evaluate on test data
            model.eval()
            with torch.no_grad():
                if model_type == "Linear Regression":
                    predicted = model(X_test)
                    actual_prices = y_test.numpy()
                    predicted_prices = predicted.numpy()
                else:
                    predicted = model(X_test)
                    actual_prices = y_test.numpy()
                    predicted_prices = predicted.squeeze().numpy()
            
            # Inverse transform to get original scale
            actual_prices_orig = scaler.inverse_transform(actual_prices.reshape(-1, 1)).flatten()
            predicted_prices_orig = scaler.inverse_transform(predicted_prices.reshape(-1, 1)).flatten()
            
            # Calculate evaluation metrics
            metrics = evaluate_model(actual_prices_orig, predicted_prices_orig)
            
            # Display metrics
            st.markdown("<h3 class='section-header'>📏 Model Performance Metrics</h3>", unsafe_allow_html=True)
            
            metric_cols = st.columns(len(metrics))
            for col, (metric_name, metric_value) in zip(metric_cols, metrics.items()):
                with col:
                    if metric_name == "R²":
                        st.metric(metric_name, f"{metric_value:.4f}")
                    elif metric_name == "MAPE":
                        st.metric(metric_name, f"{metric_value:.2f}%")
                    else:
                        st.metric(metric_name, f"{metric_value:.4f}")
            
            # Prepare dates for plotting
            dates = stock_data.index.tolist()
            split_date_idx = int(train_test_split * len(dates))
            
            # Prepare future predictions
            if prediction_days > 0:
                future_dates = pd.date_range(
                    start=dates[-1] + pd.Timedelta(days=1),
                    periods=prediction_days,
                    freq='B'  # Business days
                )
                
                # Generate future predictions
                if model_type == "Linear Regression":
                    future_X = torch.arange(
                        len(X), len(X) + prediction_days, 
                        dtype=torch.float32
                    ).unsqueeze(1)
                    
                    with torch.no_grad():
                        future_predictions = model(future_X)
                        future_predictions = scaler.inverse_transform(
                            future_predictions.numpy().reshape(-1, 1)
                        ).flatten()
                else:
                    # Get the last sequence from the data
                    last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
                    future_predictions = []
                    
                    # Make predictions one by one and update the sequence
                    current_sequence = torch.tensor(last_sequence, dtype=torch.float32)
                    
                    for _ in range(prediction_days):
                        with torch.no_grad():
                            prediction = model(current_sequence).item()
                            future_predictions.append(prediction)
                            
                            # Update the sequence by removing the first value and adding the new prediction
                            current_sequence = torch.cat([
                                current_sequence[:, 1:, :],
                                torch.tensor([[[prediction]]], dtype=torch.float32)
                            ], dim=1)
                    
                    # Inverse transform to get original scale
                    future_predictions = scaler.inverse_transform(
                        np.array(future_predictions).reshape(-1, 1)
                    ).flatten()
            else:
                future_dates = None
                future_predictions = None
            
            # Plot prediction results
            prediction_chart = plot_prediction_results(
                dates, 
                closing_prices, 
                predicted_prices_orig, 
                split_date_idx,
                future_dates,
                future_predictions
            )
            st.plotly_chart(prediction_chart, use_container_width=True)
            
            # Display future predictions in a table if available
            if future_dates is not None and future_predictions is not None:
                st.markdown("<h3 class='section-header'>🔮 Future Price Predictions</h3>", unsafe_allow_html=True)
                
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })
                
                # Safely calculate percentage changes for the future predictions
                future_df['Change %'] = [0]
                if len(future_predictions) > 1:
                    for i in range(1, len(future_predictions)):
                        pct_change = (future_predictions[i] / future_predictions[i-1] - 1) * 100
                        future_df.loc[i, 'Change %'] = pct_change
                
                # Format the dataframe for display
                future_df_display = future_df.copy()
                future_df_display['Predicted Price'] = future_df_display['Predicted Price'].apply(lambda x: f"${x:.2f}")
                future_df_display['Change %'] = future_df_display['Change %'].apply(lambda x: f"{x:+.2f}%")
                
                st.dataframe(
                    future_df_display,
                    use_container_width=True
                )
                
                # Add disclaimer
                st.info("⚠️ Disclaimer: These predictions are based on historical data and mathematical models. They should not be used as the sole basis for investment decisions.")
            
            # Show model details
            with st.expander("Model Details"):
                st.markdown(f"**Model Type:** {model_type}")
                
                if model_type == "Linear Regression":
                    # For linear model, show the parameters
                    for name, param in model.named_parameters():
                        st.markdown(f"**{name}:** {param.data.item():.6f}")
                    
                    # Also show the predicted formula
                    weight = model.linear.weight.data.item()
                    bias = model.linear.bias.data.item()
                    st.markdown(f"**Predicted equation:** Price = {weight:.6f} × Day + {bias:.6f}")
                    
                else:
                    # For LSTM/GRU, show architecture summary
                    st.markdown(f"**Sequence Length:** {sequence_length}")
                    st.markdown(f"**Hidden Units:** {hidden_size}")
                    st.markdown(f"**Number of Layers:** {num_layers}")
                    st.markdown(f"**Dropout Rate:** {dropout}")
                    
                    # Show total parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    st.markdown(f"**Total Parameters:** {total_params:,}")
            
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
        st.exception(e)
else:
    # Display initial instructions and info when no stock is selected
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h3 style='color: #4CAF50;'>📝 How to use this app</h3>
        <ol>
            <li>Enter a valid stock symbol in the sidebar (e.g., AAPL, MSFT, GOOG)</li>
            <li>Select a time period or define a custom date range</li>
            <li>Choose a model type and configure parameters</li>
            <li>Click "Run Analysis" to train the model and make predictions</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h3 style='color: #FF9800;'>📈 Available Models</h3>
        <ul>
            <li><strong>Linear Regression</strong>: Simple trend-based prediction</li>
            <li><strong>LSTM</strong>: Long Short-Term Memory neural network for complex patterns</li>
            <li><strong>GRU</strong>: Gated Recurrent Unit, a simpler and often faster alternative to LSTM</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
    <h3 style='color: #2196F3;'>⚠️ Disclaimer</h3>
    <p>This application is for educational and demonstration purposes only. The predictions made by this tool should not be used as financial advice or as the sole basis for investment decisions. Stock markets are complex systems influenced by numerous factors that cannot be fully captured by these models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show some example stock symbols
    st.markdown("""
    <h3 style='color: #9C27B0;'>🔍 Popular Stock Symbols</h3>
    """, unsafe_allow_html=True)
    
    popular_stocks = {
        "Technology": ["AAPL (Apple)", "MSFT (Microsoft)", "GOOGL (Alphabet)", "AMZN (Amazon)", "META (Meta Platforms)"],
        "Finance": ["JPM (JPMorgan)", "BAC (Bank of America)", "WFC (Wells Fargo)", "GS (Goldman Sachs)"],
        "Healthcare": ["JNJ (Johnson & Johnson)", "PFE (Pfizer)", "UNH (UnitedHealth)", "ABBV (AbbVie)"],
        "Consumer": ["KO (Coca-Cola)", "PEP (PepsiCo)", "MCD (McDonald's)", "NKE (Nike)", "SBUX (Starbucks)"]
    }
    
    tabs = st.tabs(list(popular_stocks.keys()))
    
    for i, (category, stocks) in enumerate(popular_stocks.items()):
        with tabs[i]:
            for stock in stocks:
                st.markdown(f"• {stock}")
