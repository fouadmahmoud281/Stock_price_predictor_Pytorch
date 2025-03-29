import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_stock_data(df, stock_symbol):
    """Create an interactive stock price chart with volume."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1, subplot_titles=('Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )

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
    colors = ['red' if float(df['Open'].iloc[idx]) > float(df['Close'].iloc[idx]) else 'green' for idx in range(len(df))]
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
    """Create an interactive chart showing actual vs predicted values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[:train_size], y=actual[:train_size], mode='markers', marker=dict(size=4, color='lightgray'), name='Training Data'))
    fig.add_trace(go.Scatter(x=dates[train_size:], y=actual[train_size:], mode='lines+markers', marker=dict(size=6), line=dict(width=2, color='blue'), name='Actual Prices'))
    fig.add_trace(go.Scatter(x=dates[train_size:], y=predicted, mode='lines+markers', marker=dict(size=6), line=dict(width=2, color='red', dash='dash'), name='Predicted Prices'))
    if future_dates is not None and future_preds is not None:
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', marker=dict(size=8, symbol='star'), line=dict(width=3, color='purple'), name='Future Predictions'))
    fig.update_layout(title='Actual vs Predicted Stock Prices', xaxis_title='Date', yaxis_title='Price', template="plotly_white", height=500)
    return fig
