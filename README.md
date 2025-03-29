# Stock Price Predictor with PyTorch and Streamlit

<div align="center">
  <img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" alt="Stock Price Predictor Banner" height="100"/>
  <img src="https://raw.githubusercontent.com/streamlit/docs/main/public/logo.svg" alt="Streamlit Logo" height="100"/>
  <p><em>Harness the power of AI to predict stock market trends</em></p>
</div>

## 🚀 Overview
This project is a Streamlit-based web application that predicts stock prices using advanced machine learning models. By leveraging historical stock data from Yahoo Finance, it trains models to identify patterns and make predictions about future price movements.

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Graph/SVG/ic_fluent_graph_line_24_regular.svg" width="60" alt="Chart"/></td>
      <td align="center"><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Brain/SVG/ic_fluent_brain_circuit_24_regular.svg" width="60" alt="AI"/></td>
      <td align="center"><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Chart/SVG/ic_fluent_chart_multiple_24_regular.svg" width="60" alt="Analytics"/></td>
    </tr>
    <tr>
      <td align="center"><b>Real-time Data</b></td>
      <td align="center"><b>AI Models</b></td>
      <td align="center"><b>Insights</b></td>
    </tr>
  </table>
</div>

## ✨ Features

### 📊 Interactive Data Visualization
- **Dynamic Candlestick Charts** with customizable time frames
- **Technical Indicators** including moving averages, MACD, and RSI
- **Volume Analysis** with color-coded volume bars

### 🧠 Powerful ML Models
<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <h3>Linear Regression</h3>
        <img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Line/SVG/ic_fluent_line_24_regular.svg" width="40"/>
        <p>Simple trend-based prediction for baseline comparison</p>
      </td>
      <td align="center" width="33%">
        <h3>LSTM</h3>
        <img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Arrow%20Recycle/SVG/ic_fluent_arrow_recycle_24_regular.svg" width="40"/>
        <p>Captures complex temporal patterns in stock data</p>
      </td>
      <td align="center" width="33%">
        <h3>GRU</h3>
        <img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Circuit%20Board/SVG/ic_fluent_circuit_board_24_regular.svg" width="40"/>
        <p>A faster and simpler alternative to LSTM</p>
      </td>
    </tr>
  </table>
</div>

### ⚙️ Customizable Parameters
- **Model Configuration**
  - Sequence length
  - Network architecture (hidden units, layers)
  - Regularization (dropout rate)
- **Training Options**
  - Dataset split ratio
  - Training epochs
  - Learning rate and optimization methods
  - Loss functions

### 🔮 Future Price Prediction
- Forecast stock prices for your chosen time horizon
- Visualize prediction confidence intervals
- Compare multiple models simultaneously

### 📏 Comprehensive Performance Metrics
- MSE, RMSE, MAE for error analysis
- R² for trend accuracy assessment
- MAPE for percentage-based evaluation
- Visual comparison of actual vs. predicted values

## 🗂️ Project Structure

```
📦 streamlit_app/
 ┣ 📜 app.py                  # Main Streamlit application
 ┣ 📜 styles.css              # Custom styling
 ┣ 📂 utils/                  # Utility modules
 ┃ ┣ 📜 __init__.py           # Package initialization
 ┃ ┣ 📜 data_fetching.py      # Yahoo Finance data API integration
 ┃ ┣ 📜 preprocessing.py      # Data cleaning and preparation
 ┃ ┣ 📜 models.py             # PyTorch model definitions
 ┃ ┣ 📜 evaluation.py         # Performance metrics
 ┃ ┗ 📜 plotting.py           # Visualization functions
 ┗ 📜 requirements.txt        # Dependencies
```

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8+
- Git (optional)

### Quick Start Guide

<details>
<summary><b>1. Clone the Repository</b></summary>

```bash
git clone https://github.com/your-repo/stock-price-predictor.git
cd stock-price-predictor
```
</details>

<details>
<summary><b>2. Create a Virtual Environment (Recommended)</b></summary>

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```
</details>

<details>
<summary><b>3. Install Dependencies</b></summary>

```bash
pip install -r requirements.txt
```
</details>

<details>
<summary><b>4. Launch the Application</b></summary>

```bash
streamlit run app.py
```
</details>

## 🖥️ User Interface

<div align="center">
  <img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Window%20Dev%20Tools/SVG/ic_fluent_window_dev_tools_24_regular.svg" alt="Application Screenshot" width="60"/>
</div>

### 🎛️ Control Panel

The sidebar provides all the controls you need:

1. **Data Selection**
   - Enter any valid stock ticker (e.g., AAPL, MSFT, GOOGL)
   - Choose predefined time ranges or custom dates
   - Select data granularity (daily, weekly, intraday)

2. **Model Configuration**
   - Pick your prediction model (Linear Regression, LSTM, GRU)
   - Configure network architecture and hyperparameters
   - Set training parameters

3. **Advanced Settings**
   - Optimization methods (SGD, Adam, RMSprop, Adagrad)
   - Batch processing options
   - Custom prediction horizons
   - Loss function selection

### 📈 Interactive Dashboard

The main panel provides a comprehensive analysis:

1. **Company Overview**
   - Key statistics and company information
   - Current market status and recent performance

2. **Historical Analysis**
   - Interactive candlestick chart with zoom and pan
   - Customizable technical indicators
   - Trading volume visualization

3. **Prediction Results**
   - Model performance metrics with comparisons
   - Actual vs predicted visualization
   - Future price forecasting with confidence intervals

4. **Backtesting**
   - Model accuracy on historical periods
   - Performance under different market conditions

## 🔍 How It Works

<div align="center">
  <table>
    <tr>
      <td align="center" width="25%"><b>Step 1</b><br>Data Collection</td>
      <td align="center" width="25%"><b>Step 2</b><br>Preprocessing</td>
      <td align="center" width="25%"><b>Step 3</b><br>Model Training</td>
      <td align="center" width="25%"><b>Step 4</b><br>Prediction</td>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Database/SVG/ic_fluent_database_24_regular.svg" width="40"/></td>
      <td><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Data%20Funnel/SVG/ic_fluent_data_funnel_24_regular.svg" width="40"/></td>
      <td><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Arrow%20Sync/SVG/ic_fluent_arrow_sync_24_regular.svg" width="40"/></td>
      <td><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Prediction/SVG/ic_fluent_prediction_24_regular.svg" width="40"/></td>
    </tr>
    <tr>
      <td>Fetch historical data from Yahoo Finance API</td>
      <td>Scale features, create sequences, split data</td>
      <td>Train selected model with optimized parameters</td>
      <td>Generate predictions and visualize results</td>
    </tr>
  </table>
</div>

## 📋 Code Explanation

### Main Application (`app.py`)

The core application orchestrates:

1. **UI Configuration** - Sets up the Streamlit interface
2. **Data Pipeline** - Manages data flow from source to models
3. **Model Management** - Handles training and inference
4. **Visualization** - Creates interactive charts and reports

### Neural Network Architecture

Our LSTM and GRU implementations are carefully designed for time series forecasting:

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
```

## 📦 Dependencies

- **streamlit**: Interactive web interface
- **torch**: Deep learning framework
- **yfinance**: Stock data API
- **pandas & numpy**: Data manipulation
- **plotly & matplotlib**: Data visualization
- **scikit-learn**: Evaluation metrics and preprocessing

## ⚠️ Disclaimer

<div align="center">
  <img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Warning/SVG/ic_fluent_warning_24_regular.svg" width="60" alt="Warning"/>
</div>

This application is intended for **educational and demonstration purposes only**. The predictions are based on historical data and mathematical models, not financial expertise. Stock markets are influenced by numerous factors that cannot be fully captured by these models.

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

See our [contribution guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Mail/SVG/ic_fluent_mail_24_regular.svg" width="30" alt="Email"/></td>
      <td align="center"><img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Globe/SVG/ic_fluent_globe_24_regular.svg" width="30" alt="Website"/></td>
      <td align="center"><img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" width="30" alt="GitHub"/></td>
    </tr>
    <tr>
      <td align="center"><a href="mailto:your-email@example.com">Email</a></td>
      <td align="center"><a href="https://your-website.com">Website</a></td>
      <td align="center"><a href="https://github.com/your-username">GitHub</a></td>
    </tr>
  </table>
</div>

---

<div align="center">
  <p><b>Ready to predict the future of stocks?</b></p>
  <p>⭐ Star this repo if you found it useful! ⭐</p>
  <img src="https://raw.githubusercontent.com/microsoft/fluentui-system-icons/master/assets/Star/SVG/ic_fluent_star_24_regular.svg" width="30"/>
</div>
