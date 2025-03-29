import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, start, end):
    """Fetch stock data from Yahoo Finance."""
    return yf.download(symbol, start=start, end=end)

def get_company_info(symbol):
    """Get company information using Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
        }
    except Exception:
        return {"name": symbol, "sector": "N/A", "industry": "N/A"}