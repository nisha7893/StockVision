# StockVision: AI/ML Based Stock Trend Prediction System

Predicts the next-day price trend of major Indian stocks using Linear Regression,
Random Forest and LSTM, and shows everything on an interactive Streamlit dashboard.

## Features
- Historical OHLCV data from the yFinance API (2018 to today)
- Daily, weekly, monthly and yearly analysis with candlestick charts
- Feature engineering: moving averages, RSI, MACD, returns, volatility
- Models: Linear Regression, Random Forest, LSTM (compared with RMSE, MAE, direction accuracy)
- Next-day prediction with expected range and signal strength
- Backtest simulator (model strategy vs Buy and Hold)
- Prediction tracker stored in SQLite
- "Why this prediction?" explainer

## Setup (Python 3.11)
    py -3.11 -m venv venv
    venv\Scripts\activate
    python -m pip install -r requirements.txt
    python src/data_loader.py
    python src/lstm_model.py
    python -m streamlit run app.py

## Team
Divyanshi Goel, Khushbu Saini, Nisha

Educational purpose only, not financial advice.

Live demo: https://stockvision-kjsv8xhtbvutg7einx9zqs.streamlit.app/
