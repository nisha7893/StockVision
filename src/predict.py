import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_stock_data
from features import add_features, FEATURE_COLS


def predict_next_day(ticker):
    df = load_stock_data(ticker)
    if len(df) < 200:
       raise ValueError("Not enough data or invalid ticker. For NSE stocks add .NS, e.g. TCS.NS")

    data = add_features(df)
    today = data.iloc[[-1]][FEATURE_COLS].fillna(0)   # aaj ki row (Target abhi pata nahi)
    train = data.dropna()
    X, y = train[FEATURE_COLS], train["Target_Return"]

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                               random_state=42, n_jobs=-1).fit(X, y)
    lr = LinearRegression().fit(X, y)
    pred_ret = float((rf.predict(today)[0] + lr.predict(today)[0]) / 2)

    last_close = float(df["Close"].iloc[-1])
    pred_price = last_close * (1 + pred_ret)

    # Typical daily move (pichle 60 din) = range ka base
    vol = float(data["Return"].tail(60).std())
    strength = abs(pred_ret) / vol if vol > 0 else 0
    label = "Weak" if strength < 0.15 else "Moderate" if strength < 0.4 else "Strong"

    return {
        "ticker": ticker,
        "last_date": df["Date"].iloc[-1].date(),
        "last_close": last_close,
        "pred_return_pct": pred_ret * 100,
        "pred_price": pred_price,
        "low": pred_price - vol * last_close,
        "high": pred_price + vol * last_close,
        "trend": "UP" if pred_ret > 0 else "DOWN",
        "signal": label,
    }


if __name__ == "__main__":
    t = input("Enter ticker (e.g. TCS.NS): ").strip() or "RELIANCE.NS"
    r = predict_next_day(t)
    print(f"\n{r['ticker']}  (last data: {r['last_date']})")
    print(f"Last close : {r['last_close']:.2f}")
    print(f"Next day   : {r['pred_price']:.2f} ({r['pred_return_pct']:+.2f}%)")
    print(f"Range      : {r['low']:.2f} - {r['high']:.2f}")
    print(f"Trend      : {r['trend']}  | Signal: {r['signal']}")
    print("Educational purpose only, not financial advice.")