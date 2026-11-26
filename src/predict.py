import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from data_loader import load_stock_data
from features import add_features, FEATURE_COLS
from market import add_market, MARKET_COLS

ALL_COLS = FEATURE_COLS + MARKET_COLS


def predict_next_day(ticker):
    df = load_stock_data(ticker)
    if len(df) < 200:
        raise ValueError("Not enough data or invalid ticker. For NSE stocks add .NS, e.g. TCS.NS")

    data = add_market(add_features(df))
    today = data.iloc[[-1]][ALL_COLS]
    if today.isna().values.any():
        raise ValueError("Latest market data is incomplete. Please try again later.")
    train = data.dropna(subset=ALL_COLS + ["Target_Return"])
    X = train[ALL_COLS]
    y = train["Target_Return"]

    # Direction: Random Forest classifier (same settings as the experiment)
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20,
                                 random_state=42, n_jobs=-1).fit(X, (y > 0).astype(int))
    prob_up = float(clf.predict_proba(today)[0][list(clf.classes_).index(1)])

    # Size of the expected move: linear regression, capped at 2x typical daily move
    vol = float(data["Return"].tail(60).std())
    reg = LinearRegression().fit(X, y)
    move = min(abs(float(reg.predict(today)[0])), 2 * vol)
    pred_ret = move if prob_up >= 0.5 else -move

    last_close = float(df["Close"].iloc[-1])
    pred_price = last_close * (1 + pred_ret)

    edge = abs(prob_up - 0.5)
    signal = "No clear signal" if edge < 0.03 else "Moderate" if edge < 0.08 else "Strong"

    return {
        "ticker": ticker,
        "last_date": df["Date"].iloc[-1].date(),
        "last_close": last_close,
        "prob_up": prob_up,
        "pred_return_pct": pred_ret * 100,
        "pred_price": pred_price,
        "low": pred_price - vol * last_close,
        "high": pred_price + vol * last_close,
        "trend": "UP" if prob_up >= 0.5 else "DOWN",
        "signal": signal,
    }


if __name__ == "__main__":
    t = input("Enter ticker (e.g. TCS.NS): ").strip() or "RELIANCE.NS"
    r = predict_next_day(t)
    print(f"\n{r['ticker']}  (last data: {r['last_date']})")
    print(f"Last close : {r['last_close']:.2f}")
    print(f"Next day   : {r['pred_price']:.2f} ({r['pred_return_pct']:+.2f}%)")
    print(f"Range      : {r['low']:.2f} - {r['high']:.2f}")
    print(f"UP prob.   : {r['prob_up']*100:.0f}%")
    print(f"Trend      : {r['trend']}  | Signal: {r['signal']}")
    print("Educational purpose only, not financial advice.")