import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_stock_data
from features import add_features, FEATURE_COLS

NAMES = {
    "Return": "Today's return",
    "Ret_lag1": "Return 1 day ago",
    "Ret_lag2": "Return 2 days ago",
    "Ret_lag3": "Return 3 days ago",
    "MA5_ratio": "Price vs 5-day average",
    "MA20_ratio": "Price vs 20-day average",
    "MA50_ratio": "Price vs 50-day average",
    "RSI": "RSI (14-day)",
    "MACD_pct": "MACD",
    "MACD_signal_pct": "MACD signal line",
    "Vol_change": "Volume change",
    "Volatility": "10-day volatility",
}


def explain(ticker):
    df = load_stock_data(ticker)
    data = add_features(df)
    train = data.dropna()
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                               random_state=42, n_jobs=-1)
    rf.fit(train[FEATURE_COLS], train["Target_Return"])

    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    imp.index = [NAMES[i] for i in imp.index]

    t = data.iloc[-1]
    notes = []

    rsi = t["RSI"]
    if rsi > 70:
        notes.append(f"RSI is {rsi:.0f}: above 70, the stock is often considered overbought.")
    elif rsi < 30:
        notes.append(f"RSI is {rsi:.0f}: below 30, the stock is often considered oversold.")
    else:
        notes.append(f"RSI is {rsi:.0f}: in the neutral zone (30 to 70).")

    gap = t["MA20_ratio"] * 100
    side = "above" if gap > 0 else "below"
    notes.append(f"Price is {abs(gap):.1f}% {side} its 20-day average.")

    if t["MACD_pct"] > t["MACD_signal_pct"]:
        notes.append("MACD is above its signal line: short-term momentum is positive.")
    else:
        notes.append("MACD is below its signal line: short-term momentum is negative.")

    vol = t["Volatility"] * 100
    notes.append(f"Recent daily volatility is {vol:.2f}%, which sets the width of the expected range.")

    return imp, notes
