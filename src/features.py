import numpy as np
import pandas as pd

FEATURE_COLS = [
    "Return", "Ret_lag1", "Ret_lag2", "Ret_lag3",
    "MA5_ratio", "MA20_ratio", "MA50_ratio",
    "RSI", "MACD_pct", "MACD_signal_pct",
    "Vol_change", "Volatility",
]


def add_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Ret_lag1"] = df["Return"].shift(1)
    df["Ret_lag2"] = df["Return"].shift(2)
    df["Ret_lag3"] = df["Return"].shift(3)

    # Moving averages (price ke ratio mein)
    for n in (5, 20, 50):
        df[f"MA{n}_ratio"] = df["Close"] / df["Close"].rolling(n).mean() - 1

    # RSI (14 din)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD (% mein)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_pct"] = macd / df["Close"] * 100
    df["MACD_signal_pct"] = signal / df["Close"] * 100

    df["Vol_change"] = df["Volume"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    # Targets
    df["Target"] = df["Close"].shift(-1)
    df["Target_Return"] = df["Close"].shift(-1) / df["Close"] - 1

    return df.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    raw = pd.read_csv("data/stock_data.csv", parse_dates=["Date"])
    out = add_features(raw)
    print(out[["Date"] + FEATURE_COLS + ["Target_Return"]].tail())