import pandas as pd
import matplotlib.pyplot as plt


def resample_ohlcv(df, freq):
    """freq: 'D' daily, 'W' weekly, 'ME' monthly, 'YE' yearly"""
    d = df.set_index("Date")
    out = d.resample(freq).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    out["Return %"] = out["Close"].pct_change() * 100
    return out


def plot_timeframes(df, title="Reliance"):
    frames = {
        "Daily": resample_ohlcv(df, "D"),
        "Weekly": resample_ohlcv(df, "W"),
        "Monthly": resample_ohlcv(df, "ME"),
        "Yearly": resample_ohlcv(df, "YE"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (name, data) in zip(axes.flatten(), frames.items()):
        ax.plot(data.index, data["Close"])
        ax.set_title(f"{title} - {name} Close Price")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/timeframes.png")
    plt.show()
    return frames


if __name__ == "__main__":
    df = pd.read_csv("data/stock_data.csv", parse_dates=["Date"])
    frames = plot_timeframes(df)
    for name, data in frames.items():
        print(f"\n--- {name} (last 5) ---")
        print(data[["Close", "Return %"]].tail())