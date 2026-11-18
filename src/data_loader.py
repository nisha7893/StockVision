import yfinance as yf
import pandas as pd
from config import DEFAULT_TICKER, START_DATE


def load_stock_data(ticker=DEFAULT_TICKER, start=START_DATE, end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.dropna()
    df = df.reset_index()
    return df


if __name__ == "__main__":
    df = load_stock_data()
    df.to_csv("data/stock_data.csv", index=False)
    print(df.tail())
    print("Total rows:", len(df))