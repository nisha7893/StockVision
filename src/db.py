import sqlite3
import numpy as np
import pandas as pd
from data_loader import load_stock_data

DB_PATH = "data/stockvision.db"


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS prices (
        Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume REAL,
        PRIMARY KEY (Ticker, Date))""")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        ticker TEXT, last_date TEXT, last_close REAL, pred_price REAL,
        pred_return_pct REAL, trend TEXT, signal TEXT,
        actual_close REAL, correct INTEGER,
        PRIMARY KEY (ticker, last_date))""")
    return con


def save_prices(ticker, df):
    d = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    d["Date"] = d["Date"].astype(str)
    d.insert(0, "Ticker", ticker)
    con = _conn()
    con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)",
                    d.astype(object).values.tolist())
    con.commit()
    con.close()


def save_prediction(r):
    con = _conn()
    con.execute(
        "INSERT OR REPLACE INTO predictions "
        "(ticker, last_date, last_close, pred_price, pred_return_pct, trend, signal) "
        "VALUES (?,?,?,?,?,?,?)",
        (r["ticker"], str(r["last_date"]), r["last_close"], r["pred_price"],
         r["pred_return_pct"], r["trend"], r["signal"]))
    con.commit()
    con.close()


def update_actuals():
    """Fill in the actual next-day close for past predictions and grade them."""
    con = _conn()
    pending = con.execute(
        "SELECT ticker, last_date, last_close, pred_price FROM predictions "
        "WHERE actual_close IS NULL").fetchall()
    updated = 0
    cache = {}
    for ticker, last_date, last_close, pred_price in pending:
        if ticker not in cache:
            cache[ticker] = load_stock_data(ticker)
        df = cache[ticker]
        later = df[df["Date"] > pd.Timestamp(last_date)]
        if later.empty:
            continue
        actual = float(later["Close"].iloc[0])
        correct = int(np.sign(actual - last_close) == np.sign(pred_price - last_close))
        con.execute("UPDATE predictions SET actual_close=?, correct=? "
                    "WHERE ticker=? AND last_date=?",
                    (actual, correct, ticker, last_date))
        updated += 1
    con.commit()
    con.close()
    return updated


def load_predictions():
    con = _conn()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY last_date DESC", con)
    con.close()
    return df
