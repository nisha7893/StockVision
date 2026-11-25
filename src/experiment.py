import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data_loader import load_stock_data
from features import add_features, FEATURE_COLS

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]
MARKET = {"^GSPC": "US", "^NSEI": "NIFTY", "^INDIAVIX": "VIX", "USDINR=X": "USDINR"}
MARKET_COLS = ["US_ret", "NIFTY_ret", "VIX_level", "VIX_chg", "USDINR_ret"]

MODELS = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5,
                                            min_samples_leaf=20, random_state=42, n_jobs=-1),
}


def get_market():
    frames = {}
    for sym, name in MARKET.items():
        d = yf.download(sym, start="2017-10-01", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = d["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        frames[name] = s
        print("market data ok:", sym, len(s), "rows")
    m = pd.DataFrame(frames).sort_index().ffill()
    out = pd.DataFrame(index=m.index)
    out["US_ret"] = m["US"].pct_change()
    out["NIFTY_ret"] = m["NIFTY"].pct_change()
    out["VIX_level"] = m["VIX"]
    out["VIX_chg"] = m["VIX"].pct_change()
    out["USDINR_ret"] = m["USDINR"].pct_change()
    return out


mk = get_market()
rows = []
for t in TICKERS:
    data = add_features(load_stock_data(t))
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.merge(mk, left_on="Date", right_index=True, how="left")
    data[MARKET_COLS] = data[MARKET_COLS].ffill()
    data["Target5"] = data["Close"].shift(-5) / data["Close"] - 1

    for horizon, tcol in [("1-day", "Target_Return"), ("5-day", "Target5")]:
        d = data.dropna(subset=FEATURE_COLS + MARKET_COLS + [tcol]).reset_index(drop=True)
        y = (d[tcol] > 0).astype(int).values
        for fs_name, cols in [("Price only", FEATURE_COLS),
                              ("Price + Market", FEATURE_COLS + MARKET_COLS)]:
            X = d[cols].values
            for mname, model in MODELS.items():
                for tr, te in TimeSeriesSplit(n_splits=5).split(X):
                    m_ = clone(model).fit(X[tr], y[tr])
                    acc = (m_.predict(X[te]) == y[te]).mean() * 100
                    up = y[te].mean() * 100
                    rows.append([t, horizon, fs_name, mname, acc, up])
    print("done", t)

res = pd.DataFrame(rows, columns=["Ticker", "Horizon", "Features", "Model", "Acc", "AlwaysUP"])
res["Edge"] = res["Acc"] - res["AlwaysUP"]
summary = res.groupby(["Horizon", "Features", "Model"])[["Acc", "AlwaysUP", "Edge"]].mean().round(2)
print()
print(summary.to_string())
res.to_csv("data/experiment_results.csv", index=False)