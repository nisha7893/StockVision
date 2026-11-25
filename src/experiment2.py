import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from data_loader import load_stock_data
from features import add_features, FEATURE_COLS
from market import add_market, MARKET_COLS

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]
EXTRA = {"^IXIC": "NASDAQ", "CL=F": "CRUDE", "GC=F": "GOLD", "^N225": "NIKKEI", "^HSI": "HSI"}
EXTRA_COLS = [n + "_ret" for n in EXTRA.values()]
THRESHOLD = 0.55

RF = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20,
                            random_state=42, n_jobs=-1)


def get_extra():
    out = pd.DataFrame()
    for sym, name in EXTRA.items():
        d = yf.download(sym, start="2017-10-01", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = d["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        out[name + "_ret"] = s.pct_change()
        print("extra data ok:", sym, len(s), "rows")
    return out.sort_index()


ex = get_extra()
SETS = {
    "Price + Market": FEATURE_COLS + MARKET_COLS,
    "Price + Market + Extra": FEATURE_COLS + MARKET_COLS + EXTRA_COLS,
}

rows = []
for t in TICKERS:
    data = add_market(add_features(load_stock_data(t)))
    data = data.merge(ex, left_on="Date", right_index=True, how="left")
    data[EXTRA_COLS] = data[EXTRA_COLS].ffill()
    d = data.dropna(subset=FEATURE_COLS + MARKET_COLS + EXTRA_COLS + ["Target_Return"]).reset_index(drop=True)
    y = (d["Target_Return"] > 0).astype(int).values

    for name, cols in SETS.items():
        X = d[cols].values
        for tr, te in TimeSeriesSplit(n_splits=5).split(X):
            m = clone(RF).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, list(m.classes_).index(1)]
            pred = (p >= 0.5).astype(int)
            yt = y[te]
            acc = (pred == yt).mean() * 100
            up = yt.mean() * 100

            conf = np.maximum(p, 1 - p) >= THRESHOLD
            if conf.sum() >= 10:
                acc_c = (pred[conf] == yt[conf]).mean() * 100
                up_c = yt[conf].mean() * 100
                cov = conf.mean() * 100
            else:
                acc_c, up_c, cov = np.nan, np.nan, 0.0
            rows.append([t, name, acc, up, acc_c, up_c, cov])
    print("done", t)

res = pd.DataFrame(rows, columns=["Ticker", "Features", "Acc", "AlwaysUP",
                                  "AccConf", "UpConf", "Coverage"])
res["Edge"] = res["Acc"] - res["AlwaysUP"]
res["EdgeConf"] = res["AccConf"] - res["UpConf"]
summary = res.groupby("Features")[["Acc", "AlwaysUP", "Edge", "AccConf", "EdgeConf", "Coverage"]].mean().round(2)
print()
print(summary.to_string())
print()
print(res.groupby(["Ticker", "Features"])["Edge"].mean().unstack().round(2))
res.to_csv("data/experiment2_results.csv", index=False)