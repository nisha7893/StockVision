import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from features import add_features, FEATURE_COLS

COST = 0.001   # 0.1% cost every time we buy or sell


def run_backtest(df, capital=10000, test_size=0.2, exit_threshold=0.0):
    data = add_features(df).dropna().reset_index(drop=True)
    split = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split], data.iloc[split:].copy()

    X, y = train[FEATURE_COLS], train["Target_Return"]
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                               random_state=42, n_jobs=-1).fit(X, y)
    lr = LinearRegression().fit(X, y)
    pred = (rf.predict(test[FEATURE_COLS]) + lr.predict(test[FEATURE_COLS])) / 2

    # Rule: if the model predicts UP, hold the stock tomorrow. Otherwise stay in cash.
    position = (pred > -exit_threshold).astype(int)
    trades = np.abs(np.diff(position, prepend=0))
    strat_ret = position * test["Target_Return"].values - trades * COST
    bh_ret = test["Target_Return"].values

    curve = pd.DataFrame({
        "Date": test["Date"].values,
        "Strategy": capital * np.cumprod(1 + strat_ret),
        "Buy & Hold": capital * np.cumprod(1 + bh_ret),
    })

    def max_dd(s):
        return float(((s / s.cummax()) - 1).min() * 100)

    stats = {
        "capital": capital,
        "strategy_final": float(curve["Strategy"].iloc[-1]),
        "bh_final": float(curve["Buy & Hold"].iloc[-1]),
        "strategy_dd": max_dd(curve["Strategy"]),
        "bh_dd": max_dd(curve["Buy & Hold"]),
        "trades": int(trades.sum()),
        "days": len(curve),
        "start": str(curve["Date"].iloc[0])[:10],
        "end": str(curve["Date"].iloc[-1])[:10],
    }
    return curve, stats
