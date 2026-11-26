import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from features import add_features, FEATURE_COLS
from market import add_market, MARKET_COLS

COST = 0.001   # 0.1% cost every time we buy or sell
ALL_COLS = FEATURE_COLS + MARKET_COLS


def run_backtest(df, capital=10000, test_size=0.2, exit_threshold=0.0):
    data = add_market(add_features(df))
    data = data.dropna(subset=ALL_COLS + ["Target_Return"]).reset_index(drop=True)
    split = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split], data.iloc[split:].copy()

    clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20,
                                 random_state=42, n_jobs=-1)
    clf.fit(train[ALL_COLS], (train["Target_Return"] > 0).astype(int))
    prob_up = clf.predict_proba(test[ALL_COLS])[:, list(clf.classes_).index(1)]

    # Hold the stock when UP probability is at least (50% - margin). Otherwise stay in cash.
    position = (prob_up >= 0.5 - exit_threshold).astype(int)
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