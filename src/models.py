import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from features import add_features, FEATURE_COLS


def split_data(df, test_size=0.2):
    data = add_features(df).dropna()
    split = int(len(data) * (1 - test_size))   # time order same, shuffle nahi
    return data.iloc[:split], data.iloc[split:]


def evaluate(test, pred_ret):
    """pred_ret = predicted next-day return. Price = Close * (1 + return)."""
    pred_price = test["Close"].values * (1 + pred_ret)
    y_true = test["Target"].values
    rmse = np.sqrt(mean_squared_error(y_true, pred_price))
    mae = mean_absolute_error(y_true, pred_price)
    acc = (np.sign(pred_ret) == np.sign(test["Target_Return"].values)).mean() * 100
    return rmse, mae, acc, pred_price


def run_models(df):
    train, test = split_data(df)
    X_train, y_train = train[FEATURE_COLS], train["Target_Return"]
    X_test = test[FEATURE_COLS]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
    }

    rows, preds = [], {}

    # Baseline 1: tomorrow's price = today's price
    r, m, _, _ = evaluate(test, np.zeros(len(test)))
    rows.append(["Naive", r, m, np.nan])

    # Baseline 2: hamesha UP bolo
    r, m, a, _ = evaluate(test, np.full(len(test), 1e-9))
    rows.append(["Always UP", r, m, a])

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_ret = model.predict(X_test)
        r, m, a, pred_price = evaluate(test, pred_ret)
        preds[name] = pred_price
        rows.append([name, r, m, a])

    results = pd.DataFrame(rows, columns=["Model", "RMSE", "MAE", "Direction Acc %"])
    return results, test, preds


if __name__ == "__main__":
    df = pd.read_csv("data/stock_data.csv", parse_dates=["Date"])
    results, test, preds = run_models(df)
    print(results.round(2).to_string(index=False))
    results.to_csv("data/results.csv", index=False)