import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from features import add_features, FEATURE_COLS
from models import run_models, evaluate

WINDOW = 30   # pichle 30 din dekhke agla din predict karega

np.random.seed(42)
tf.random.set_seed(42)


def make_sequences(X, window):
    seqs = [X[i - window + 1: i + 1] for i in range(window - 1, len(X))]
    return np.array(seqs)


def run_lstm(df, test_size=0.2, epochs=40):
    data = add_features(df).dropna().reset_index(drop=True)
    split = int(len(data) * (1 - test_size))

    scaler = MinMaxScaler().fit(data[FEATURE_COLS].iloc[:split])
    X = scaler.transform(data[FEATURE_COLS])
    y = data["Target_Return"].values * 100        # % mein, training aasan hoti hai

    X_seq = make_sequences(X, WINDOW)
    y_seq = y[WINDOW - 1:]
    idx = np.arange(WINDOW - 1, len(data))

    train_mask = idx < split
    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_test = X_seq[~train_mask]
    test = data.iloc[split:]                       # models.py wala hi test set

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, len(FEATURE_COLS))),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
              batch_size=32, callbacks=[stop], verbose=2)

    pred_ret = model.predict(X_test, verbose=0).flatten() / 100
    rmse, mae, acc, pred_price = evaluate(test, pred_ret)
    return model, [rmse, mae, acc], pred_price


if __name__ == "__main__":
    df = pd.read_csv("data/stock_data.csv", parse_dates=["Date"])

    results, test, preds = run_models(df)
    model, lstm_row, lstm_pred = run_lstm(df)

    results.loc[len(results)] = ["LSTM", *lstm_row]
    print("\n", results.round(2).to_string(index=False))
    results.to_csv("data/results.csv", index=False)
    model.save("data/lstm_model.keras")
