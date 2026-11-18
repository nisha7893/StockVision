import sys
sys.path.insert(0, "src")

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import STOCKS
from data_loader import load_stock_data
from analysis import resample_ohlcv
from predict import predict_next_day
from db import save_prediction, save_prices, update_actuals, load_predictions
from backtest import run_backtest
from explain import explain

st.set_page_config(page_title="StockVision", page_icon="📈", layout="wide")
st.title("📈 StockVision")
st.caption("AI/ML based stock trend prediction system")


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker):
    df = load_stock_data(ticker)
    save_prices(ticker, df)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_prediction(ticker):
    return predict_next_day(ticker)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Select a stock")
    choice = st.selectbox("Quick select", ["Custom"] + list(STOCKS.keys()))
    default = "RELIANCE.NS" if choice == "Custom" else STOCKS[choice]
    ticker = st.text_input(
        "Ticker (add .NS for NSE stocks)", value=default, key=f"ticker_{choice}"
    ).strip().upper()
    st.info("Educational purpose only, not financial advice.")

# ---------- Data ----------
try:
    with st.spinner("Fetching data..."):
        df = get_data(ticker)
    if len(df) < 200:
        st.error("Not enough data or invalid ticker. Example: TCS.NS")
        st.stop()
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]
change = (last["Close"] / prev["Close"] - 1) * 100

st.subheader(f"{ticker}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close", f"₹{last['Close']:.2f}", f"{change:+.2f}%")
c2.metric("Day High", f"₹{last['High']:.2f}")
c3.metric("Day Low", f"₹{last['Low']:.2f}")
c4.metric("Volume", f"{int(last['Volume']):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Chart", "🔮 Prediction", "🧪 Model Comparison", "💰 Backtest", "📒 Tracker"])

# ---------- Tab 1: Chart ----------
with tab1:
    tf = st.radio("Timeframe", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Yearly": "YE"}[tf]
    data = resample_ohlcv(df, freq)
    if tf == "Daily":
        data = data.tail(250)

    fig = go.Figure(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="Price"))
    if len(data) > 20:
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"].rolling(20).mean(),
                                 name="MA20", line=dict(width=1.5)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                      title=f"{ticker} - {tf} chart")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Last 5 periods")
    st.dataframe(data[["Close", "Return %"]].tail().round(2))

# ---------- Tab 2: Prediction ----------
with tab2:
    st.write("Linear Regression and Random Forest are trained on the selected stock. This takes 5-10 seconds.")
    thr = st.slider("Sell only if the predicted drop is bigger than (%)", 0.0, 1.0, 0.0, 0.05)
    if st.button("Predict next day", type="primary"):
        try:
            with st.spinner("Training models..."):
                r = get_prediction(ticker)
                                 
                save_prediction(r)
            p1, p2, p3 = st.columns(3)
            arrow = "📈" if r["trend"] == "UP" else "📉"
            p1.metric("Predicted price", f"₹{r['pred_price']:.2f}",
                      f"{r['pred_return_pct']:+.2f}%")
            p2.metric("Trend", f"{arrow} {r['trend']}")
            p3.metric("Signal strength", r["signal"])
            st.write(f"**Expected range:** ₹{r['low']:.2f} to ₹{r['high']:.2f}")
            st.warning(
                "Next-day stock movement is highly random. The signal will be 'Weak' "
                "on most days, and that is normal. Do not treat this as trading advice."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- Tab 3: Model comparison ----------
with tab3:
    st.write("Comparison of all models on the Reliance test data:")
    if os.path.exists("data/results.csv"):
        res = pd.read_csv("data/results.csv")
        st.dataframe(res.round(2), hide_index=True)
        st.bar_chart(res.set_index("Model")[["RMSE", "MAE"]])
        st.info(
            "All models performed about the same as the naive baseline. This shows "
            "that next-day price prediction is inherently noisy."
        )
    else:
        st.warning("data/results.csv not found. Run `python src/lstm_model.py` first.")

# ---------- Tab 4: Backtest ----------
with tab4:
    st.write("What if you had followed this model's UP/DOWN signals with Rs 10,000? "
             "The model is trained on the first 80% of the data and tested on the last 20%, "
             "which it has never seen. A 0.1% cost is charged on every buy and sell.")
    if st.button("Run backtest", type="primary"):
        try:
            with st.spinner("Running backtest..."):
                curve, s = run_backtest(df, exit_threshold=thr / 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["Date"], y=curve["Strategy"], name="Model strategy", line=dict(color="#ff7f0e")))
            fig.add_trace(go.Scatter(x=curve["Date"], y=curve["Buy & Hold"], name="Buy & Hold"))
            fig.update_layout(height=450, title=f"{ticker}: Rs 10,000 from {s['start']} to {s['end']}",
                              yaxis_title="Portfolio value (Rs)")
            st.plotly_chart(fig, use_container_width=True)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Model strategy", f"Rs {s['strategy_final']:,.0f}",
                      f"{(s['strategy_final']/s['capital']-1)*100:+.1f}%")
            b2.metric("Buy & Hold", f"Rs {s['bh_final']:,.0f}",
                      f"{(s['bh_final']/s['capital']-1)*100:+.1f}%")
            b3.metric("Worst drop (strategy)", f"{s['strategy_dd']:.1f}%", f"Buy & Hold: {s['bh_dd']:.1f}%", delta_color="off")
            b4.metric("Trades made", s["trades"])

            if s["strategy_final"] > s["bh_final"]:
                st.success("The model strategy beat Buy & Hold on this test period.")
            else:
                st.info("Buy & Hold did better on this test period. This is common, "
                        "and it shows how hard it is to beat the market with price data alone.")
            st.caption("Past performance does not guarantee future results. Educational purpose only.")
        except Exception as e:
            st.error(f"Backtest failed: {e}")

# ---------- Tab 5: Prediction Tracker ----------
with tab5:
    st.write("Every prediction made in the Prediction tab is saved in SQLite. "
             "After the next trading day, click the button below to grade it against the real price.")
    if st.button("Update results"):
        try:
            with st.spinner("Checking real prices..."):
                n = update_actuals()
            st.success(f"{n} prediction(s) updated.")
        except Exception as e:
            st.error(f"Update failed: {e}")

    hist = load_predictions()
    if hist.empty:
        st.info("No predictions saved yet. Make one in the Prediction tab.")
    else:
        done = hist.dropna(subset=["actual_close"])
        if len(done) > 0:
            st.metric("Direction accuracy so far", f"{done['correct'].mean()*100:.1f}%",
                      f"{int(done['correct'].sum())} of {len(done)} correct")
        else:
            st.info("Predictions are saved. They can be graded after the next trading day.")
        st.dataframe(hist.round(2), hide_index=True)

# ---------- Explainer (inside the Prediction tab) ----------
@st.cache_data(ttl=3600, show_spinner=False)
def get_explanation(ticker):
    return explain(ticker)


with tab2:
    st.divider()
    st.subheader("🔍 Why this prediction?")
    if st.checkbox("Show explanation"):
        try:
            with st.spinner("Analysing indicators..."):
                imp, notes = get_explanation(ticker)
            st.write("**Current indicator readings**")
            for n in notes:
                st.write("- " + n)
            st.write("**What the Random Forest model relies on most**")
            st.bar_chart(imp.head(6))
            st.caption("Importance shows which inputs the model used most while learning. "
                       "It does not mean they cause the price to move.")
        except Exception as e:
            st.error(f"Explanation failed: {e}")
