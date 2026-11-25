import pandas as pd
import yfinance as yf

MARKET = {"^GSPC": "US", "^NSEI": "NIFTY", "^INDIAVIX": "VIX", "USDINR=X": "USDINR"}
MARKET_COLS = ["US_ret", "NIFTY_ret", "VIX_level", "VIX_chg", "USDINR_ret"]


def get_market(start="2017-10-01"):
    frames = {}
    for sym, name in MARKET.items():
        d = yf.download(sym, start=start, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = d["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        frames[name] = s
    m = pd.DataFrame(frames).sort_index().ffill()
    out = pd.DataFrame(index=m.index)
    out["US_ret"] = m["US"].pct_change()
    out["NIFTY_ret"] = m["NIFTY"].pct_change()
    out["VIX_level"] = m["VIX"]
    out["VIX_chg"] = m["VIX"].pct_change()
    out["USDINR_ret"] = m["USDINR"].pct_change()
    return out


def add_market(data):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.merge(get_market(), left_on="Date", right_index=True, how="left")
    data[MARKET_COLS] = data[MARKET_COLS].ffill()
    return data