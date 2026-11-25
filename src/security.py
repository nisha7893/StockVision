import os
import re
import time
import logging
import streamlit as st

os.makedirs("data", exist_ok=True)
logging.basicConfig(filename="data/app.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stockvision")

TICKER_RE = re.compile(r"^[A-Z0-9&\-\.\^=]{1,20}$")


def validate_ticker(raw):
    t = (raw or "").strip().upper()
    return t if TICKER_RE.match(t) else None


def rate_limited(key, max_calls=15, window=600):
    now = time.time()
    calls = [c for c in st.session_state.get(key, []) if now - c < window]
    limited = len(calls) >= max_calls
    if not limited:
        calls.append(now)
    st.session_state[key] = calls
    return limited