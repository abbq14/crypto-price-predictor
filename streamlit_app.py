import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from main import fetch_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "trading_model.pkl"
FEATURE_COLS = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'SMA_20', 'SMA_50',
                'BB_upper', 'BB_middle', 'BB_lower', 'Volume']

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
TIMEFRAMES = ['15m', '1h', '4h', '1d']

st.set_page_config(page_title="Crypto Price Predictor", page_icon="📈", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=300)
def load_features(symbol, timeframe):
    return fetch_features(symbol, timeframe=timeframe, limit=200)


st.title("📈 Crypto Price Predictor")
st.caption(
    "Predicts next-candle direction for a trading pair from technical indicators "
    "(RSI, MACD, SMA, Bollinger Bands) using a RandomForest classifier."
)

with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Trading pair", SYMBOLS, index=0)
    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=1)
    st.divider()
    st.warning(
        "**Educational project — not financial advice.** The model was trained on "
        "BTC/USDT 1h data; other pairs and timeframes are exploratory only.",
        icon="⚠️",
    )
    st.info(
        "The model's features include raw price levels (SMA, Bollinger Bands), so it "
        "only behaves meaningfully near the price range it was trained on. Predictions "
        "for non-BTC pairs are out-of-distribution and should not be trusted.",
        icon="ℹ️",
    )
    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

try:
    model = load_model()
    df = load_features(symbol, timeframe)
except Exception as e:
    st.error(f"Could not load market data or model: {e}")
    st.stop()

if len(df) < 2 or df[FEATURE_COLS].iloc[-2].isna().any():
    st.error("Not enough candle history to compute indicators for this pair/timeframe.")
    st.stop()

last_complete = df.iloc[-2]
features = pd.DataFrame([{c: last_complete[c] for c in FEATURE_COLS}])

pred = model.predict(features)[0]
confidence = model.predict_proba(features)[0][pred]
direction = "UP" if pred == 1 else "DOWN"

current_price = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2]
change_pct = (current_price - prev_close) / prev_close * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pair", symbol)
c2.metric("Current price", f"${current_price:,.2f}", f"{change_pct:+.2f}%")
c3.metric("Next-candle prediction", f"{'🟢' if pred == 1 else '🔴'} {direction}")
c4.metric("Model confidence", f"{confidence:.1%}")

if symbol != 'BTC/USDT':
    st.warning(
        f"**{symbol} is outside the model's training distribution.** This model was "
        "fitted on BTC/USDT price levels and will produce unreliable — often "
        "constant — predictions on other pairs. Shown for demonstration only.",
        icon="⚠️",
    )

st.divider()

tab_price, tab_ind, tab_data = st.tabs(["Price & Bollinger Bands", "Indicators", "Raw data"])

with tab_price:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name=symbol,
    ))
    for col, dash in [('BB_upper', 'dot'), ('BB_middle', 'dash'), ('BB_lower', 'dot')]:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df[col], name=col.replace('_', ' '),
            line=dict(width=1, dash=dash),
        ))
    for col in ['SMA_20', 'SMA_50']:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col, line=dict(width=1.5)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

with tab_ind:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("RSI (14)")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], name='RSI'))
        rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold")
        rsi_fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(rsi_fig, width="stretch")
    with col_b:
        st.subheader("MACD")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD'], name='MACD'))
        macd_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD_signal'], name='Signal'))
        macd_fig.add_trace(go.Bar(x=df['timestamp'], y=df['MACD_diff'], name='Histogram'))
        macd_fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(macd_fig, width="stretch")

    st.subheader("Latest indicator values (last closed candle)")
    st.dataframe(features.T.rename(columns={0: 'Value'}), width="stretch")

with tab_data:
    st.dataframe(df.tail(100).iloc[::-1], width="stretch", height=520)

st.caption(
    "Data: Binance public market data via ccxt. Predictions are based on the last "
    "**closed** candle. This is a research project — do not trade real funds on it."
)
