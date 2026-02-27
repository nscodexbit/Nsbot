from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="CPU Quant Trading Dashboard", layout="wide")
st.title("🧠 CPU-Optimized AI Quant Trading Dashboard")

uploaded = st.file_uploader("Upload merged backtest CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["timestamp"]).set_index("timestamp")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                )
            ]
        )
        for ma in ["ema20", "ema50", "ema200"]:
            if ma in df:
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Latest AI probability", f"{df['ai_prob'].iloc[-1]:.2%}")
        st.metric("Latest confidence", f"{df['confidence'].iloc[-1]:.2%}" if "confidence" in df else "N/A")
        st.line_chart(df[[c for c in ["ai_prob", "rolling_sharpe", "atr_pctile"] if c in df]].dropna())

    st.subheader("Equity & Drawdown")
    if "equity" in df:
        st.line_chart(df[["equity"]])
        dd = (df["equity"] / df["equity"].cummax() - 1.0)
        st.area_chart(dd)

    st.subheader("Trade History")
    st.dataframe(df[df.get("signal", 0) == 1].tail(100))
else:
    st.info("Upload a CSV with OHLCV + model columns to view dashboard.")
