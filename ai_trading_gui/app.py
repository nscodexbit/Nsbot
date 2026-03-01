from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve

from ai_trading_gui.backtester import run_backtest
from ai_trading_gui.data_loader import load_market_data
from ai_trading_gui.features import build_dataset
from ai_trading_gui.metrics import compute_performance_metrics
from ai_trading_gui.model import train_xgb_model, walk_forward_validation
from ai_trading_gui.utils import set_seed


st.set_page_config(page_title="Institutional AI Trading Bot", layout="wide")
st.title("🧠 Institutional AI Trading Research Dashboard")
st.caption("Offline research and backtesting system (no broker API).")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker", value="SPY")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=4)
    start_date = st.date_input("Start Date", value=date(2018, 1, 1))
    end_date = st.date_input("End Date", value=date.today())

    horizon = st.number_input("Prediction Horizon (candles)", min_value=1, max_value=120, value=5)
    return_threshold = st.number_input("Return Threshold", value=0.003, format="%.4f")
    prob_threshold = st.number_input("Probability Entry Threshold", value=0.55, format="%.2f")

    initial_capital = st.number_input("Initial Capital", value=10000.0)
    fee_pct = st.number_input("Fee %", value=0.0005, format="%.4f")
    slippage_pct = st.number_input("Slippage %", value=0.0005, format="%.4f")
    stop_loss = st.number_input("Stop Loss %", value=0.02, format="%.3f")
    take_profit = st.number_input("Take Profit %", value=0.04, format="%.3f")

    train_test_split = st.slider("Train/Test Split", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    enable_walk_forward = st.checkbox("Enable Walk Forward", value=True)
    long_short_mode = st.checkbox("Long/Short Mode", value=False)

    max_position_size = st.slider("Max Position Size", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    capital_protection = st.slider("Capital Protection Threshold", min_value=0.2, max_value=1.0, value=0.7, step=0.05)

    run_button = st.button("Run Model", type="primary")

if run_button:
    set_seed(42)
    try:
        with st.spinner("Downloading market data..."):
            data = load_market_data(ticker, str(start_date), str(end_date), timeframe)

        with st.spinner("Engineering 300+ features and building target..."):
            x, y, aligned_close = build_dataset(data, int(horizon), float(return_threshold))

        st.success(f"Dataset ready: {len(x):,} rows, {x.shape[1]} features.")

        with st.spinner("Training XGBoost model..."):
            artifacts = train_xgb_model(x, y, float(train_test_split), 42)

        wf_df = pd.DataFrame()
        if enable_walk_forward:
            with st.spinner("Running walk-forward validation..."):
                wf_df = walk_forward_validation(x, y, float(train_test_split), 42)

        test_close = aligned_close.loc[artifacts.test_index]
        bt, trades = run_backtest(
            close=test_close,
            proba=artifacts.test_proba,
            entry_threshold=float(prob_threshold),
            initial_capital=float(initial_capital),
            fee_pct=float(fee_pct),
            slippage_pct=float(slippage_pct),
            stop_loss_pct=float(stop_loss),
            take_profit_pct=float(take_profit),
            max_position_size=float(max_position_size),
            capital_protection_threshold=float(capital_protection),
            long_short_mode=long_short_mode,
        )
        perf = compute_performance_metrics(bt, "60m" if timeframe == "1h" else timeframe)

        charts_tab, perf_tab, model_tab, export_tab = st.tabs(["📈 Charts", "📊 Performance", "🧠 Model", "📁 Export"])

        with charts_tab:
            price_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
            price_fig.add_trace(go.Scatter(x=bt.index, y=bt["close"], mode="lines", name="Price"), row=1, col=1)
            buys = bt[(bt["position"] > 0) & (bt["trade"] == 1)]
            sells = bt[(bt["position"] <= 0) & (bt["trade"] == 1)]
            price_fig.add_trace(go.Scatter(x=buys.index, y=buys["close"], mode="markers", marker=dict(color="green", size=8), name="Buy"), row=1, col=1)
            price_fig.add_trace(go.Scatter(x=sells.index, y=sells["close"], mode="markers", marker=dict(color="red", size=8), name="Sell"), row=1, col=1)
            price_fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], mode="lines", name="Equity", line=dict(color="royalblue")), row=2, col=1)
            price_fig.update_layout(height=700, title="Price/Signals + Equity Curve")
            st.plotly_chart(price_fig, use_container_width=True)

            dd_fig = go.Figure()
            dd_fig.add_trace(go.Scatter(x=bt.index, y=bt["drawdown"], fill="tozeroy", name="Drawdown"))
            dd_fig.update_layout(height=300, title="Drawdown Curve")
            st.plotly_chart(dd_fig, use_container_width=True)

            y_test = y.loc[artifacts.test_index]
            if y_test.nunique() > 1:
                fpr, tpr, _ = roc_curve(y_test, artifacts.test_proba)
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                roc_fig.update_layout(height=350, title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.warning("ROC curve is unavailable because the test set has a single class.")

            fi_top = artifacts.feature_importance.head(20)
            fi_fig = go.Figure(go.Bar(x=fi_top["importance"][::-1], y=fi_top["feature"][::-1], orientation="h"))
            fi_fig.update_layout(height=550, title="Top 20 Feature Importances")
            st.plotly_chart(fi_fig, use_container_width=True)

        with perf_tab:
            c1, c2, c3, c4 = st.columns(4)
            cols = [c1, c2, c3, c4]
            for i, (k, v) in enumerate(perf.items()):
                if isinstance(v, float):
                    cols[i % 4].metric(k, f"{v:,.4f}")
                else:
                    cols[i % 4].metric(k, f"{v}")

            st.subheader("Classification Metrics")
            st.json(artifacts.metrics)
            st.write("Confusion Matrix")
            st.dataframe(pd.DataFrame(artifacts.confusion_matrix))
            if enable_walk_forward and not wf_df.empty:
                st.subheader("Walk-Forward Summary")
                st.dataframe(wf_df)

        with model_tab:
            st.write("Model Parameters")
            st.json(artifacts.best_params)
            st.write(f"Feature count: **{x.shape[1]}**")
            st.write(f"Train size: **{int(len(x)*train_test_split)}** | Test size: **{len(x)-int(len(x)*train_test_split)}**")
            st.dataframe(artifacts.feature_importance.head(20))

        with export_tab:
            pred_df = pd.DataFrame(
                {
                    "timestamp": artifacts.test_proba.index,
                    "probability": artifacts.test_proba.values,
                    "actual": y.loc[artifacts.test_proba.index].values,
                }
            )
            st.download_button(
                "Download predictions CSV",
                data=pred_df.to_csv(index=False).encode(),
                file_name=f"{ticker}_predictions.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download trade log CSV",
                data=trades.reset_index().to_csv(index=False).encode(),
                file_name=f"{ticker}_trades.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download feature importance CSV",
                data=artifacts.feature_importance.to_csv(index=False).encode(),
                file_name=f"{ticker}_feature_importance.csv",
                mime="text/csv",
            )

    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:  # broad catch for GUI safety
        st.exception(exc)
else:
    st.info("Configure parameters in the sidebar and click **Run Model**.")
