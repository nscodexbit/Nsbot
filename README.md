# Nsbot — CPU-Optimized AI Quant Trading System

Production-oriented, modular quant framework for **Binance Futures** (primary) and optional equities feeds, designed for **8GB RAM / CPU-only** environments.

## Highlights
- Strict anti-leakage design (feature lagging + forward-only target creation).
- Walk-forward time-series validation with XGBoost probabilities.
- Confidence-gated strategy with multi-timeframe trend alignment.
- Risk controls: capped Kelly, daily/weekly loss guards, consecutive-loss shutdown.
- Candle-level backtesting with fees, slippage, spread, and funding.
- Monte Carlo stress testing (500 bootstrap runs + slippage noise).
- Streamlit dashboard for visual monitoring.

## Project Structure

- `data_engine/` historical + streaming ingestion, resampling/alignment, parquet storage.
- `features/` vectorized technical/quant features with strict lag.
- `models/` walk-forward XGBoost training and importance tracking.
- `ensemble/` dynamic weighted probability ensembling.
- `strategy/` signal logic + portfolio correlation/rotation helpers.
- `risk/` position sizing and account-level protections.
- `backtest/` simulation + metrics + Monte Carlo.
- `execution/` retry-safe execution primitives + encrypted secret helper + alerts.
- `dashboard/` Streamlit + Plotly views.
- `config/` YAML configuration + loader.
- `tests/` unit tests for core leakage/risk/strategy behavior.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
streamlit run dashboard/app.py
```

## Non-Negotiables Built In
- No lookahead bias (`features` are shifted by one bar before modeling).
- No data leakage (walk-forward split only, no shuffled CV).
- Monte Carlo stress required in pipeline output (`data/monte_carlo.csv`).
- Risk-off behavior supported by `RiskManager.can_trade` guardrails.

## Notes
- Default config targets Binance Futures symbols and 5m/1h/4h stack.
- Keep data history capped to 2 years to respect RAM constraints.
- Set secrets via `.env` and environment variables in deployment.

### Web UI

```bash
make webui
```

Then open `http://localhost:8501`.

## Institutional AI Trading GUI (Research/Backtesting)

A standalone modular implementation is available in `ai_trading_gui/`.

```bash
streamlit run ai_trading_gui/app.py
```

It includes:
- yfinance OHLCV ingestion with validation
- 300+ engineered micro indicators
- XGBoost classification with time-series split + randomized search
- optional walk-forward validation
- vectorized backtester with fees/slippage/SL/TP and capital protection
- interactive Plotly dashboard with exports
