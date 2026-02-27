from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from pathlib import Path

from backtest.backtester import BacktestConfig, Backtester, export_trade_log, monte_carlo_bootstrap, performance_metrics
from config.settings import Settings
from data_engine.data_engine import DataEngine
from features.feature_engineering import build_features, correlation_filter
from models.model_trainer import make_target, top_features_by_importance, walk_forward_xgb
from strategy.strategy_engine import generate_entries, trend_state

logging.basicConfig(level=logging.INFO)


def to_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def run() -> None:
    settings = Settings.from_yaml()
    data_cfg = settings.section("data")
    model_cfg = settings.section("model")
    risk_cfg = settings.section("risk")
    exec_cfg = settings.section("execution")

    engine = DataEngine(parquet_dir=Path(data_cfg["parquet_dir"]))

    symbol = data_cfg["symbols"][0]
    base = engine.fetch_binance_klines(symbol, "5m", to_ms(data_cfg["start"]), to_ms(data_cfg["end"]))
    aligned = engine.resample_align(base, data_cfg["timeframes"])

    merged = aligned["5m"].join(aligned["1h"], how="left").join(aligned["4h"], how="left").ffill().dropna()
    feat = build_features(base)

    y = make_target(merged["close"], model_cfg["target_horizon_bars"], model_cfg["target_threshold"])
    feat = feat.reindex(merged.index).dropna()
    y = y.reindex(feat.index)

    filtered_cols = correlation_filter(feat)
    wf = walk_forward_xgb(feat[filtered_cols], y, model_cfg["xgboost"], n_splits=model_cfg["walk_forward_splits"])
    top_cols = top_features_by_importance(wf.feature_importance, 25)

    out = merged.loc[wf.oos_probs.index].copy()
    out["ai_prob"] = wf.oos_probs
    out["atr_pctile"] = feat.loc[out.index, "atr_pctile"] if "atr_pctile" in feat else 0.5
    out["atr14"] = feat.loc[out.index, "atr14"] if "atr14" in feat else out["close"].pct_change().rolling(14).std()
    out["trend_1h"] = trend_state(out["close_1h"], out["close_1h"].ewm(span=20, adjust=False).mean(), out["close_1h"].ewm(span=50, adjust=False).mean()) if "close_1h" in out else 1
    out["trend_4h"] = trend_state(out["close_4h"], out["close_4h"].ewm(span=20, adjust=False).mean(), out["close_4h"].ewm(span=50, adjust=False).mean()) if "close_4h" in out else 1
    out["rr_ratio"] = 2.0
    out["signal"] = generate_entries(out, model_cfg["confidence_threshold"])

    bt = Backtester(
        BacktestConfig(
            fee_bps=exec_cfg["fee_bps"],
            slippage_bps=exec_cfg["slippage_bps"],
            spread_bps=exec_cfg["spread_bps"],
            funding_rate_per_8h=risk_cfg["funding_rate_per_8h"],
        )
    )
    trades, curve = bt.run(out, initial_equity=risk_cfg["account_equity"])
    metrics = performance_metrics(curve, trades)
    mc = monte_carlo_bootstrap(curve.pct_change())

    out["equity"] = curve.reindex(out.index).ffill()
    out.to_csv("data/backtest_merged.csv")
    export_trade_log(trades, "data/trades.csv")
    mc.to_csv("data/monte_carlo.csv", index=False)

    logging.info("Top features: %s", top_cols)
    logging.info("Fold scores: %s", wf.fold_scores)
    logging.info("Metrics: %s", metrics)


if __name__ == "__main__":
    run()
