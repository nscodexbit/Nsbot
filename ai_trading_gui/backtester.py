from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    close: pd.Series,
    proba: pd.Series,
    entry_threshold: float,
    initial_capital: float,
    fee_pct: float,
    slippage_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_position_size: float,
    capital_protection_threshold: float,
    long_short_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame({"close": close, "proba": proba}).dropna().copy()
    ret = df["close"].pct_change().fillna(0.0)

    long_signal = (df["proba"] >= entry_threshold).astype(int)
    if long_short_mode:
        short_signal = (df["proba"] <= (1 - entry_threshold)).astype(int) * -1
        signal = long_signal + short_signal
    else:
        signal = long_signal

    position = signal.shift(1).fillna(0.0).clip(-max_position_size, max_position_size)
    trade = (position != position.shift(1)).astype(int).fillna(0)

    strategy_ret = position * ret
    strategy_ret -= trade * (fee_pct + slippage_pct)

    # Approximate stop/take at candle level
    strategy_ret = np.where(strategy_ret < -stop_loss_pct, -stop_loss_pct, strategy_ret)
    strategy_ret = np.where(strategy_ret > take_profit_pct, take_profit_pct, strategy_ret)

    equity = initial_capital * (1 + pd.Series(strategy_ret, index=df.index)).cumprod()
    protection_floor = initial_capital * capital_protection_threshold
    equity = equity.clip(lower=protection_floor)

    peak = equity.cummax()
    drawdown = equity / peak - 1

    bt = pd.DataFrame(
        {
            "close": df["close"],
            "proba": df["proba"],
            "position": position,
            "returns": ret,
            "strategy_returns": strategy_ret,
            "equity": equity,
            "drawdown": drawdown,
            "trade": trade,
        },
        index=df.index,
    )

    trades = bt[bt["trade"] == 1].copy()
    trades["equity_change"] = bt["equity"].diff().reindex(trades.index)
    return bt, trades
