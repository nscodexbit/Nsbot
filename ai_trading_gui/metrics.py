from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_gui.utils import annualization_factor


def compute_performance_metrics(bt: pd.DataFrame, interval: str) -> dict:
    returns = bt["strategy_returns"].fillna(0.0)
    equity = bt["equity"]

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_factor = annualization_factor(interval)
    years = max(len(bt) / ann_factor, 1 / ann_factor)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    vol = returns.std(ddof=0)
    sharpe = (returns.mean() / vol) * np.sqrt(ann_factor) if vol > 0 else 0.0

    downside = returns[returns < 0].std(ddof=0)
    sortino = (returns.mean() / downside) * np.sqrt(ann_factor) if downside > 0 else 0.0

    max_dd = bt["drawdown"].min()

    trade_returns = bt.loc[bt["trade"] == 1, "strategy_returns"]
    wins = (trade_returns > 0).sum()
    losses = (trade_returns < 0).sum()
    win_rate = wins / len(trade_returns) if len(trade_returns) else 0.0

    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = -trade_returns[trade_returns < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Number of Trades": int(len(trade_returns)),
    }
