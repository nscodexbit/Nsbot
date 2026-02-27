from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from risk.risk_manager import compute_drawdown
from strategy.strategy_engine import compute_exit_levels


@dataclass
class BacktestConfig:
    fee_bps: float
    slippage_bps: float
    spread_bps: float
    funding_rate_per_8h: float


class Backtester:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self, df: pd.DataFrame, initial_equity: float = 10_000.0) -> tuple[pd.DataFrame, pd.Series]:
        equity = initial_equity
        position = 0
        entry_price = 0.0
        qty = 0.0
        trade_log = []
        equity_curve = []

        for ts, row in df.iterrows():
            px = float(row["close"])
            fee_mult = (self.config.fee_bps + self.config.slippage_bps + self.config.spread_bps) / 10_000

            if position == 0 and row.get("signal", 0) == 1:
                risk_amount = equity * 0.01
                stop = px - row["atr14"]
                qty = risk_amount / max(px - stop, 1e-6)
                entry_price = px * (1 + fee_mult)
                position = 1
                exits = compute_exit_levels(entry_price, row["atr14"], side=1)
                trade_log.append({"timestamp": ts, "event": "entry", "price": entry_price, "qty": qty})
            elif position == 1:
                funding_fee = qty * px * self.config.funding_rate_per_8h / 96
                if px <= exits["stop"] or row.get("trend_1h", 1) < 0:
                    exit_px = px * (1 - fee_mult)
                    pnl = qty * (exit_px - entry_price) - funding_fee
                    equity += pnl
                    trade_log.append({"timestamp": ts, "event": "stop_exit", "price": exit_px, "qty": qty, "pnl": pnl})
                    position = 0
                elif px >= exits["tp2"]:
                    exit_px = px * (1 - fee_mult)
                    pnl = qty * (exit_px - entry_price) - funding_fee
                    equity += pnl
                    trade_log.append({"timestamp": ts, "event": "tp_exit", "price": exit_px, "qty": qty, "pnl": pnl})
                    position = 0

            equity_curve.append((ts, equity))

        trades = pd.DataFrame(trade_log)
        curve = pd.Series(dict(equity_curve), dtype="float64").sort_index()
        return trades, curve


def performance_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
    ret = equity_curve.pct_change().dropna()
    dd = compute_drawdown(equity_curve)
    wins = trades[trades.get("pnl", 0) > 0]
    losses = trades[trades.get("pnl", 0) <= 0]

    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365 / max(len(ret), 1)) - 1
    sharpe = np.sqrt(365) * ret.mean() / (ret.std() + 1e-9)
    sortino = np.sqrt(365) * ret.mean() / (ret[ret < 0].std() + 1e-9)
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum() + 1e-9) if "pnl" in trades else 0.0

    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd.min()),
        "profit_factor": float(pf),
        "win_rate": float((trades.get("pnl", pd.Series(dtype=float)) > 0).mean() if len(trades) else 0),
        "expectancy": float(trades.get("pnl", pd.Series(dtype=float)).mean() if len(trades) else 0),
    }


def monte_carlo_bootstrap(returns: pd.Series, sims: int = 500, noise_std: float = 0.0005) -> pd.DataFrame:
    arr = returns.dropna().to_numpy()
    if arr.size == 0:
        return pd.DataFrame()
    out = []
    for i in range(sims):
        sampled = np.random.choice(arr, size=arr.size, replace=True)
        sampled = sampled + np.random.normal(0, noise_std, size=arr.size)
        equity = (1 + sampled).cumprod()
        out.append({"sim": i, "final_return": float(equity[-1] - 1), "max_dd": float(((equity / np.maximum.accumulate(equity)) - 1).min())})
    return pd.DataFrame(out)


def export_trade_log(trades: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(path, index=False)
