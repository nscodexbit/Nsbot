from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    daily_pnl: float = 0.0
    weekly_drawdown: float = 0.0
    consecutive_losses: int = 0
    enabled: bool = True
    open_trades: int = 0
    returns: list[float] = field(default_factory=list)


@dataclass
class RiskManager:
    risk_per_trade_pct: float
    kelly_cap_pct: float
    max_daily_loss_pct: float
    max_weekly_drawdown_pct: float
    max_open_trades: int
    disable_after_losses: int
    min_rolling_sharpe: float
    maintenance_margin_rate: float

    def position_size(self, equity: float, entry: float, stop: float, win_prob: float, rr: float, atr_pct: float) -> float:
        risk_amount = equity * self.risk_per_trade_pct
        stop_distance = abs(entry - stop) + 1e-9
        base_qty = risk_amount / stop_distance

        kelly = max(0.0, min((win_prob * (rr + 1) - 1) / rr, self.kelly_cap_pct))
        vol_adj = max(0.25, 1 - atr_pct)
        return float(base_qty * (1 + kelly) * vol_adj)

    def liquidation_price(self, entry: float, leverage: float, side: int) -> float:
        if side > 0:
            return entry * (1 - (1 / leverage) + self.maintenance_margin_rate)
        return entry * (1 + (1 / leverage) - self.maintenance_margin_rate)

    def can_trade(self, state: RiskState) -> bool:
        rolling_sharpe = self._rolling_sharpe(state.returns)
        checks = [
            state.daily_pnl > -state.equity * self.max_daily_loss_pct,
            state.weekly_drawdown < self.max_weekly_drawdown_pct,
            state.consecutive_losses < self.disable_after_losses,
            state.open_trades < self.max_open_trades,
            rolling_sharpe >= self.min_rolling_sharpe,
        ]
        state.enabled = all(checks)
        return state.enabled

    @staticmethod
    def _rolling_sharpe(returns: list[float], window: int = 200) -> float:
        if len(returns) < 30:
            return 0.0
        sample = np.array(returns[-window:], dtype=float)
        std = sample.std()
        if std == 0:
            return 0.0
        return float(np.sqrt(365) * sample.mean() / std)


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak
