from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class AppConfig:
    ticker: str = "SPY"
    interval: str = "1d"
    start_date: date = date(2018, 1, 1)
    end_date: date = date.today()
    horizon: int = 5
    return_threshold: float = 0.003
    prob_entry_threshold: float = 0.55
    initial_capital: float = 10000.0
    fee_pct: float = 0.0005
    slippage_pct: float = 0.0005
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    train_size: float = 0.7
    enable_walk_forward: bool = True
    long_short_mode: bool = False
    max_position_size: float = 1.0
    capital_protection_threshold: float = 0.7
    random_state: int = 42


INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "1d": "1d",
}
