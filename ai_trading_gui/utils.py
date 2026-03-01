from __future__ import annotations

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def annualization_factor(interval: str) -> int:
    mapping = {
        "1m": 252 * 390,
        "5m": 252 * 78,
        "15m": 252 * 26,
        "60m": 252 * 6,
        "1d": 252,
    }
    return mapping.get(interval, 252)


def safe_div(a: pd.Series | float, b: pd.Series | float, fill: float = 0.0):
    result = a / b
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(fill)
    if np.isfinite(result):
        return result
    return fill
