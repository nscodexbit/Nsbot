from __future__ import annotations

import pandas as pd


def trend_state(close: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
    up = (close > ema_fast) & (ema_fast > ema_slow)
    down = (close < ema_fast) & (ema_fast < ema_slow)
    out = pd.Series(0, index=close.index, dtype="int8")
    out[up] = 1
    out[down] = -1
    return out


def generate_entries(df: pd.DataFrame, confidence_threshold: float = 0.70) -> pd.Series:
    cond = (
        (df["ai_prob"] >= confidence_threshold)
        & (df["trend_1h"] == 1)
        & (df["trend_4h"] == 1)
        & (df["atr_pctile"] > 0.2)
        & (df["rr_ratio"] >= 2.0)
    )
    return cond.astype("int8")


def compute_exit_levels(entry_price: float, atr_value: float, side: int = 1) -> dict:
    stop = entry_price - side * atr_value
    tp1 = entry_price + side * atr_value
    tp2 = entry_price + side * 2 * atr_value
    return {"stop": stop, "tp1": tp1, "tp2": tp2}
