from __future__ import annotations

import numpy as np
import pandas as pd


def dynamic_weighted_ensemble(
    primary_prob: pd.Series,
    secondary_prob: pd.Series | None,
    y_true: pd.Series,
    window: int = 300,
) -> pd.Series:
    if secondary_prob is None:
        return primary_prob

    p1_hit = ((primary_prob >= 0.5).astype(int) == y_true).astype(float)
    p2_hit = ((secondary_prob >= 0.5).astype(int) == y_true).astype(float)

    w1 = p1_hit.rolling(window).mean().fillna(0.5)
    w2 = p2_hit.rolling(window).mean().fillna(0.5)
    denom = (w1 + w2).replace(0, np.nan).fillna(1.0)

    return ((w1 * primary_prob + w2 * secondary_prob) / denom).astype("float32")
