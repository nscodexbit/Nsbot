from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_corr_filter(price_map: dict[str, pd.Series], max_corr: float = 0.8, window: int = 288) -> list[str]:
    returns = pd.DataFrame({k: v.pct_change() for k, v in price_map.items()}).dropna()
    if returns.empty:
        return list(price_map.keys())
    corr = returns.tail(window).corr().abs()
    selected: list[str] = []
    for asset in corr.columns:
        if all(corr.loc[asset, s] < max_corr for s in selected):
            selected.append(asset)
    return selected


def allocate_equal_risk(confidence: pd.Series, max_assets: int = 5) -> pd.Series:
    top = confidence.sort_values(ascending=False).head(max_assets)
    if top.empty:
        return top
    weights = pd.Series(1.0 / len(top), index=top.index)
    return weights
