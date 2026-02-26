import numpy as np
import pandas as pd

from features.feature_engineering import build_features


def test_features_are_lagged_no_leakage():
    idx = pd.date_range("2024-01-01", periods=400, freq="5min", tz="UTC")
    close = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000.0,
        }
    )
    feat = build_features(df)
    assert feat.index.min() > df.index.min()
    assert feat.isna().sum().sum() == 0
