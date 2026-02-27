import pandas as pd

from strategy.strategy_engine import generate_entries


def test_entry_conditions_all_required():
    df = pd.DataFrame(
        {
            "ai_prob": [0.71, 0.90, 0.65],
            "trend_1h": [1, -1, 1],
            "trend_4h": [1, 1, 1],
            "atr_pctile": [0.4, 0.4, 0.4],
            "rr_ratio": [2.0, 2.0, 2.0],
        }
    )
    s = generate_entries(df, 0.70)
    assert s.tolist() == [1, 0, 0]
